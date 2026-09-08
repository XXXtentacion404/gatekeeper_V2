#include "gate_id_counter.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace {
// 脚底点相对框底的上移比例（框底中心向上一点，落在脚上）
constexpr float kFootRatio = 0.92f;
// 边界容差（px）：脚底在标记线附近 ±tol 视为"标记线上"，不改变侧（滞回）
constexpr float kLineTolerance = 10.0f;
// 侧切换确认帧数：候选侧连续出现该帧数才切换稳定侧（防单帧抖动词跃）
constexpr int kSideConfirmFrames = 2;
// 离开：ID 从活跃输出消失后，确认"离开类"事件的帧数门限
constexpr int kLostExitFrames = 3;
// 状态保留帧数（超过则清理，防止 map 无限增长）
constexpr int kStateKeepFrames = 120;
// 检测质量门限：低置信度/刚出现的新框不参与"进入类"判定
constexpr float kMinConfidence = 0.25f;
constexpr int kMinHitStreak = 2;
}

GateIdCounter::GateIdCounter(GateRole role, const Tripwire& marker_line,
                             float entry_depth, float exit_depth)
    : role_(role),
      marker_line_(marker_line),
      entry_depth_(std::abs(entry_depth)),
      exit_depth_(std::abs(exit_depth)) {
    // 入侧单位法向量，方向由 entry_depth 符号决定（与旧 inwardNormal 约定一致：
    // depth > 0 时入侧在 normal=(dir.y,-dir.x)/L 方向，depth < 0 时反向）。
    cv::Point2f dir = marker_line_.p2 - marker_line_.p1;
    float length = std::sqrt(dir.dot(dir));
    cv::Point2f normal = length < 1.0f
        ? cv::Point2f(1.0f, 0.0f)
        : cv::Point2f(dir.y / length, -dir.x / length);
    inward_n_ = normal * (entry_depth >= 0.0f ? 1.0f : -1.0f);
    if (entry_depth_ < 1.0f) entry_depth_ = 100.0f;
    if (exit_depth_ < 1.0f) exit_depth_ = 50.0f;
}

float GateIdCounter::signedDist(const cv::Point2f& pt) const {
    if (!marker_line_.isValid()) return 0.0f;
    return (pt - marker_line_.p1).dot(inward_n_);
}

int GateIdCounter::sideOf(const cv::Point2f& pt) const {
    if (!marker_line_.isValid()) return 0;
    const float d = signedDist(pt);
    if (d > kLineTolerance) return 1;   // 入侧
    if (d < -kLineTolerance) return 2;  // 出侧
    return 0;                           // 标记线附近
}

void GateIdCounter::reset() { states_.clear(); }

std::vector<GateCountEvent> GateIdCounter::update(
    const std::vector<TrackedObject>& tracks) {
    std::vector<GateCountEvent> events;
    std::set<int> active_ids;  // 本帧 ByteTrack 输出的所有 track（含预测框）

    for (const auto& track : tracks) {
        active_ids.insert(track.track_id);
        // 预测框(observed=false)保持轨迹连续，但不改变计数状态。
        if (!track.observed) continue;

        auto it = states_.find(track.track_id);
        if (it == states_.end())
            it = states_.emplace(track.track_id, State{}).first;
        State& state = it->second;

        // 异步推理可能乱序返回，旧结果不得让轨迹倒退回退。
        if (track.result_frame_id >= 0 &&
            track.result_frame_id <= state.last_result_frame_id)
            continue;
        state.last_result_frame_id = track.result_frame_id;
        state.lost_frames = 0;

        const cv::Point2f foot(track.box.x + track.box.width * 0.5f,
                               track.box.y + track.box.height * kFootRatio);
        state.last_foot = foot;
        const float d = signedDist(foot);

        const int raw_side = (d > kLineTolerance) ? 1
                           : (d < -kLineTolerance) ? 2
                           : 0;

        // 侧切换确认（滞回）：标记线附近(raw_side==0)不改侧并清零候选；
        // 候选侧需连续 kSideConfirmFrames 帧才切换稳定侧（计数封顶防长期溢出）。
        if (raw_side == 0) {
            state.pending_frames = 0;
        } else if (raw_side == state.pending_side) {
            if (state.pending_frames < kSideConfirmFrames)
                state.pending_frames++;
        } else {
            state.pending_side = raw_side;
            state.pending_frames = 1;
        }

        if (state.pending_side != 0 &&
            state.pending_frames >= kSideConfirmFrames &&
            state.pending_side != state.stable_side) {
            const int prev = state.stable_side;
            state.stable_side = state.pending_side;

            if (state.stable_side == 1) {
                state.was_inside = true;
                // 记录进入侧跨越时刻的检测质量，离开类用同一门限对称判定，
                // 使"进入类/离开类"漏判概率一致（杜绝 IN 漏计却仍 OUT 的不对称）。
                if (track.hit_streak >= kMinHitStreak &&
                    track.confidence >= kMinConfidence)
                    state.entry_quality_ok = true;
                // 外→内：进入类事件（201=IN，200=RETRO），每人至多一次。
                // 与离开类共用 entry_quality_ok 标记，保证两类判定口径一致。
                if (prev == 2 && !state.entered && state.entry_quality_ok) {
                    state.entered = true;
                    GateCountEvent ev;
                    ev.track_id = track.track_id;
                    ev.foot = foot;
                    ev.type = (role_ == GATE_ROLE_ENTRY)
                                  ? GATE_EVENT_IN : GATE_EVENT_RETRO;
                    events.push_back(ev);
                }
            }
        }
    }

    // 内→外消失：离开类事件（201=RETRO，200=OUT），每人至多一次。
    // 触发条件：脚底曾在入侧稳定出现过(was_inside)，当前稳定侧为出侧(stable_side==2)，
    // 且该 ID 连续多帧未再出现在活跃输出中（真正离开，而非遮挡预测）。
    for (auto it = states_.begin(); it != states_.end();) {
        State& state = it->second;
        if (active_ids.count(it->first)) { ++it; continue; }

        state.lost_frames++;
        // 离开类与进入类共用 entry_quality_ok 门限：进入侧跨越时检测质量不达标者，
        // 既不会发进入类事件，也不会发离开类事件，两类漏判概率一致。
        if (!state.exited && state.was_inside && state.stable_side == 2 &&
            state.entry_quality_ok && state.lost_frames >= kLostExitFrames) {
            state.exited = true;
            GateCountEvent ev;
            ev.track_id = it->first;
            ev.foot = state.last_foot;
            ev.type = (role_ == GATE_ROLE_ENTRY)
                          ? GATE_EVENT_RETRO : GATE_EVENT_OUT;
            events.push_back(ev);
        }

        if (state.lost_frames > kStateKeepFrames) {
            it = states_.erase(it);
        } else {
            ++it;
        }
    }

    return events;
}
