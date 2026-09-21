#include "tripwire_counter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>

namespace {
constexpr int kRearmGapFrames = 12;
constexpr int kStateKeepFrames = 240;
constexpr int kRegionConfirmFrames = 2;
constexpr int kSideGapFrames = 6;
constexpr int kLostExitConfirmFrames = 3;
constexpr int kConfirmFrames = 3;
// 同一 track 两次计数事件的最小间隔。闸机通道实测正向穿越 >=2s，
// 而检测框抖动导致的连发间隔仅 50~70ms（OUT->RETRO->OUT），此门限可直接滤除。
constexpr int64_t kEventCooldownMs = 800;
// ---- 遮挡脚底校正门限 ----
// 用户 3 人实测发现：门口遮挡时检测框底部从脚底跳到胸口（单帧上跳
// 100~300px），遮挡结束又落回脚底，脚底点大幅垂直跳动反复穿越绊线区域，
// 造成 OUT/RETRO 连发。ByteTrack 遮挡期间保持同一 ID，故用 per-ID 地面基准：
// 新观测脚底较基准上跳超过 max(kFootLiftMinPx, kFootLiftBoxRatio*框高)
// 判定为遮挡抬起，脚底 y 冻结在基准上（x 仍跟随胸部框），恢复后自动解冻。
constexpr float kFootLiftMinPx = 40.0f;    // 小于此上跳视为正常行走/抖动
constexpr float kFootLiftBoxRatio = 0.30f; // 上跳超过 30% 框高必非行走
constexpr int kCandidateMaxFrames = 12;
constexpr int kMinInwardSteps = 2;
constexpr float kRegionGapTolerance = 10.0f;
constexpr float kMinBirthMovement = 24.0f;
constexpr float kMinTransitionMovement = 12.0f;
constexpr float kMinDepartureMovement = 4.0f;
constexpr float kMinLostDepartureMovement = 8.0f;
constexpr float kDepartureCancelInwardMovement = 24.0f;
constexpr float kMinBirthConfidence = 0.36f;
constexpr float kCrossLineTolerance = 5.0f;
// ---- ByteTrack ID 绑定 ----
// ID 交接：遮挡等原因使 ByteTrack 杀死旧轨迹并在附近生成新 ID 时，新 ID 在
// kHandoverMaxLostFrames 内、且出生位置距旧轨迹最后位置 < kHandoverRadius，
// 即继承旧 ID 的计数上下文（等价于旧版跨线状态机的"就近合并"）。
constexpr int kHandoverMaxLostFrames = 30;
constexpr float kHandoverRadius = 180.0f;
// 终审结算：轨迹消亡（kSettleLostFrames 内无任何检测）且已 IN 未 OUT 时，
// 判定此人已从出口离开，补一条 OUT，堵住"门口遮挡吞掉离场"造成的漏检。
constexpr int kSettleLostFrames = 90;
constexpr float kSettleMergeRadius = 200.0f;
}

TripwireCounter::TripwireCounter(const Tripwire& entry_line,
                                 const Tripwire& exit_line,
                                 float entry_depth,
                                 float exit_depth,
                                 const cv::Mat& detection_mask)
    : entry_line_(entry_line),
      exit_line_(exit_line),
      entry_depth_(entry_depth),
      exit_depth_(exit_depth),
      detection_mask_(detection_mask.clone()),
      entry_zone_(zonePolygon(entry_line, entry_depth)),
      exit_zone_(zonePolygon(exit_line, exit_depth)) {
    if (!detection_mask_.empty()) {
        cv::Mat mask_copy = detection_mask_.clone();
        cv::findContours(mask_copy, detection_contours_, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
    }
}

cv::Point2f TripwireCounter::inwardNormal(const Tripwire& tw, float depth) const {
    cv::Point2f dir = tw.p2 - tw.p1;
    float length = std::sqrt(dir.dot(dir));
    if (length < 1.0f) return cv::Point2f(0.0f, 0.0f);
    cv::Point2f normal(dir.y / length, -dir.x / length);
    return normal * (depth >= 0.0f ? 1.0f : -1.0f);
}

float TripwireCounter::signedDist(const cv::Point2f& pt, const Tripwire& tw,
                                  const cv::Point2f& normal) const {
    return (pt - tw.p1).dot(normal);
}

std::vector<cv::Point2f> TripwireCounter::zonePolygon(const Tripwire& tw,
                                                       float depth) const {
    if (!tw.isValid() || std::abs(depth) < 1.0f) return {};
    cv::Point2f offset = inwardNormal(tw, depth) * std::abs(depth);
    return {tw.p1, tw.p2, tw.p2 + offset, tw.p1 + offset};
}

bool TripwireCounter::pointInPolygon(const cv::Point2f& pt,
                                     const std::vector<cv::Point2f>& polygon,
                                     float tolerance) const {
    return polygon.size() >= 3 &&
           cv::pointPolygonTest(polygon, pt, true) >= -tolerance;
}

bool TripwireCounter::pointInMask(const cv::Point2f& pt, float tolerance) const {
    if (detection_mask_.empty()) return true;
    for (const auto& contour : detection_contours_) {
        if (contour.size() >= 3 &&
            cv::pointPolygonTest(contour, pt, true) >= -tolerance) {
            return true;
        }
    }
    return false;
}

int TripwireCounter::regionOf(const cv::Point2f& pt) const {
    if (!entry_line_.isValid() || !exit_line_.isValid()) return 0;

    // Side zones have priority over the detection polygon at shared borders.
    // The tolerance bridges the small gap left by manually drawn polygons.
    if (pointInPolygon(pt, entry_zone_, kRegionGapTolerance)) return 1;
    if (pointInPolygon(pt, exit_zone_, kRegionGapTolerance)) return 3;
    if (pointInMask(pt, kRegionGapTolerance)) return 2;
    return 0;
}

std::vector<TripwireDebugEvent> TripwireCounter::takeDebugEvents() {
    std::vector<TripwireDebugEvent> result;
    result.swap(debug_events_);
    return result;
}

std::vector<TripwireCountEvent> TripwireCounter::update(
    const std::vector<TrackedObject>& tracks) {
    debug_events_.clear();
    std::vector<TripwireCountEvent> events;
    std::set<int> observed_ids;
    const cv::Point2f entry_normal = inwardNormal(entry_line_, entry_depth_);
    const cv::Point2f exit_normal = inwardNormal(exit_line_, exit_depth_);
    auto trace = [&](int track_id, const char* reason, const State& state,
                     int raw_region, float movement = 0.0f) {
        if (!debug_enabled_) return;
        TripwireDebugEvent event;
        event.track_id = track_id;
        event.reason = reason;
        event.raw_region = raw_region;
        event.stable_region = state.stable_region;
        event.pending_region = state.pending_region;
        event.pending_frames = state.pending_frames;
        event.approach_side = state.approach_side;
        event.approach_observations = state.approach_observed_frames;
        event.departure_side = state.departure_side;
        event.departure_observations = state.departure_observed_frames;
        event.inside_committed = state.inside_committed;
        event.observed_streak = state.observed_streak;
        event.hit_streak = state.last_hit_streak;
        event.lost_frames = state.lost_frames;
        event.confidence = state.last_confidence;
        event.movement = movement;
        event.foot = state.last_foot;
        debug_events_.push_back(event);
    };

    // 事件冷却：同一 track 在 kEventCooldownMs 内最多出一个计数事件。
    // 检测框在绊线附近抖动会让 segmentCrosses 补偿在 50~70ms 内连发
    // OUT->RETRO->OUT，物理上不可能，用此门限直接滤除。
    auto nowMs = []() -> int64_t {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    auto emitAllowed = [](const State& st, int64_t now_ms) -> bool {
        return st.last_event_ms == 0 ||
               (now_ms - st.last_event_ms) >= kEventCooldownMs;
    };

    for (const auto& track : tracks) {
        // Predicted boxes preserve track continuity but never alter count state.
        if (!track.observed) continue;
        observed_ids.insert(track.track_id);

        const cv::Point2f foot_raw(track.box.x + track.box.width / 2.0f,
                                   track.box.y + track.box.height * 0.92f);
        auto state_it = states_.find(track.track_id);
        const bool is_new_track = state_it == states_.end();
        if (is_new_track)
            state_it = states_.emplace(track.track_id, State{}).first;
        State& state = state_it->second;
        const int64_t now_ms = track.capture_timestamp_ms > 0
            ? track.capture_timestamp_ms : nowMs();

        // Never let an old asynchronous result move an existing trajectory
        // backward in time.  The tracker must only see monotonic frame IDs.
        if (track.result_frame_id >= 0 &&
            track.result_frame_id <= state.last_result_frame_id) {
            trace(track.track_id, "out-of-order-result-ignored", state, -1);
            continue;
        }

        if (state.lost_frames >= kRearmGapFrames) {
            const int gap_frames = state.lost_frames;
            state.trail.clear();
            state.observed_streak = 0;
            state.stable_region = 0;
            state.pending_region = -1;
            state.pending_frames = 0;
            state.approach_side = 0;
            state.approach_observed_frames = 0;
            state.approach_gap_frames = 0;
            state.inside_committed = false;
            state.has_inside_foot = false;
            state.departure_side = 0;
            state.departure_observed_frames = 0;
            state.candidate_active = false;
            // Do not run birth compensation twice for the same track ID.
            state.baseline_ready = true;
            // 长间隙后人可能已移动一段距离，旧脚底基准不可信，重建。
            state.has_ref_foot = false;
            state.lost_frames = gap_frames;
            trace(track.track_id, "state-rearmed-after-long-gap", state, -1);
        }
        if (state.lost_frames > 0) state.observed_streak = 0;
        state.lost_frames = 0;
        state.observed_streak++;
        state.last_hit_streak = track.hit_streak;
        state.last_confidence = track.confidence;
        state.last_result_frame_id = track.result_frame_id;

        // ---- 遮挡脚底校正（per-ID 地面基准）----
        // 遮挡时框底上跳到胸口，冻结 y 到最近可靠脚底；遮挡结束后脚底落回
        // 基准附近，lift 变小或为负，自动解冻并更新基准。
        const float foot_lift_px = state.has_ref_foot
            ? state.ref_foot.y - foot_raw.y : 0.0f;
        const bool foot_lifted = state.has_ref_foot &&
            foot_lift_px > std::max(kFootLiftMinPx,
                                    kFootLiftBoxRatio * track.box.height);
        const cv::Point2f foot = foot_lifted
            ? cv::Point2f(foot_raw.x, state.ref_foot.y)
            : foot_raw;
        if (!foot_lifted) {
            state.ref_foot = foot_raw;
            state.has_ref_foot = true;
        }

        // ---- ByteTrack ID 交接（遮挡致旧 ID 死亡 + 新 ID 在附近出生）----
        // 新 ID 继承旧 ID 的计数上下文： enter/exit 已计标志、inside_committed、
        // 轨迹。旧 ID 封存（不可再发事件、不可终审结算），保证同一个物理的人
        // 跨 ID 切换仍然只有一对 IN/OUT。
        if (is_new_track) {
            int best_old_id = -1;
            float best_handover_score = -1.0f;
            for (auto& [old_id, old_state] : states_) {
                if (old_id == track.track_id) continue;
                if (old_state.lost_frames <= 0 ||
                    old_state.lost_frames > kHandoverMaxLostFrames) continue;
                if (old_state.enter_events == 0 || old_state.exit_events != 0 ||
                    !old_state.inside_committed) continue;
                const float distance = cv::norm(old_state.last_foot - foot);
                if (distance > kHandoverRadius) continue;
                if (regionOf(foot) == 0) {
                    const float ep = signedDist(old_state.last_foot, entry_line_, entry_normal);
                    const float en = signedDist(foot, entry_line_, entry_normal);
                    const float xp = signedDist(old_state.last_foot, exit_line_, exit_normal);
                    const float xn = signedDist(foot, exit_line_, exit_normal);
                    if (!((ep >= kCrossLineTolerance && en <= -kCrossLineTolerance) ||
                          (xp >= kCrossLineTolerance && xn <= -kCrossLineTolerance))) continue;
                }
                const float score = 1.0f - distance / kHandoverRadius;
                if (score <= best_handover_score) continue;
                best_handover_score = score;
                best_old_id = old_id;
            }
            if (best_old_id >= 0) {
                State& old_state = states_.at(best_old_id);
                state.enter_events = std::max(state.enter_events, old_state.enter_events);
                state.exit_events = std::max(state.exit_events, old_state.exit_events);
                state.inside_committed = old_state.inside_committed;
                state.has_inside_foot = old_state.has_inside_foot;
                state.last_inside_foot = old_state.last_inside_foot;
                state.stable_region = old_state.stable_region;
                state.trail = old_state.trail;
                state.baseline_ready = true;   // 继承旅程, 不再做出生补偿
                state.candidate_active = false;
                old_state.inside_committed = false;
                old_state.has_inside_foot = false;
                old_state.enter_events = state.enter_events;
                old_state.exit_events = state.exit_events;
                trace(track.track_id, "id-handover-inherited-context", state, -1);
            }
        }

        const bool has_previous_real = state.trail.size() >= 1;
        const cv::Point2f previous_real = has_previous_real ? state.trail.back() : foot;
        state.last_foot = foot;
        state.trail.push_back(foot);
        if (state.trail.size() > 8) state.trail.pop_front();

        const int raw_region = regionOf(foot);
        if (foot_lifted)
            trace(track.track_id, "foot-lifted-occlusion-frozen", state,
                  raw_region, foot_lift_px);
        if (is_new_track) trace(track.track_id, "track-start", state, raw_region);

        if (!state.baseline_ready) {
            state.baseline_ready = true;
            if (raw_region == 2) {
                const float entry_distance = signedDist(foot, entry_line_, entry_normal);
                const float exit_distance = signedDist(foot, exit_line_, exit_normal);
                state.candidate_active = true;
                state.candidate_side = entry_distance <= exit_distance ? 1 : 2;
                state.candidate_birth_foot = foot;
                state.candidate_last_depth = state.candidate_side == 1
                    ? entry_distance : exit_distance;
                state.candidate_frames = 1;
                state.candidate_inward_steps = 0;
                state.candidate_conf_sum = track.confidence;
                trace(track.track_id, "birth-candidate-start", state, raw_region);
            }
        } else if (state.candidate_active) {
            if (raw_region != 2) {
                state.candidate_active = false;
                trace(track.track_id, "birth-candidate-cancel-left-region2",
                      state, raw_region);
            } else {
                const Tripwire& line = state.candidate_side == 1
                    ? entry_line_ : exit_line_;
                const cv::Point2f& normal = state.candidate_side == 1
                    ? entry_normal : exit_normal;
                const float depth = signedDist(foot, line, normal);
                if (depth - state.candidate_last_depth >= 2.0f)
                    state.candidate_inward_steps++;
                state.candidate_last_depth = depth;
                state.candidate_frames++;
                state.candidate_conf_sum += track.confidence;

                const float movement = (foot - state.candidate_birth_foot).dot(normal);
                const float average_confidence = state.candidate_conf_sum /
                    std::max(1, state.candidate_frames);
                if (state.candidate_frames >= kConfirmFrames &&
                    state.observed_streak >= kConfirmFrames &&
                    state.candidate_inward_steps >= kMinInwardSteps &&
                    movement >= kMinBirthMovement &&
                    average_confidence >= kMinBirthConfidence &&
                    track.hit_streak >= kConfirmFrames &&
                    state.enter_events == 0 &&          // ID 绑定: 每人至多 1 次
                    emitAllowed(state, now_ms)) {
                    const int type = state.candidate_side == 1
                        ? TW_EVENT_IN : TW_EVENT_RETRO;
                    state.last_event_ms = now_ms;
                    state.enter_events = 1;
                    events.push_back({track.track_id, type, state.candidate_side,
                                      foot, true, average_confidence, movement});
                    state.inside_committed = true;
                    state.has_inside_foot = true;
                    state.last_inside_foot = foot;
                    state.candidate_active = false;
                    trace(track.track_id, "birth-candidate-confirmed", state,
                          raw_region, movement);
                } else if (state.candidate_frames >= kCandidateMaxFrames) {
                    state.candidate_active = false;
                    trace(track.track_id, "birth-candidate-expired", state,
                          raw_region, movement);
                }
            }
        }

        // Compensate detector drops using the line segment between two real
        // observations.  This deliberately runs before region debouncing:
        // a valid 0/1 -> 2 or 2 -> side jump need not expose every region
        // frame.  Predicted tracker boxes never reach this block because the
        // update loop skips !observed tracks above.
        auto segmentCrosses = [](const cv::Point2f& a, const cv::Point2f& b,
                                 const Tripwire& line, const cv::Point2f& n,
                                 bool inward) {
            const float da = (a - line.p1).dot(n);
            const float db = (b - line.p1).dot(n);
            if (inward ? !(da <= -kCrossLineTolerance && db >= kCrossLineTolerance)
                       : !(da >= kCrossLineTolerance && db <= -kCrossLineTolerance)) return false;
            const float denom = db - da;
            if (std::abs(denom) < 1e-3f) return false;
            const float t = -da / denom;
            if (t < 0.0f || t > 1.0f) return false;
            const cv::Point2f hit = a + (b - a) * t;
            const cv::Point2f ab = line.p2 - line.p1;
            const float len2 = ab.dot(ab);
            if (len2 < 1.0f) return false;
            const float u = (hit - line.p1).dot(ab) / len2;
            return u >= -0.05f && u <= 1.05f;
        };

        // Two real matched detections are enough for a segment: the previous
        // point supplies the required motion evidence, unlike a birth inside
        // the zone which still needs the stricter multi-frame candidate.
        // hit_streak 从 2 提到 kConfirmFrames(3)：原先离场补偿分支完全没有检测
        // 质量门限，低质量/抖动轨迹一跨线就出事件，这是连发的主要来源。
        if (has_previous_real && track.hit_streak >= kConfirmFrames &&
            track.confidence >= kMinBirthConfidence) {
            const int previous_region = regionOf(previous_real);
            const float displacement = cv::norm(foot - previous_real);
            if (!state.inside_committed && raw_region == 2 &&
                (previous_region == 0 || previous_region == 1 || previous_region == 3) &&
                displacement >= kMinTransitionMovement &&
                state.enter_events == 0 &&          // ID 绑定: 每人至多 1 次
                emitAllowed(state, now_ms)) {
                int side = 0;
                const cv::Point2f* normal = nullptr;
                if ((previous_region == 0 || previous_region == 1) &&
                    segmentCrosses(previous_real, foot, entry_line_, entry_normal, true)) {
                    side = 1;
                    normal = &entry_normal;
                } else if ((previous_region == 0 || previous_region == 3) &&
                           segmentCrosses(previous_real, foot, exit_line_, exit_normal, true)) {
                    side = 2;
                    normal = &exit_normal;
                }
                if (side != 0) {
                    const float movement = (foot - previous_real).dot(*normal);
                    const int type = side == 1 ? TW_EVENT_IN : TW_EVENT_RETRO;
                    state.last_event_ms = now_ms;
                    state.enter_events = 1;
                    events.push_back({track.track_id, type, side, foot, false,
                                      track.confidence, movement});
                    state.inside_committed = true;
                    state.has_inside_foot = true;
                    state.last_inside_foot = foot;
                    state.candidate_active = false;
                    state.approach_side = 0;
                    state.approach_observed_frames = 0;
                    trace(track.track_id, "segment-entry-compensated", state,
                          raw_region, movement);
                }
            } else if (state.inside_committed &&
                       (raw_region == 1 || raw_region == 3) &&
                       previous_region == 2 &&
                       displacement >= kMinDepartureMovement &&
                       state.exit_events == 0 &&           // ID 绑定: 每人至多 1 次
                       emitAllowed(state, now_ms)) {
                const int side = raw_region == 1 ? 1 : 2;
                const Tripwire& line = side == 1 ? entry_line_ : exit_line_;
                const cv::Point2f& normal = side == 1 ? entry_normal : exit_normal;
                if (segmentCrosses(previous_real, foot, line, normal, false)) {
                    const float movement = -(foot - previous_real).dot(normal);
                    const int type = side == 2 ? TW_EVENT_OUT : TW_EVENT_RETRO;
                    state.last_event_ms = now_ms;
                    state.exit_events = 1;
                    events.push_back({track.track_id, type, side, foot, false,
                                      track.confidence, movement});
                    state.inside_committed = false;
                    state.has_inside_foot = false;
                    state.departure_side = 0;
                    state.departure_observed_frames = 0;
                    trace(track.track_id, "segment-exit-compensated", state, raw_region,
                          movement);
                }
            } else if (state.inside_committed && raw_region == 0 &&
                       previous_region == 2 &&
                       displacement >= kMinDepartureMovement &&
                       state.exit_events == 0 &&           // ID 绑定: 每人至多 1 次
                       emitAllowed(state, now_ms)) {
                // A dropped frame may land outside both side polygons.  Accept
                // this only when the real-to-real segment crosses a tripwire;
                // a bare 2->0 classification jump remains ignored below.
                int side = 0;
                const cv::Point2f* normal = nullptr;
                if (segmentCrosses(previous_real, foot, exit_line_, exit_normal, false)) {
                    side = 2; normal = &exit_normal;
                } else if (segmentCrosses(previous_real, foot, entry_line_, entry_normal, false)) {
                    side = 1; normal = &entry_normal;
                }
                if (side != 0) {
                    const float movement = -(foot - previous_real).dot(*normal);
                    const int type = side == 2 ? TW_EVENT_OUT : TW_EVENT_RETRO;
                    state.last_event_ms = now_ms;
                    state.exit_events = 1;
                    events.push_back({track.track_id, type, side, foot, false,
                                      track.confidence, movement});
                    state.inside_committed = false;
                    state.has_inside_foot = false;
                    state.departure_side = 0;
                    state.departure_observed_frames = 0;
                    trace(track.track_id, "segment-exit-line-compensated", state,
                          raw_region, movement);
                }
            }
        }

        if (raw_region == 2 && state.inside_committed) {
            state.has_inside_foot = true;
            state.last_inside_foot = foot;
        }

        // Keep one real side-zone observation as directional evidence. The
        // destination still needs debouncing/motion confirmation before count.
        if (raw_region == 1 || raw_region == 3) {
            const int raw_side = raw_region == 1 ? 1 : 2;
            if (state.inside_committed) {
                if (state.departure_side != raw_side) {
                    state.departure_side = raw_side;
                    state.departure_observed_frames = 1;
                    state.departure_foot = state.has_inside_foot
                        ? state.last_inside_foot : foot;
                    state.departure_side_foot = foot;
                    trace(track.track_id, "departure-side-observed", state,
                          raw_region);
                } else {
                    state.departure_observed_frames++;
                    state.departure_side_foot = foot;
                }
            } else {
                if (state.approach_side != raw_side) {
                    state.approach_side = raw_side;
                    state.approach_observed_frames = 1;
                    state.approach_foot = foot;
                    state.approach_gap_frames = 0;
                    trace(track.track_id, "approach-side-observed", state,
                          raw_region);
                } else {
                    state.approach_observed_frames++;
                }
            }
        }

        if (raw_region == 0 && state.approach_side != 0) {
            if (++state.approach_gap_frames > kSideGapFrames) {
                trace(track.track_id, "approach-side-expired-in-gap", state,
                      raw_region);
                state.approach_side = 0;
                state.approach_observed_frames = 0;
                state.approach_gap_frames = 0;
            }
        } else if (raw_region != 0) {
            state.approach_gap_frames = 0;
        }

        if (raw_region == state.pending_region) {
            state.pending_frames++;
        } else {
            state.pending_region = raw_region;
            state.pending_frames = 1;
            state.pending_start_foot = foot;
        }
        if (state.pending_frames < kRegionConfirmFrames) continue;

        const int new_region = state.pending_region;
        const bool region_changed = new_region != state.stable_region;
        if (region_changed) {
            state.stable_region = new_region;
            trace(track.track_id, "stable-region-change", state, raw_region);
        }

        if (new_region == 1 || new_region == 3) {
            if (region_changed) {
                if (state.inside_committed) {
                    trace(track.track_id, "departure-side-confirmed", state,
                          raw_region);
                } else {
                    trace(track.track_id, "approach-side-confirmed", state,
                          raw_region);
                }
            }
            continue;
        }

        if (new_region == 2) {
            if (state.inside_committed) {
                // Region classification can briefly bounce 3/1 -> 2 while a
                // person is still moving out near a polygon endpoint. Cancel
                // departure only after a sustained, clearly inward return.
                if (state.departure_side != 0) {
                    const cv::Point2f& departure_normal =
                        state.departure_side == 1 ? entry_normal : exit_normal;
                    const float inward_return =
                        (foot - state.departure_side_foot).dot(departure_normal);
                    if (state.pending_frames >= kConfirmFrames &&
                        inward_return >= kDepartureCancelInwardMovement) {
                        trace(track.track_id,
                              "departure-cancelled-inward-return", state,
                              raw_region, inward_return);
                        state.departure_side = 0;
                        state.departure_observed_frames = 0;
                    } else if (region_changed) {
                        trace(track.track_id,
                              "departure-preserved-region-bounce", state,
                              raw_region, inward_return);
                    }
                }
                continue;
            }
            if (state.approach_side == 0 ||
                state.approach_gap_frames > kSideGapFrames) {
                if (region_changed)
                    trace(track.track_id, "entry-wait-no-stable-side", state,
                          raw_region);
                continue;
            }

            const cv::Point2f& normal = state.approach_side == 1
                ? entry_normal : exit_normal;
            const float movement = (foot - state.approach_foot).dot(normal);
            float recent_step = movement;
            if (state.trail.size() >= 2) {
                const int offset = std::min(
                    3, static_cast<int>(state.trail.size()) - 1);
                recent_step = (state.trail.back() -
                               state.trail[state.trail.size() - 1 - offset])
                                  .dot(normal) / offset;
            }
            if (movement >= kMinTransitionMovement && recent_step > 0.5f &&
                state.enter_events == 0 &&          // ID 绑定: 每人至多 1 次
                emitAllowed(state, now_ms)) {
                if (track.hit_streak < kConfirmFrames ||
                    track.confidence < kMinBirthConfidence) {
                    if (region_changed)
                        trace(track.track_id, "entry-wait-detection-quality",
                              state, raw_region, movement);
                    continue;
                }
                const int type = state.approach_side == 1
                    ? TW_EVENT_IN : TW_EVENT_RETRO;
                state.last_event_ms = now_ms;
                state.enter_events = 1;
                events.push_back({track.track_id, type, state.approach_side,
                                  foot, false, track.confidence, movement});
                state.inside_committed = true;
                state.has_inside_foot = true;
                state.last_inside_foot = foot;
                state.candidate_active = false;
                state.approach_side = 0;
                state.approach_observed_frames = 0;
                state.approach_gap_frames = 0;
            } else if (region_changed) {
                trace(track.track_id, "entry-wait-inward-motion", state,
                      raw_region, movement);
            }
            continue;
        }

        if (new_region == 0) {
            // A direct 2->0 transition is intentionally ignored. An outgoing
            // event requires a confirmed entry/exit side zone in between.
            if (!state.inside_committed || state.departure_side == 0) {
                if (region_changed)
                    trace(track.track_id, "outside-ignored-no-departure-side",
                          state, raw_region);
                continue;
            }

            const cv::Point2f& normal = state.departure_side == 1
                ? entry_normal : exit_normal;
            const float movement = -(foot - state.departure_foot).dot(normal);
            if (movement < kMinDepartureMovement) {
                if (region_changed)
                    trace(track.track_id, "departure-wait-outward-motion",
                          state, raw_region, movement);
                continue;
            }
            if (track.hit_streak < kConfirmFrames ||
                track.confidence < kMinBirthConfidence) {
                if (region_changed)
                    trace(track.track_id, "departure-wait-detection-quality",
                          state, raw_region, movement);
                continue;
            }
            if (!emitAllowed(state, now_ms) || state.exit_events != 0) {
                if (region_changed)
                    trace(track.track_id,
                          state.exit_events != 0 ? "departure-wait-id-once"
                                                 : "departure-wait-event-cooldown",
                          state, raw_region, movement);
                continue;
            }

            const int type = state.departure_side == 2
                ? TW_EVENT_OUT : TW_EVENT_RETRO;
            state.last_event_ms = now_ms;
            state.exit_events = 1;
            events.push_back({track.track_id, type, state.departure_side,
                              foot, false, track.confidence, movement});
            state.inside_committed = false;
            state.has_inside_foot = false;
            state.departure_side = 0;
            state.departure_observed_frames = 0;
            state.approach_side = 0;
            state.approach_observed_frames = 0;
            state.approach_gap_frames = 0;
        }
    }

    for (auto it = states_.begin(); it != states_.end();) {
        if (!observed_ids.count(it->first)) {
            State& state = it->second;
            state.lost_frames++;
            if (state.lost_frames == 1) {
                trace(it->first, "track-missing", state, -1);
            }

            if (state.lost_frames == kLostExitConfirmFrames &&
                state.inside_committed && state.departure_side != 0 &&
                state.departure_observed_frames > 0) {
                const cv::Point2f& normal = state.departure_side == 1
                    ? entry_normal : exit_normal;
                const float movement =
                    -(state.last_foot - state.departure_foot).dot(normal);
                const bool quality_ok =
                    state.last_hit_streak >= kConfirmFrames &&
                    state.last_confidence >= kMinBirthConfidence;
                const int64_t lost_now_ms = nowMs();
                if (movement >= kMinLostDepartureMovement && quality_ok &&
                    state.exit_events == 0 &&       // ID 绑定: 每人至多 1 次
                    emitAllowed(state, lost_now_ms)) {
                    const int type = state.departure_side == 2
                        ? TW_EVENT_OUT : TW_EVENT_RETRO;
                    state.last_event_ms = lost_now_ms;
                    state.exit_events = 1;
                    events.push_back({it->first, type, state.departure_side,
                                      state.last_foot, false,
                                      state.last_confidence, movement, true});
                    trace(it->first, "lost-exit-confirmed", state, -1,
                          movement);
                    state.inside_committed = false;
                    state.has_inside_foot = false;
                    state.departure_side = 0;
                    state.departure_observed_frames = 0;
                    state.approach_side = 0;
                    state.approach_observed_frames = 0;
                    state.approach_gap_frames = 0;
                } else {
                    trace(it->first,
                          quality_ok ? "lost-exit-rejected-motion"
                                     : "lost-exit-rejected-quality",
                          state, -1, movement);
                }
            }

            // ---- 终审结算：轨迹长时间无检测且"已进未出" ----
            // 门口遮挡常把离场过程整个吞掉（ByteTrack 杀死轨迹），导致 IN 已计
            // 而 OUT 永远等不到。轨迹消亡 kSettleLostFrames 后仍未出，且附近
            // 没有活跃轨迹接管（说明不是 ID 切换），补一条 OUT。
            if (state.lost_frames == kSettleLostFrames &&
                state.inside_committed && state.enter_events > 0 &&
                state.exit_events == 0 &&
                state.last_confidence >= kMinBirthConfidence) {
                bool handed_over = false;
                for (const auto& [other_id, other] : states_) {
                    if (other_id == it->first) continue;
                    if (other.lost_frames > kHandoverMaxLostFrames) continue;
                    if (cv::norm(other.last_foot - state.last_foot) <=
                        kSettleMergeRadius) {
                        handed_over = true;
                        break;
                    }
                }
                if (handed_over) {
                    trace(it->first, "settle-skipped-nearby-active", state, -1);
                } else {
                    state.exit_events = 1;
                    state.inside_committed = false;
                    events.push_back({it->first, TW_EVENT_OUT, 2,
                                      state.last_foot, false,
                                      state.last_confidence,
                                      kMinDepartureMovement, true});
                    trace(it->first, "settle-out-on-track-death", state, -1,
                          kMinDepartureMovement);
                }
            }

            if (state.lost_frames > kStateKeepFrames) {
                it = states_.erase(it);
                continue;
            }
        }
        ++it;
    }
    return events;
}
