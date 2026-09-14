#include "gate_id_counter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>

namespace {
// 脚底点相对框底的上移比例（框底中心向上一点，落在脚上）
constexpr float kFootRatio = 0.92f;

// ---- 区域序列 ----
// 区域切换确认帧数（滞回）：候选区域连续出现该帧数才切换稳定区域，防单帧抖动
constexpr int kRegionConfirmFrames = 2;
// 边界容差(px)：手绘多边形与入口带之间的缝隙用该容差桥接。
// 脚底距离检测区边界 -kRegionGapTolerance 之内仍视为"在检测区内"。
constexpr float kRegionGapTolerance = 10.0f;

// ---- 本相机标记线(t1)的跨线判定（角色隔离的核心）----
// 每台相机只对自己画的那条标记线负责：脚底到 t1 的带符号距离
//   signed > 0 → 线内侧（检测区侧，即 201 入口区域 / 200 出口区域所在的一侧）
//   signed < 0 → 线外侧（真·外面：201 的入口外 / 200 的出口外）
// 只有"确实跨过本相机的线"才可能产生事件：
//   - 进入类：曾在线外侧(signed ≤ -kCrossLineDepth) → 现在进到线内侧
//   - 离开类：现在到了线外侧(signed ≤ -kCrossLineDepth)
// 这样"人从检测区远端（对面相机那一侧）进出"在本相机上 signed 全程为正，
// 既不会误报进入类，也不会误报离开类。
// 取值需大于脚底抖动幅度，避免在线上抖动反复跨界。
constexpr float kCrossLineDepth = 20.0f;

// ---- 侧翻转边沿检测（唯一触发源）----
// 候选侧连续出现该帧数才确认翻转并产生一次事件（滞回 + 时间确认，防抖）
constexpr int kLineSideConfirmFrames = 3;
// 翻转后的冷却帧数：一次翻转后短时间内不再接受反向翻转，杜绝线上抖动连发
constexpr int kLineFlipCooldownFrames = 3;
// 脚底投影跨度容差(px)：脚底在 t1 线段延长线上、但投影超出线段两端该范围时，
// 视为"不在闸口跨度内"，不参与跨线判定（防人在闸口侧面远处横向走动被判跨线）
constexpr float kLineSpanMargin = 50.0f;

// ---- 消失回补 ----
// 脚底已明确跨到线的另一侧、但翻转确认帧数还没凑够(< kLineSideConfirmFrames)就
// 消失在活跃输出里的，连续消失该帧数后按"已完成一次跨线"补发一次事件。
constexpr int kLostCrossFrames = 3;

// ---- 显示标记：清晰在外 ----
// "清晰在区域外"的门限(px)：仅用于把 inside_region 标记翻转为区域外（不参与触发）
constexpr float kMinOutsideDepth = kRegionGapTolerance + 15.0f;

// ---- ID 交接（req3）----
// 新 ID 与某旧 ID 距离 ≤ kHandoverRadius、且旧 ID 在 kHandoverMaxLostFrames 帧内失联时，
// 继承旧 ID 的所在侧与区域上下文。
// ⚠️ 跨线瞬间跟丢时人已走到线的另一侧，新旧距离会明显大于"原地抖动"，
// 实测 lost 也可能到数十帧 → 半径与时间窗比纯遮挡场景放宽一些。
constexpr int kHandoverMaxLostFrames = 30;
constexpr float kHandoverRadius = 180.0f;

// 状态保留帧数（超过则清理，防止 map 无限增长）
constexpr int kStateKeepFrames = 120;

// 检测质量门限：低置信度/刚出现的新框不参与进入类/离开类判定（两类口径对称）
constexpr float kMinConfidence = 0.25f;
constexpr int kMinHitStreak = 2;

// ---- 遮挡脚底校正 ----
// 遮挡时检测框底部会从脚底跳到胸口（单帧上跳上百像素），遮挡结束又落回。
// ByteTrack 遮挡期间保持同一 ID，故用 per-ID 地面基准识别"被抬起"的框，
// 把脚底冻结回地面，避免脚底点在区域边界上反复穿越造成误计。
constexpr float kFootLiftMinPx = 40.0f;
constexpr float kFootLiftBoxRatio = 0.30f;
}

GateIdCounter::GateIdCounter(GateRole role, const Tripwire& entry_line,
                             float entry_zone_depth, const cv::Mat& detection_mask)
    : role_(role),
      entry_line_(entry_line),
      entry_depth_(std::abs(entry_zone_depth)),
      entry_depth_signed_(entry_zone_depth),
      detection_mask_(detection_mask.clone()) {
    if (!detection_mask_.empty()) {
        cv::Mat mask_copy = detection_mask_.clone();
        cv::findContours(mask_copy, region_contours_, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
    }

    // 标记线法向必须指向"检测区侧"（201 入口区域 / 200 出口区域都在这一侧），
    // 否则跨线判定会整体反相。配置里 depth 缺失(<1)时用检测区多边形反推，
    // depth 画反时自动纠正并告警。
    if (entry_line_.isValid() && !region_contours_.empty()) {
        const std::vector<cv::Point>* big = nullptr;
        size_t big_n = 0;
        for (const auto& c : region_contours_)
            if (c.size() > big_n) { big_n = c.size(); big = &c; }
        if (big && big->size() >= 3) {
            auto probeDepth = [&](float sign) {
                const cv::Point2f n = inwardNormal(entry_line_, sign);
                const cv::Point2f mid = (entry_line_.p1 + entry_line_.p2) * 0.5f;
                const cv::Point2f probe =
                    mid + n * std::max(30.0f, std::abs(entry_depth_signed_));
                return (float)cv::pointPolygonTest(*big, probe, true);
            };
            const float d_pos = probeDepth(1.0f);
            const float d_neg = probeDepth(-1.0f);
            if (std::abs(entry_depth_signed_) < 1.0f) {
                entry_depth_signed_ = (d_pos >= d_neg) ? 100.0f : -100.0f;
                std::fprintf(stderr,
                             "[GATE] WARN: 配置未给出 t1 depth，按检测区方向推断为 %.0f\n",
                             entry_depth_signed_);
            } else if (d_neg > d_pos) {
                entry_depth_signed_ = -entry_depth_signed_;
                std::fprintf(stderr,
                             "[GATE] WARN: t1 depth 方向与检测区相反，已自动纠正为 %.0f\n",
                             entry_depth_signed_);
            }
        }
    }
    entry_depth_ = std::abs(entry_depth_signed_);

    // 入口带：t1 向检测区侧延伸 |depth| px 的矩形。|depth| < 1 视为无入口带。
    if (entry_line_.isValid() && entry_depth_ >= 1.0f)
        entry_zone_ = zonePolygon(entry_line_, entry_depth_signed_);

    if (region_contours_.empty() && !entry_line_.isValid()) {
        std::fprintf(stderr,
                     "[GATE] WARN: 检测区多边形与标记线(t1)均无效，区域判定恒为 R0，"
                     "闸机计数不会产生事件\n");
    }
}

cv::Point2f GateIdCounter::inwardNormal(const Tripwire& tw, float depth) const {
    cv::Point2f dir = tw.p2 - tw.p1;
    const float length = std::sqrt(dir.dot(dir));
    if (length < 1.0f) return cv::Point2f(0.0f, 0.0f);
    const cv::Point2f normal(dir.y / length, -dir.x / length);
    return normal * (depth >= 0.0f ? 1.0f : -1.0f);
}

float GateIdCounter::signedDist(const cv::Point2f& pt, const Tripwire& tw,
                                const cv::Point2f& normal) const {
    return (pt - tw.p1).dot(normal);
}

std::vector<cv::Point2f> GateIdCounter::zonePolygon(const Tripwire& tw,
                                                    float depth) const {
    if (!tw.isValid() || std::abs(depth) < 1.0f) return {};
    const cv::Point2f offset = inwardNormal(tw, depth) * std::abs(depth);
    return {tw.p1, tw.p2, tw.p2 + offset, tw.p1 + offset};
}

bool GateIdCounter::pointInPolygon(const cv::Point2f& pt,
                                   const std::vector<cv::Point2f>& polygon,
                                   float tolerance) const {
    return polygon.size() >= 3 &&
           cv::pointPolygonTest(polygon, pt, true) >= -tolerance;
}

float GateIdCounter::regionDistance(const cv::Point2f& pt) const {
    if (!region_contours_.empty()) {
        // 到检测区边界的有符号距离：>0 在区内。<0 时取"最不坏"的那个多边形距离。
        float best = -1e9f;
        for (const auto& contour : region_contours_) {
            if (contour.size() < 3) continue;
            const float d = (float)cv::pointPolygonTest(contour, pt, true);
            best = std::max(best, d);
        }
        if (best > -1e8f) return best;
    }
    // 兜底（无检测区多边形）：回退"标记线二分"，>0 = 入侧(视为检测区内)
    if (entry_line_.isValid())
        return signedDist(pt, entry_line_, inwardNormal(entry_line_, entry_depth_));
    return -1e6f;  // 无任何几何信息 → 恒为区域外
}

int GateIdCounter::regionOf(const cv::Point2f& pt) const {
    // 入口带优先级最高，避免带与多边形交叠处的歧义
    if (!entry_zone_.empty() && pointInPolygon(pt, entry_zone_, kRegionGapTolerance))
        return GATE_REGION_ENTRY;
    if (regionDistance(pt) >= -kRegionGapTolerance)
        return GATE_REGION_DETECT;
    return GATE_REGION_OUTSIDE;
}

void GateIdCounter::reset() { states_.clear(); }

std::vector<GateIdMark> GateIdCounter::marks() const {
    std::vector<GateIdMark> out;
    out.reserve(states_.size());
    for (const auto& kv : states_) {
        const State& st = kv.second;
        GateIdMark mark;
        mark.track_id = kv.first;
        mark.region = st.region;
        mark.inside = st.inside_region;
        mark.entered = st.entered;
        mark.exited = st.exited;
        mark.inherited = st.inherited;
        mark.origin_id = st.origin_id;
        mark.lost_frames = st.lost_frames;
        out.push_back(mark);
    }
    return out;
}

int GateIdCounter::insideCount() const {
    // 只统计"仍然代表某个人的 ID"：被交接封存的旧 ID 已经由新 ID 代表，
    // 若继续计入会让区域内人数虚高（遮挡切换后 3 号与 5 号会同时算一人在内）。
    int n = 0;
    for (const auto& kv : states_)
        if (kv.second.inside_region && !kv.second.sealed) n++;
    return n;
}

std::vector<GateCountEvent> GateIdCounter::update(
    const std::vector<TrackedObject>& tracks) {
    std::vector<GateCountEvent> events;
    std::set<int> active_ids;  // 本帧 ByteTrack 输出的所有 track（含预测框）

    // 事件构造：role 决定标签，201=入口(进入类 IN / 离开类 RETRO)，
    // 200=出口(进入类 RETRO / 离开类 OUT)。
    auto makeEvent = [&](int track_id, int type, const cv::Point2f& foot, int region,
                         int origin_id, bool via_entry_band, bool settled,
                         int lost_frames, const char* reason) {
        GateCountEvent ev;
        ev.track_id = track_id;
        ev.type = type;
        ev.foot = foot;
        ev.region = region;
        ev.origin_id = origin_id;
        ev.via_entry_band = via_entry_band;
        ev.settled = settled;
        ev.lost_frames = lost_frames;
        ev.reason = reason;
        return ev;
    };
    const int enter_type = (role_ == GATE_ROLE_ENTRY) ? GATE_EVENT_IN : GATE_EVENT_RETRO;
    const int leave_type = (role_ == GATE_ROLE_ENTRY) ? GATE_EVENT_RETRO : GATE_EVENT_OUT;

    for (const auto& track : tracks) {
        active_ids.insert(track.track_id);
        // 预测框(observed=false)只维持轨迹连续，不改变区域、标记与计数状态：
        // 遮挡期间状态被"冻结"在原区域，ID 始终标记为还在区域内。
        if (!track.observed) continue;

        auto it = states_.find(track.track_id);
        const bool is_new_track = (it == states_.end());
        if (is_new_track) it = states_.emplace(track.track_id, State{}).first;
        State& state = it->second;

        // 异步推理可能乱序返回，旧结果不得让轨迹状态倒退
        if (track.result_frame_id >= 0 &&
            track.result_frame_id <= state.last_result_frame_id)
            continue;
        state.last_result_frame_id = track.result_frame_id;
        state.lost_frames = 0;
        state.last_hit_streak = track.hit_streak;
        state.last_confidence = track.confidence;

        // ---- 脚底点（含遮挡抬起校正）----
        const cv::Point2f foot_raw(track.box.x + track.box.width * 0.5f,
                                   track.box.y + track.box.height * kFootRatio);
        const float foot_lift = state.has_ref_foot ? (state.ref_foot.y - foot_raw.y)
                                                   : 0.0f;
        const bool foot_lifted = state.has_ref_foot &&
            foot_lift > std::max(kFootLiftMinPx,
                                 kFootLiftBoxRatio * (float)track.box.height);
        const cv::Point2f foot = foot_lifted
            ? cv::Point2f(foot_raw.x, state.ref_foot.y) : foot_raw;
        if (!foot_lifted) {
            state.ref_foot = foot_raw;
            state.has_ref_foot = true;
        }
        state.last_foot = foot;

        const int raw_region = regionOf(foot);
        const float region_dist = regionDistance(foot);  // >0 区内, <0 区外

        // ---- 本相机标记线(t1)跨线状态（角色隔离）----
        // signed > 0 = 线内侧(检测区侧)，< 0 = 线外侧(真·外面)。
        // line_active 还要求脚底投影落在 t1 线段跨度内，避免"人在闸口侧面远处
        // 横向走动"被判成跨过这条（无限延长的）线。
        const bool has_line = entry_line_.isValid();
        float line_signed = 0.0f;
        bool line_in_span = false;
        if (has_line) {
            const cv::Point2f ldir = entry_line_.p2 - entry_line_.p1;
            const float llen = std::sqrt(ldir.dot(ldir));
            if (llen >= 1.0f) {
                const cv::Point2f lu(ldir.x / llen, ldir.y / llen);
                const float line_t = (foot - entry_line_.p1).dot(lu);
                line_in_span = (line_t >= -kLineSpanMargin &&
                                line_t <= llen + kLineSpanMargin);
            }
            line_signed = signedDist(foot, entry_line_,
                                     inwardNormal(entry_line_, entry_depth_signed_));
        }
        const bool line_active = has_line && line_in_span;
        state.line_signed = line_signed;
        state.line_active = line_active;
        state.line_outside = line_active && (line_signed <= -kCrossLineDepth);

        // 检测质量（进入类/离开类共用）
        const bool quality_ok =
            track.hit_streak >= kMinHitStreak && track.confidence >= kMinConfidence;

        // ---- ID 交接（req3）：新 ID 在旧 ID 刚失联的位置附近出生时继承其上下文 ----
        // 只继承 line_side（= 上一次确认的所在侧）与区域/脚底上下文，**不再继承
        // entered/exited 一次性封顶**：只要侧不翻转就不会产生事件，此后每一次真实
        // 跨线都会重新计数。（旧实现继承封顶 + "进出自计满就跳过交接"正是单人多次
        // 往返被连续漏计的根因。）
        if (is_new_track) {
            State* best = nullptr;
            int best_id = -1;
            float best_dist = kHandoverRadius;
            for (auto& kv : states_) {
                State& old = kv.second;
                if (kv.first == track.track_id) continue;
                if (old.sealed) continue;                       // 已交付过，不再被继承
                if (old.line_side < 0) continue;                // 未确立所在侧，无可继承
                if (old.lost_frames < 1 ||
                    old.lost_frames > kHandoverMaxLostFrames) continue;
                const float d = (float)cv::norm(old.last_foot - foot);
                if (d > best_dist) continue;
                best_dist = d; best = &old; best_id = kv.first;
            }
            if (best) {
                state.line_side          = best->line_side;
                state.line_side_pending  = best->line_side;
                state.line_side_pending_frames = kLineSideConfirmFrames;
                state.line_flip_cooldown = best->line_flip_cooldown;
                state.ever_in_R2       = best->ever_in_R2 || best->inside_region;
                state.inside_region    = best->inside_region;
                state.entered          = best->entered;
                state.exited           = best->exited;
                state.region           = best->region;   // 沿用旧 ID 的区域，避免瞬时跳变
                state.pending_region   = best->region;
                state.pending_frames   = kRegionConfirmFrames;
                state.outside_clear    = false;
                state.outside_frames   = 0;
                state.has_ref_foot     = best->has_ref_foot;
                state.ref_foot         = best->ref_foot;
                state.has_last_inside_foot = best->has_last_inside_foot;
                state.last_inside_foot = best->last_inside_foot;
                state.inherited        = true;
                state.origin_id        = (best->origin_id >= 0) ? best->origin_id : best_id;
                // 旧 ID 封存：不再发事件、不再被继承，并把显示标记翻到区域外。
                best->sealed = true;
                best->inside_region = false;
            }
        }

        // ---- 区域确认（滞回；仅用于显示标记与日志）----
        if (raw_region == state.pending_region) {
            if (state.pending_frames < kRegionConfirmFrames) state.pending_frames++;
        } else {
            state.pending_region = raw_region;
            state.pending_frames = 1;
        }
        const int prev_region = state.region;
        if (state.pending_frames >= kRegionConfirmFrames &&
            state.pending_region != state.region) {
            state.region = state.pending_region;
        }
        if (state.region != prev_region) {
            if (state.region == GATE_REGION_DETECT) {
                state.ever_in_R2 = true;
                state.inside_region = true;   // req1：标记"在区域内"
                state.outside_clear = false;
                state.outside_depth = 0.0f;
                state.outside_frames = 0;
            } else if (state.region == GATE_REGION_OUTSIDE) {
                // 离开前最后一站是否为入口带（2→1→0 即逆行离开）
                state.exit_via_entry_band = (prev_region == GATE_REGION_ENTRY);
            }
        }
        // "清晰在区域外"仅用于把显示标记翻转为区域外（不参与触发）
        if (state.region == GATE_REGION_OUTSIDE) {
            const float depth = -region_dist;   // >0 = 在检测区边界外多深
            if (depth >= kMinOutsideDepth) {
                state.outside_clear = true;
                state.outside_depth = depth;
                state.outside_frames++;
                state.inside_region = false;
            } else {
                state.outside_frames = 0;
            }
        } else if (state.region == GATE_REGION_DETECT) {
            state.outside_clear = false;
            state.outside_frames = 0;
            state.has_last_inside_foot = true;
            state.last_inside_foot = foot;
        }

        // ---- 侧翻转边沿检测（唯一触发源）----
        // 每次"确认的侧翻转"恰好产生一次事件：
        //   线外 → 线内 = 进入类（201=IN / 200=RETRO）
        //   线内 → 线外 = 离开类（201=RETRO / 200=OUT）
        // 与 ID 生命周期解耦 → 同一 ID 反复往返，每次跨线都计一次。
        if (state.line_flip_cooldown > 0) state.line_flip_cooldown--;
        if (line_active) {
            int raw_side = -1;
            if (line_signed <= -kCrossLineDepth) raw_side = 0;
            else if (line_signed >= kCrossLineDepth) raw_side = 1;

            if (raw_side >= 0) {
                if (raw_side == state.line_side_pending) {
                    if (state.line_side_pending_frames < kLineSideConfirmFrames)
                        state.line_side_pending_frames++;
                } else {
                    state.line_side_pending = raw_side;
                    state.line_side_pending_frames = 1;
                }
                const bool confirmed =
                    state.line_side_pending_frames >= kLineSideConfirmFrames;

                if (confirmed && state.line_side < 0) {
                    state.line_side = state.line_side_pending;   // 首次确立，不发事件
                } else if (confirmed && state.line_side_pending != state.line_side &&
                           state.line_flip_cooldown == 0 && quality_ok && !state.sealed) {
                    if (state.line_side_pending == 1) {
                        state.entered = true;
                        events.push_back(makeEvent(
                            track.track_id, enter_type, foot, state.region,
                            state.origin_id, false, false, 0,
                            state.inherited ? "enter-cross-line-inherited"
                                            : "enter-cross-line"));
                    } else {
                        state.exited = true;
                        state.inside_region = false;
                        events.push_back(makeEvent(
                            track.track_id, leave_type, foot, state.region,
                            state.origin_id, state.exit_via_entry_band, false, 0,
                            state.exit_via_entry_band ? "leave-cross-line-via-entry"
                                                      : "leave-cross-line"));
                    }
                    state.line_side = state.line_side_pending;
                    state.line_flip_cooldown = kLineFlipCooldownFrames;
                }
            }
        }

    }

    // ---- 消失回补（lost 循环）----
    // 跨线事件在"确认翻转"的那一刻就发出，正常跨线无需等 ID 消失。本循环只兜一种
    // 漏计：脚底已明确跨到线的另一侧，但翻转确认帧数(< kLineSideConfirmFrames)
    // 还没凑够就消失在活跃输出里 → 按"已完成一次跨线"补发一次事件。每次跨线只补
    // 一次：补发后立即提交 line_side，后续帧 implied 与 line_side 相同，不再重复。
    for (auto it = states_.begin(); it != states_.end();) {
        State& state = it->second;
        if (active_ids.count(it->first)) { ++it; continue; }

        state.lost_frames++;

        if (entry_line_.isValid() && state.line_active && state.line_side >= 0 &&
            !state.sealed &&
            state.lost_frames >= kLostCrossFrames &&
            state.last_hit_streak >= kMinHitStreak &&
            state.last_confidence >= kMinConfidence) {
            const int implied = (state.line_signed <= -kCrossLineDepth) ? 0
                              : (state.line_signed >= kCrossLineDepth) ? 1 : -1;
            if (implied >= 0 && implied != state.line_side) {
                if (implied == 1) {
                    state.entered = true;
                    events.push_back(makeEvent(it->first, enter_type, state.last_foot,
                                               state.region, state.origin_id, false, true,
                                               state.lost_frames,
                                               "enter-cross-line-on-loss"));
                } else {
                    state.exited = true;
                    state.inside_region = false;
                    events.push_back(makeEvent(it->first, leave_type, state.last_foot,
                                               state.region, state.origin_id,
                                               state.exit_via_entry_band, true,
                                               state.lost_frames,
                                               "leave-cross-line-on-loss"));
                }
                state.line_side = implied;
            }
        }

        if (state.lost_frames > kStateKeepFrames) {
            it = states_.erase(it);
        } else {
            ++it;
        }
    }

    return events;
}
