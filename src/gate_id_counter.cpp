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

// ---- 诊断注记（只读，不影响计数）----
// 脚底离线这个距离以内，若某帧只有预测框(observed=false)，就记一条注记：
// 现场"视频里明明有完整轨迹却没计数"，多半就是这些帧没有被计数模块看到。
constexpr float kNoteNearLinePx = 250.0f;
// 首次确立所在侧时，脚底已离线超过该距离 → 说明这个 ID"出生时就没被观测到跨线"。
constexpr float kNoteFarFromLinePx = 120.0f;

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

// 状态保留帧数（超过则清理，防止 map 无限增长）。
// ⚠️ 必须 > tracker 的 max_lost_（main_tracking.cpp:299 传 180），否则会出现
//    "旧轨迹被 tracker 重连、以原 ID 回到输出，但计数器早已 erase 它的状态" →
//    被当成全新 ID → 触发**假交接**并封存附近真实 ID（现场 2026-09-18：
//    `track=18 ... inherited_from=21`，而 tracker ID 单调递增，18 < 21 本不该出现）。
//    取 240 = 180 + 60 帧余量；与 kHandoverMaxLostFrames(30) 无关，交接窗口不受影响。
constexpr int kStateKeepFrames = 240;

// "计数器见过哪些 track_id" 的记忆长度（单位 = update() 帧）。
// 只需覆盖 tracker 彻底忘记一条轨迹所需的时间（max_lost_ = 180 帧），取 600 帧(≈24s) 足够；
// 超过此长度仍在输出里的轨迹说明它一直活着，不可能"复活"，忘掉它没有风险。
constexpr int kSeenIdHorizonUpdates = 600;

// 解封确认帧数（M3，2026-09-18）。见 update() 内"解封"一节。
// 必须**严格大于** kLineSideConfirmFrames(3)：解封本身也要是一道比计数更严的门。
constexpr int kUnsealConfirmFrames = 5;

// ---- M4：用"运动性(轨迹)"排除静止误检 ----
// 现场背景：检测区里有一件**静止物品**被误识别成人（现场是门口的饮水机）。它产生的 ID 永远不动，
// 但真人跨门时它会搅乱关联，甚至"接管"真人的轨迹把它封存，导致真人那次跨线凭空消失。
// 判据出发点（用户 2026-09-18 提出）：**人是有轨迹的，误识别物品没有轨迹**。
//   ⚠️ 但用户同时提醒"人在区域内可能会站着不动" → 所以不能简单用"瞬时速度=0"判假：
//      "站着不动的人"与"静止假框"在**短窗口**内位移特征完全重叠，不可分。
//   ⇒ 因此本方案**不去猜谁是假的**，而是改变"不可逆动作"的资格：
//      一个 ID 想**终结(封存)另一条轨迹**，必须由**它自己走出来的轨迹**背书。
//      静止假框永远走不出这段路 ⇒ 永远没有资格封存真人；而"站着不动的人"只是**暂时**没资格，
//      他既不会被误杀（继承来的上下文仍然生效，走起来照样计数），也不会因此漏计。
//
// 累计有效位移的量化粒度：与锚点距离 ≥ 此值才累加并推进锚点。
// 用"锚点法"而不是"逐帧位移求和"：慢速移动（每帧 3px，但确实在走）能累积起来，
// 而框在原地的高频抖动（正负抵消）不会累积。实测框抖动幅度约 1~5px。
constexpr float kMoveMinStep = 6.0f;
// "延迟封存"考察期：新 ID 出生并继承上下文后，给它这么多帧去证明"自己会动"。
constexpr int kSealProbationFrames = 5;
// 考察期内累计位移须达到此值，才允许它去封存旧 ID。
// 30px 对真人只需 1~2 帧正常行走（每帧 15~25px）⇒ 不会误杀；
// 对静止假框则永远达不到（抖动被 kMoveMinStep 滤掉）。
constexpr float kSealMinTravel = 30.0f;

// 检测质量门限：低置信度/刚出现的新框不参与进入类/离开类判定（两类口径对称）
constexpr float kMinConfidence = 0.25f;
constexpr int kMinHitStreak = 2;

// ---- 遮挡脚底校正 ----
// 遮挡时检测框底部会从脚底跳到胸口（单帧上跳上百像素），遮挡结束又落回。
// ByteTrack 遮挡期间保持同一 ID，故用 per-ID 地面基准识别"被抬起"的框，
// 把脚底冻结回地面，避免脚底点在区域边界上反复穿越造成误计。
constexpr float kFootLiftMinPx = 40.0f;
constexpr float kFootLiftBoxRatio = 0.30f;
constexpr float kMaxUnexplainedJumpPx = 300.0f;

// ---- 跟丢越线推断（治"跨线瞬间跟丢且没有任何后继 ID"的整段漏计）----
// ★★ 默认关闭 ★★（GateIdCounter::enableLostCrossingInference()，默认 false）。
//   2026-09-17 用户裁定：这一类"补计数"没有实时观测依据，现场出现过"人没出区域却被计一次 OUT"
//   （200 相机 OUT:6→7，见文档 §19/§20）。关闭后所有 IN/OUT/RETRO 都必须由真实跨线观测支撑。
//   本节逻辑保留在代码里、由开关控制，是为了将来做"推断 + 实时观测"的混合方案时可直接复用。
//
// 背景（当初为什么要写它）：消失回补(lost 循环里的 implied 分支)只兜"最后观测脚底已经越过
//       死区到线外"；交接跨线兑付要有"后继 ID"才能观测到跨线后的那一侧。二者都覆盖不到
//       "人正跨线的那一瞬间彻底跟丢、之后再没出现" → 该趟跨线整段漏计（现场 P12）。
//       ⇒ 这是**已知代价**：开关关闭时，这类"跨线瞬间被吞"会漏 1 个。
//
// 判据（开关打开时才生效；四项同时成立才推断，只对【离开类】方向生效）：
//   ① 最后一次被接受的观测时 line_side == 1（已确证在线内侧）；
//   ② 脚底离标记线不远：line_signed ≤ kInferNearLinePx（还没走到检测区深处）；
//   ③ 仍以 ≥ kInferMinOutwardPx 的速度朝线外移动；
//   ④ ★跨线在望★：按该速度推算，从最后观测位置到"越过死区"只需 ≤ kInferMaxReachFrames 帧；
//      再把该速度外推 lost_frames 帧后确实已越过死区（≤ -kCrossLineDepth）→ 判定完成跨线。
//      "在望"是这条推断的立身之本：只有当人**马上**就要跨过去时，"跟丢"才等价于"跨过去了"。
//      不紧迫就说明人还没跨（在区域内活动 / 走到线边又停下 / 被遮挡），不能替他补一次跨线。
// 为什么不做"线外→线内"的对称推断：进入类的前提是"确实观测到人到了线内"（见 T4/T13
//   "不凭空计 IN"）；而离开类的前提"人曾在线内侧"已被 ① 确证，故不对称是刻意的。
// 为什么用帧间速度：走到门口后站定再消失（T11/T22）用平均速度看仍"朝外"会被误判，
//   帧间速度归零即可正确排除；T7/T21（区内深处消失）则被 ② 排除，不会抢对面相机的 OUT。
//
// ⚠️ ④ 的由来（2026-09-17 现场假 OUT，见文档 §19）：
//   板端 200 出口相机，一个人从区域外进区、在区内转圈，走到判定线**末端附近**但没跨出去，
//   随后跟丢 → 补发了一次 OUT（OUT:6→7）。用真配置复算几何：最后脚底 (1449,590.04)
//   最近的是判定线末段 (1502,631)->(1504,625)，line_signed=+63px；事件在第 7 个丢失帧发出，
//   反推帧间速度约 -12px/帧 → 越过死区需要 ~6.9 帧。
//   旧判据只看"外推 lost_frames 帧后跨过去"（窗口放到 20 帧），于是第 7 帧凑够就误判了；
//   本质是**允许在外推窗口里用 20 帧的余量去凑一次"本来没发生的跨线"**。
//   补上 ④（need ≤ 5 帧 = 翻转确认窗口 3 帧 + 2 帧余量）后，同一组数字被拒（6.9 > 5）；
//   而 T20/T23 这种"真·跨线瞬间跟丢"（need ≈ 1.5~3 帧）照旧能推断出来。
//   ★ 但用户随后裁定：即使 ④ 能拒掉这一例，"没有观测就补计数"的机制本身也不接受 →
//     默认关闭（§20），④ 只作为"将来若要打开时的最低准入条件"保留。
constexpr float kInferNearLinePx   = 100.0f;
constexpr float kInferMinOutwardPx = 2.5f;
constexpr int   kInferMaxReachFrames = 5;   // ④："还要 ≤5 帧就越过死区"
constexpr int   kInferMaxLostFrames  = 20;  // 外推窗口硬上限（④ 生效后一般轮不到它）

// ④ 的量化：从当前位置出发、按帧间速度还需要多少帧才能越过死区到线外。
// 速度不够朝外（或朝内）时返回一个极大的数 = 永远跨不过去，等价于"不在望"。
float framesToCross(float signed_d, float vel) {
    if (vel >= -kInferMinOutwardPx) return 1e9f;
    return (signed_d + kCrossLineDepth) / (-vel);
}
}

// ---------- 判定线的两个构造入口 ----------
// 两点直线（旧的 t1）直接退化成 2 点折线：之后所有算法只有一条实现路径，
// 且 2 点时逐位等价于旧实现（测试 T1~T26 沿用竖直 2 点线，必须无回归）。
GateIdCounter::GateIdCounter(GateRole role, const Tripwire& entry_line,
                             float entry_zone_depth, const cv::Mat& detection_mask,
                             int cross_dwell_frames)
    : GateIdCounter(role,
                    entry_line.isValid()
                        ? std::vector<cv::Point2f>{entry_line.p1, entry_line.p2}
                        : std::vector<cv::Point2f>{},
                    entry_zone_depth, detection_mask, cross_dwell_frames) {}

GateIdCounter::GateIdCounter(GateRole role, const std::vector<cv::Point2f>& line_pts,
                             float entry_zone_depth, const cv::Mat& detection_mask,
                             int cross_dwell_frames)
    : role_(role),
      cross_dwell_frames_(std::max(1, cross_dwell_frames)),
      entry_depth_(std::abs(entry_zone_depth)),
      entry_depth_signed_(entry_zone_depth),
      detection_mask_(detection_mask.clone()) {
    // 剔除重合的相邻点：零长度线段算不出法向与投影
    for (const auto& p : line_pts) {
        if (line_pts_.empty() ||
            std::hypot(p.x - line_pts_.back().x, p.y - line_pts_.back().y) >= 1.0f)
            line_pts_.push_back(p);
    }
    if (line_pts_.size() < 2) line_pts_.clear();

    if (!detection_mask_.empty()) {
        cv::Mat mask_copy = detection_mask_.clone();
        cv::findContours(mask_copy, region_contours_, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);
    }

    // 判定线法向必须指向"检测区侧"（区域就在这一侧），否则跨线判定会整体反相。
    // 探测点 = 折线弧长中点 ± 该段法向 * max(30,|depth|)；2 点直线时就是线段中点，
    // 与旧实现完全一致。配置里 depth 缺失(<1) 时用检测区多边形反推，画反了自动纠正并告警。
    if (line_pts_.size() >= 2 && !region_contours_.empty()) {
        const std::vector<cv::Point>* big = nullptr;
        size_t big_n = 0;
        for (const auto& c : region_contours_)
            if (c.size() > big_n) { big_n = c.size(); big = &c; }
        if (big && big->size() >= 3) {
            auto rawNormal = [&](int seg) -> cv::Point2f {
                const cv::Point2f d = line_pts_[seg + 1] - line_pts_[seg];
                const float L = std::sqrt(d.dot(d));
                if (L < 1.0f) return cv::Point2f(0.0f, 0.0f);
                return cv::Point2f(d.y / L, -d.x / L);      // 右手系法向（未定向）
            };
            const cv::Point2f mid = lineMidPoint();
            const cv::Point2f nref = rawNormal(midSegment());
            auto probeDepth = [&](float sign) {
                const cv::Point2f probe =
                    mid + nref * (sign * std::max(30.0f, std::abs(entry_depth_signed_)));
                return (float)cv::pointPolygonTest(*big, probe, true);
            };
            const float d_pos = probeDepth(1.0f);
            const float d_neg = probeDepth(-1.0f);
            // 内侧方向 = "mid + nref" 那一侧在检测区内更深就取 +，否则取 −；幅值沿用配置。
            // ⚠️ 必须"先定方向、再乘幅值"，不能写成"d_neg>d_pos 就取反"：那样在配置里
            //    已经写成负值时会把方向又翻回去（正负来回错），表现为区域整体反相。
            const float side = (d_pos >= d_neg) ? 1.0f : -1.0f;
            if (std::abs(entry_depth_signed_) < 1.0f) {
                entry_depth_signed_ = 100.0f * side;
                std::fprintf(stderr,
                             "[GATE] WARN: 配置未给出区域宽度，按检测区方向推断为 %.0f\n",
                             entry_depth_signed_);
            } else if (entry_depth_signed_ * side < 0.0f) {
                entry_depth_signed_ = std::abs(entry_depth_signed_) * side;
                std::fprintf(stderr,
                             "[GATE] WARN: 区域宽度方向与检测区相反，已自动纠正为 %.0f\n",
                             entry_depth_signed_);
            }
        }
    }
    entry_depth_ = std::abs(entry_depth_signed_);

    // 区域环 = 判定线 ∪ 其沿 x 平移 |depth| px 的副本（y 不变）。|depth| < 1 视为无区域。
    if (line_pts_.size() >= 2 && entry_depth_ >= 1.0f) {
        entry_zone_ = zonePolygon(zoneOffsetX());
        // 区域是"沿 x 平移"出来的：判定线的**中间线段**越接近水平，该处的实际带宽
        //（|平移量| × |内侧法向 x 分量|）越窄，极端情况退化成贴着判定线的窄条纹。
        // 只影响区域显示/日志（regionOf），不影响跨线判定，提示一下即可；
        // 折线只要有一段接近竖直就仍有正常带宽，这里只看弧长中点所在的那段。
        const float nx = inwardNormal(midSegment()).x;
        if (std::abs(nx) < 0.25f)
            std::fprintf(stderr,
                         "[GATE] WARN: 判定线的中间线段接近水平(内侧法向 x 分量=%.2f)，"
                         "该处区域带宽仅 %.0fpx(趋于 0)。跨线判定不受影响；"
                         "若希望区域看上去更宽，可把判定线画在接近竖直的那段边缘上\n",
                         nx, entry_depth_ * std::abs(nx));
    }

    if (line_pts_.size() < 2) {
        // 判定线缺失是"计数彻底不工作"级故障：跨线边沿永远不会触发。
        // 必须显式告警，不能因为检测区多边形还在就静默通过 ——
        // 那样 Region/Zone 显示仍正常，会让人误以为一切正常（现场踩过）。
        std::fprintf(stderr,
                     "[GATE] WARN: 判定线无效(有效点数<2)%s → 本相机不会产生任何 "
                     "IN/OUT/RETRO 事件\n",
                     region_contours_.empty() ? "，且检测区多边形也无效" : "");
    }
}

cv::Point2f GateIdCounter::inwardNormal(int seg) const {
    if (seg < 0 || seg + 1 >= (int)line_pts_.size()) return cv::Point2f(0.0f, 0.0f);
    const cv::Point2f dir = line_pts_[seg + 1] - line_pts_[seg];
    const float length = std::sqrt(dir.dot(dir));
    if (length < 1.0f) return cv::Point2f(0.0f, 0.0f);
    const cv::Point2f normal(dir.y / length, -dir.x / length);   // 右手系法向
    return normal * (entry_depth_signed_ >= 0.0f ? 1.0f : -1.0f);
}

int GateIdCounter::midSegment() const {
    if (line_pts_.size() < 2) return -1;
    float total = 0.0f;
    for (size_t i = 0; i + 1 < line_pts_.size(); ++i)
        total += (float)cv::norm(line_pts_[i + 1] - line_pts_[i]);
    const float half = total * 0.5f;
    float acc = 0.0f;
    for (size_t i = 0; i + 1 < line_pts_.size(); ++i) {
        const float L = (float)cv::norm(line_pts_[i + 1] - line_pts_[i]);
        if (L < 1e-3f) continue;
        if (acc + L >= half) return (int)i;
        acc += L;
    }
    return (int)line_pts_.size() - 2;
}

cv::Point2f GateIdCounter::lineMidPoint() const {
    if (line_pts_.size() < 2) return cv::Point2f(0.0f, 0.0f);
    float total = 0.0f;
    for (size_t i = 0; i + 1 < line_pts_.size(); ++i)
        total += (float)cv::norm(line_pts_[i + 1] - line_pts_[i]);
    const float half = total * 0.5f;
    float acc = 0.0f;
    for (size_t i = 0; i + 1 < line_pts_.size(); ++i) {
        const cv::Point2f d = line_pts_[i + 1] - line_pts_[i];
        const float L = std::sqrt(d.dot(d));
        if (L < 1e-3f) continue;
        if (acc + L >= half) return line_pts_[i] + d * ((half - acc) / L);
        acc += L;
    }
    return line_pts_.back();
}

GateIdCounter::LineEval GateIdCounter::evalLine(const cv::Point2f& pt) const {
    LineEval ev;
    if (line_pts_.size() < 2) return ev;
    const int nseg = (int)line_pts_.size() - 1;
    float best_dist = 1e18f;
    float best_t = 0.0f, best_len = 1.0f;
    for (int i = 0; i < nseg; ++i) {
        const cv::Point2f a = line_pts_[i];
        const cv::Point2f d = line_pts_[i + 1] - a;
        const float L2 = d.dot(d);
        if (L2 < 1e-6f) continue;
        const float t = (pt - a).dot(d) / L2;
        const float tc = std::min(1.0f, std::max(0.0f, t));
        const float dist = (float)cv::norm(pt - (a + d * tc));
        if (dist < best_dist) {
            best_dist = dist; ev.seg = i; best_t = t; best_len = std::sqrt(L2);
        }
    }
    if (ev.seg < 0) return ev;
    // 带符号垂距 = 到【命中线段所在直线】的投影。线段只有一段时就是旧的两点直线口径。
    ev.signed_dist = (pt - line_pts_[ev.seg]).dot(inwardNormal(ev.seg));
    // 跨度约束：内部线段的投影天然落在段内；只有两个端段需要 ±kLineSpanMargin 余量。
    // 2 点直线时唯一那段同时是首段与末段 → 与旧实现的行为完全一致。
    const float t_px = best_t * best_len;
    const bool is_first = (ev.seg == 0);
    const bool is_last = (ev.seg == nseg - 1);
    ev.in_span = (!is_first && !is_last) ||
                 (t_px >= -kLineSpanMargin && t_px <= best_len + kLineSpanMargin);
    return ev;
}

float GateIdCounter::zoneOffsetX() const {
    const float width = entry_depth_;
    if (line_pts_.size() < 2) return width;
    // 平移方向 = 内侧法向的 x 分量所指的那一侧（配置画反了也能自愈）
    return (inwardNormal(midSegment()).x >= 0.0f) ? width : -width;
}

std::vector<cv::Point2f> GateIdCounter::zonePolygon(float offset) const {
    std::vector<cv::Point2f> ring;
    if (line_pts_.size() < 2 || std::abs(offset) < 1.0f) return ring;
    // 区域 = 判定线 ∪ 其沿 x 平移 offset 的副本（y 不变），首尾相接成一个闭环。
    // 为什么不做"沿法向偏移的等距带"：判定线是斜线/折线时法向带也是斜的，与相邻区域
    // 拼不严实，走廊里会留缝；沿 x 平移后相邻区域共享同一条 x 方向边界，中间无缝。
    ring.reserve(line_pts_.size() * 2);
    for (const auto& p : line_pts_) ring.push_back(p);
    for (auto it = line_pts_.rbegin(); it != line_pts_.rend(); ++it)
        ring.push_back(cv::Point2f(it->x + offset, it->y));
    return ring;
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
    // 兜底（无检测区多边形）：回退"判定线二分"，>0 = 内侧(视为检测区内)。
    // evalLine 的法向已按带符号 depth 定向（entry_depth_signed_），
    // 故配置里 depth 为负时符号也不会反相（旧版用 entry_depth_(恒正) 会整块反相）。
    const LineEval ev = evalLine(pt);
    if (ev.seg >= 0) return ev.signed_dist;
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
        mark.line_signed = st.line_signed;
        mark.line_side = st.line_side;
        mark.line_active = st.line_active;
        out.push_back(mark);
    }
    return out;
}

int GateIdCounter::insideCount() const {
    // 只统计"此刻仍代表区域内某个人的 ID"：
    //   · 被**确认**交接的旧 ID（sealed）已由新 ID 代表，继续计入会让人数虚高
    //     （遮挡切换后 3 号与 5 号会同时算一人在内）；
    //   · 已经失联（lost_frames > 0）的 ID 此刻并没有被观测到，也不该算作"区域内的 ID"。
    //     M4 改成"延迟封存"后，处在待封存考察期的旧 ID 既没 sealed 也还没被清理，
    //     靠这一条把"已被接手但尚未确认"的重复身份排除掉。
    int n = 0;
    for (const auto& kv : states_)
        if (kv.second.inside_region && !kv.second.sealed &&
            kv.second.lost_frames == 0) n++;
    return n;
}

std::vector<GateCountEvent> GateIdCounter::update(
    const std::vector<TrackedObject>& tracks) {
    std::vector<GateCountEvent> events;
    notes_.clear();            // 诊断注记每帧重填（见 notes()）
    std::set<int> active_ids;  // 本帧 ByteTrack 输出的所有 track（含预测框）

    // ---- 复活 ID 识别（缺陷 B 的结构性防线，2026-09-18）----
    // tracker 的 max_lost_ 与状态保留期一旦不一致，被重连复活的旧轨迹会以"全新 ID"的姿态
    // 出现在本模块里，进而去**继承并封存附近真实轨迹**（现场：复活的 18 把真实的人 21 封存了，
    // 导致 21 连"逆行"都没有）。这里记一份"计数器见过哪些 track_id"的短期记忆，
    // 让"是否真的第一次见到"不再依赖状态保留期是否够长。
    ++update_seq_;
    if (last_seen_seq_.size() > 64 && (update_seq_ % 256) == 0) {
        // 按"最后一次见到"回收：超过 kSeenIdHorizonUpdates 没再出现的 ID 忘掉。
        // 该阈值远大于 tracker 把轨迹彻底删除所需的帧数，故不会漏判复活。
        for (auto sit = last_seen_seq_.begin(); sit != last_seen_seq_.end();) {
            if (update_seq_ - sit->second > kSeenIdHorizonUpdates)
                sit = last_seen_seq_.erase(sit);
            else
                ++sit;
        }
    }

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
    // 两个真实脚底连线是否"确实穿过判定线"：逐段求交 ——
    // 以该段的带符号距离判断 a/b 是否越过死区分居两侧，再要求交点投影落在
    // 该段跨度内（只在两个端段留 ±kLineSpanMargin 余量，与脚底跨度约束同源）。
    // 判定线退化成 2 点直线时只有一段，行为与旧的"单直线版"逐位一致。
    auto segmentCrossesLine = [&](const cv::Point2f& a, const cv::Point2f& b) {
        if (line_pts_.size() < 2) return false;
        const cv::Point2f ab = b - a;
        for (int i = 0; i + 1 < (int)line_pts_.size(); ++i) {
            const cv::Point2f seg = line_pts_[i + 1] - line_pts_[i];
            const float len2 = seg.dot(seg);
            if (len2 < 1.0f) continue;
            const cv::Point2f n = inwardNormal(i);
            const float da = (a - line_pts_[i]).dot(n);
            const float db = (b - line_pts_[i]).dot(n);
            if (!((da >= kCrossLineDepth && db <= -kCrossLineDepth) ||
                  (da <= -kCrossLineDepth && db >= kCrossLineDepth))) continue;
            const float denom = db - da;
            if (std::abs(denom) < 1e-3f) continue;
            const float t = -da / denom;
            if (t < 0.0f || t > 1.0f) continue;
            const cv::Point2f hit = a + ab * t;
            const float t_px = (hit - line_pts_[i]).dot(seg) / std::sqrt(len2);
            if (t_px >= -kLineSpanMargin &&
                t_px <= std::sqrt(len2) + kLineSpanMargin) return true;
        }
        return false;
    };

    for (const auto& track : tracks) {
        active_ids.insert(track.track_id);
        // 计数器此前见过这个 track_id 吗？（复活 ID 识别，见本函数顶部说明）
        const bool first_time_seen =
            (last_seen_seq_.find(track.track_id) == last_seen_seq_.end());
        last_seen_seq_[track.track_id] = update_seq_;
        // 预测框(observed=false)只维持轨迹连续，不改变区域、标记与计数状态：
        // 遮挡期间状态被"冻结"在原区域，ID 始终标记为还在区域内。
        // ★ 注意：画面上画的框包含这些预测框，所以"视频里看得到轨迹"≠"计数模块看到了观测"★
        if (!track.observed) {
            // 诊断：离判定线很近却只有预测框 → 这一帧的跨线信息完全丢失（现场头号漏计来源）
            auto ip = states_.find(track.track_id);
            if (ip != states_.end() && ip->second.line_active &&
                std::fabs(ip->second.line_signed) <= kNoteNearLinePx) {
                char buf[192];
                std::snprintf(buf, sizeof(buf),
                    "predicted-frame track=%d side=%d signed=%+.0fpx → 本帧无真实检测，不参与跨线判定",
                    track.track_id, ip->second.line_side, ip->second.line_signed);
                notes_.push_back(buf);
            }
            continue;
        }

        auto it = states_.find(track.track_id);
        const bool is_new_track = (it == states_.end());
        if (is_new_track) it = states_.emplace(track.track_id, State{}).first;
        State& state = it->second;

        // 异步推理可能乱序返回，旧结果不得让轨迹状态倒退
        if (track.result_frame_id >= 0 &&
            track.result_frame_id <= state.last_result_frame_id)
            continue;
        const int prev_result_fid = state.last_result_frame_id;   // 供帧间速度用
        state.last_result_frame_id = track.result_frame_id;
        state.lost_frames = 0;
        state.last_hit_streak = track.hit_streak;
        state.last_confidence = track.confidence;

        // ---- 解封（M3，2026-09-18）：被封存的 ID 又收到了自己的真实观测 ----
        // 背景：交接把旧 ID 封存（sealed）的前提是"它已经终止、不会再有后继观测"。
        // 若它此后**又被真实检测到**（observed=true，不是预测框），说明前提不成立 ——
        // 那次交接是误判（典型：附近出现假框/短暂误检把真人顶掉，误判者反倒把真人封存）。
        // 不解封的代价极大：`!state.sealed` 是**正常翻转路径**、**消失回补**与**交接兑付**
        // 三条路径的硬门 → 这个 ID 之后的一切真实轨迹被永久静默（现场真人 track=21
        // 被封存后再无任何事件，而它的轨迹全是真人真实轨迹）。
        //
        // 两条代价控制（防止解封变成"多计"）：
        //   ① 必须连续 kUnsealConfirmFrames(5) 帧**真实观测**（比跨线确认窗 3 帧更严），
        //      预测框不计入也不清零后再累积 → 瞬时噪声活不过这道门；静止假框虽能过门，
        //      但它不翻侧、不会产生事件，无害。
        //   ② 解封时把 line_side / 待确认侧**一并作废为 -1** → 必须重新确立所在侧，
        //      **绝不拿封存前的旧证据补计数**（符合"只认观测"铁律），下一次真实跨线才计
        //      → 天然不会与继承者就同一次跨线双计。
        if (state.sealed) {
            if (++state.unseal_streak >= kUnsealConfirmFrames) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "unsealed track=%d 被封存后又连续 %d 帧真实观测 → 判定那次交接是误判，"
                    "解封并恢复计数资格(旧侧已作废，需重新确立；不补发历史事件)",
                    track.track_id, state.unseal_streak);
                notes_.push_back(buf);
                state.sealed = false;
                state.unseal_streak = 0;
                state.line_side = -1;
                state.line_side_pending = -1;
                state.line_side_pending_frames = 0;
                state.inherited = false;
                state.line_flip_cooldown = 0;
                state.inside_region = (state.region == GATE_REGION_DETECT);
                // 它已被证明是自己活着的独立轨迹 → 恢复"自行补历史"的资格。
                state.handover_done = false;
            }
        } else {
            state.unseal_streak = 0;
        }

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
        const cv::Point2f previous_foot = state.last_foot;
        const bool had_previous_foot = state.has_last_foot;

        const int raw_region = regionOf(foot);
        const float region_dist = regionDistance(foot);  // >0 区内, <0 区外

        // ---- 本相机判定线（折线）跨线状态（角色隔离）----
        // signed > 0 = 线内侧(检测区侧)，< 0 = 线外侧(真·外面)。
        // evalLine 取"最近线段所在直线"的带符号垂距；in_span 还要求脚底投影落在
        // 折线跨度内（端段留 ±kLineSpanMargin 余量），避免"人在闸口侧面远处横向走动"
        // 被判成跨过这条（无限延长的）线。判定线是 2 点直线时与旧实现逐位一致。
        const bool has_line = (line_pts_.size() >= 2);
        const LineEval le = evalLine(foot);
        float line_signed = le.signed_dist;
        const bool line_in_span = le.in_span;
        const bool line_active = has_line && line_in_span;
        state.line_signed = line_signed;
        state.line_active = line_active;
        state.line_outside = line_active && (line_signed <= -kCrossLineDepth);

        // 检测质量（进入类/离开类共用）
        const bool quality_ok =
            track.hit_streak >= kMinHitStreak && track.confidence >= kMinConfidence;

        if (!is_new_track && had_previous_foot && state.inside_region &&
            raw_region == GATE_REGION_OUTSIDE &&
            cv::norm(foot - previous_foot) > kMaxUnexplainedJumpPx &&
            !segmentCrossesLine(previous_foot, foot)) {
            // Do not let a wrong outside box consume the real ID's departure
            // evidence; wait for the next consistent observation.
            state.lost_frames = 0;
            continue;
        }

        state.last_foot = foot;
        state.has_last_foot = true;
        state.obs_frames++;   // 记一次"被接受的真实观测"（诊断用）

        // ---- 轨迹账本（M4）：累计有效位移 ----
        // 锚点法：只有当脚底离"上一个锚点"≥ kMoveMinStep 时才把这距离计入并推进锚点。
        //   · 慢慢走（每帧 3px）：走两帧就离锚点 6px → 累积 → 不会漏掉慢速位移；
        //   · 原地抖动（±3px 来回摆）：永远回不到 6px 外的锚点以外再回来 → 累积极少；
        //   · 静止假框：几乎恒为 0。
        // 放在假框守卫之后（上面的 `continue` 已把不合规帧排除），保证被丢弃的帧不污染轨迹。
        if (state.has_travel_anchor) {
            const float moved = (float)cv::norm(foot - state.travel_anchor);
            if (moved >= kMoveMinStep) {
                state.travel_px += moved;
                state.travel_anchor = foot;
            }
        } else {
            state.travel_anchor = foot;
            state.has_travel_anchor = true;
        }

        // 越线推断用：最近两次"被接受的观测"之间 line_signed 的变化率(px/帧)。
        // 放在假框守卫之后，保证被丢弃的帧不污染速度；<0 = 朝标记线外移动。
        if (state.has_prev_line_signed) {
            const int df = std::max(1, track.result_frame_id - prev_result_fid);
            state.line_vel = (line_signed - state.prev_line_signed) / (float)df;
        }
        state.prev_line_signed = line_signed;
        state.has_prev_line_signed = true;

        // ---- ID 交接（req3）：新 ID 在旧 ID 刚失联的位置附近出生时继承其上下文 ----
        // 只继承 line_side（= 上一次确认的所在侧）与区域/脚底上下文，**不再继承
        // entered/exited 一次性封顶**：只要侧不翻转就不会产生事件，此后每一次真实
        // 跨线都会重新计数。（旧实现继承封顶 + "进出自计满就跳过交接"正是单人多次
        // 往返被连续漏计的根因。）
        // ID 交接资格的硬门：**必须真的是本模块第一次见到这个 track_id**。
        // 若"见过但状态已过期"，说明它是被 tracker 重连复活的旧轨迹（不是新生的人）——
        // 它不得继承、更不得封存附近真实轨迹（现场 2026-09-18：复活的 18 把真实轨迹 21 封存，
        // 导致真人这一次跨线连事件都没有）。正常报新 ID 的情形不受影响。
        if (is_new_track && !first_time_seen) {
            char buf[224];
            std::snprintf(buf, sizeof(buf),
                "revived-id track=%d 曾被计数器见过但状态已过期 → 判为被 tracker 重连复活的旧轨迹，"
                "不参与 ID 交接(不继承也不封存他人)", track.track_id);
            notes_.push_back(buf);
        }
        if (is_new_track && first_time_seen) {
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
                // ★★ 封存前先兑现旧 ID 的跨线证据（2026-09-18 修复缺陷 A，文档 §22.3）★★
                // 为什么必须在这里结算：`!state.sealed` 是**正常翻转路径**(:728) 与
                // **消失回补路径**(:767) 的硬门，一旦 sealed，旧 ID 那份"已观测到线的另一侧、
                // 但没凑够 kLineSideConfirmFrames 帧确认"的证据就永久作废；若继承者只落在
                // 死区内就消失 → **整次跨线凭空消失**（现场 P21：`track=21` 最后观测
                // signed=-129px 却一条事件都没有，而跨线最后归到了复活的 18 名下）。
                //
                // 证据口径与消失回补**完全一致**（不是新增一类推断）：最后一次**被接受的观测**
                // 脚底已在线的另一侧 ≥ 死区 —— 即用户裁定的"三条合法证据"第 ③ 条
                // （单帧观测，最弱但仍是观测）。质量门同样与回补对齐。
                //
                // ⚠️ 顺序：结算必须在【捕获 old_side / 继承字段】之前。否则继承者会拿到
                //    "结算前的旧侧"，下一帧再把同一次跨线确认一遍 → **双计**。
                bool old_settled = false;
                {
                    const int old_side_before = best->line_side;
                    const int old_implied =
                        (best->line_signed <= -kCrossLineDepth) ? 0
                      : (best->line_signed >= kCrossLineDepth) ? 1 : -1;
                    if (best->line_active && old_side_before >= 0 &&
                        old_implied >= 0 && old_implied != old_side_before) {
                        const bool old_quality_ok =
                            best->last_hit_streak >= kMinHitStreak &&
                            best->last_confidence >= kMinConfidence;
                        if (old_quality_ok) {
                            if (old_implied == 1) {
                                best->entered = true;
                                events.push_back(makeEvent(
                                    best_id, enter_type, best->last_foot, best->region,
                                    best->origin_id, false, true, best->lost_frames,
                                    "enter-cross-line-on-seal"));
                            } else {
                                best->exited = true;
                                best->inside_region = false;
                                events.push_back(makeEvent(
                                    best_id, leave_type, best->last_foot, best->region,
                                    best->origin_id, best->exit_via_entry_band, true,
                                    best->lost_frames,
                                    "leave-cross-line-on-seal"));
                            }
                            // 提交后 line_side 与 line_signed 自洽（implied 由 line_signed 推出），
                            // 不会出现"过期 line_signed 推出相反侧"的逐帧横跳。
                            best->line_side = old_implied;
                            old_settled = true;
                        }
                        char buf[320];
                        std::snprintf(buf, sizeof(buf),
                            "sealed-with-pending 旧 ID=%d 带未确认跨线证据(side %d→%d, signed=%+.0fpx) 被继承封存"
                            " → %s",
                            best_id, old_side_before, old_implied, best->line_signed,
                            old_settled ? "已在封存前兑现(该次跨线已计数)"
                                        : "质量门未过 → 未兑现(这次仍会漏)");
                        notes_.push_back(buf);
                    }
                }
                const int old_side = best->line_side;   // ★ 结算**之后**的侧（决定了继承给谁）
                const bool segment_cross = best->has_last_foot &&
                    segmentCrossesLine(best->last_foot, foot);
                state.line_side          = best->line_side;
                state.line_side_pending  = best->line_side;
                state.line_side_pending_frames = kLineSideConfirmFrames;
                state.line_flip_cooldown = best->line_flip_cooldown;
                state.ever_in_R2       = best->ever_in_R2 || best->inside_region;
                state.inside_region    = best->inside_region;
                state.entered          = best->entered;
                state.exited           = best->exited;
                state.exit_via_entry_band = best->exit_via_entry_band;
                state.region           = best->region;   // 沿用旧 ID 的区域，避免瞬时跳变
                state.pending_region   = best->region;
                state.pending_frames   = kRegionConfirmFrames;
                state.outside_clear    = false;
                state.outside_frames   = 0;
                state.has_ref_foot     = best->has_ref_foot;
                state.ref_foot         = best->ref_foot;
                state.has_last_foot    = best->has_last_foot;
                state.last_foot        = best->last_foot;
                state.has_last_inside_foot = best->has_last_inside_foot;
                state.last_inside_foot = best->last_inside_foot;
                state.inherited        = true;
                state.origin_id        = (best->origin_id >= 0) ? best->origin_id : best_id;
                // ★★ M4：**不再当场封存旧 ID**，改为"延迟封存"（登记考察期）★★
                // 理由（用户 2026-09-18）：封存是**不可逆**的 —— 一旦 sealed，旧 ID 此后的一切
                // 真实观测都会被 `!sealed` 这道硬门静默（正常翻转 / 消失回补 / 交接兑付三条路径
                // 全被挡住）。可是"这个刚出生的 ID 究竟是一个会走路的真人，还是检测区里一件
                // 静止物品（现场：门口饮水机）的误检框"，在它出生的这一瞬间**根本无法判断**。
                //   ⇒ 先给它 kSealProbationFrames 帧，让**它自己走出来的轨迹**来背书：
                //      走出 ≥ kSealMinTravel → 确认是会动的真实目标 → 正式封存旧 ID；
                //      原地没动（位移被 kMoveMinStep 滤成个位数）→ 判定为静止目标，
                //      **放弃封存** → 旧 ID 保持活跃，走它自己的"消失回补"，一条真实轨迹都不会丢。
                // ⚠️ 延迟封存的副作用已核对：旧 ID 的待确认证据在继承前就结算过（见上），
                //    且结算把它的 line_side 提交到了结算后的侧 → 之后它的"消失回补"不会再补一次，
                //    新 ID 继承到的也是这个新侧 → 两者都不会对同一次跨线重复计数。
                state.seal_target_id = best_id;
                state.seal_probation  = kSealProbationFrames;
                // 被继承方：跨线账已移交 → 冻结它的 lost 补发（防与新 ID 的兑付双计）。
                // 注意它**不是** sealed，正常侧翻转（需要新观测）不受影响。
                best->handover_done = true;

                // If the ID changed exactly while crossing t1, the normal
                // side-confirmation window cannot observe the flip. Use the
                // two real endpoints to emit the missing event immediately.
                int current_side = -1;
                if (line_signed >= kCrossLineDepth) current_side = 1;
                else if (line_signed <= -kCrossLineDepth) current_side = 0;
                // ⚠️ 兑付**不能**用 `!state.entered` / `!state.exited` 做门。那等于在交接路径上
                // 偷偷恢复"每人 1 进 1 出"的一次性封顶（T17 已废弃），而 entered/exited 又是
                // 从旧 ID 继承来的 → 人第 2 趟起在线上换 ID 时兑付被吞掉；更糟的是下面的
                // line_side 提交仍会执行，于是这条真实跨线既没发事件、正常路径也再补不回来。
                // 兑付与"侧翻转边沿"同源：判据是"两个真实脚底连线确实穿过 t1 且侧发生了变化"，
                // 每个跨越动作恰好一次；同帧已把 line_side 提交为 current_side，故正常路径
                // 不会再对同一次跨越重复发事件（不会双计）。
                // !old_settled：上面已按"最后一次观测"给旧 ID 结算过一次跨越，
                // 同一次跨越不得再由本 ID 兑付一次（两条路径实测互斥，此门是显式保险）。
                if (segment_cross && current_side >= 0 &&
                    current_side != old_side && quality_ok && !state.sealed &&
                    !old_settled) {
                    if (current_side == 1) {
                        state.entered = true;
                        events.push_back(makeEvent(track.track_id, enter_type, foot,
                                                   state.region, best_id, false, false, 0,
                                                   "handover-segment-enter"));
                    } else {   // current_side == 0（old_side ≥ 0 且两者不等 ⇒ 只能取 0/1）
                        state.exited = true;
                        state.inside_region = false;
                        events.push_back(makeEvent(track.track_id, leave_type, foot,
                                                   state.region, best_id,
                                                   state.exit_via_entry_band, false, 0,
                                                   "handover-segment-leave"));
                    }
                    state.line_side = current_side;
                    state.line_side_pending = current_side;
                    state.line_side_pending_frames = kLineSideConfirmFrames;
                }
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
                    state.line_side_pending_frames++;   // ★不封顶：还要用它判"站住"
                } else {
                    // 诊断：候选侧被反向覆盖 → 本次跨线作废。分两档，现场一眼看出是哪种：
                    //   ① < kLineSideConfirmFrames   = "差一点就计数"（1~2 帧的线上抖动）
                    //   ② 已确认但没站住(< cross_dwell) = "在线上蹭/徘徊"（跨过去又回来）
                    if (state.line_side_pending >= 0 &&
                        state.line_side_pending != raw_side &&
                        state.line_side_pending_frames > 0 &&
                        !state.sealed) {
                        char buf[224];
                        if (state.line_side_pending_frames < kLineSideConfirmFrames) {
                            std::snprintf(buf, sizeof(buf),
                                "cand-abandoned track=%d cand_side=%d seen=%d帧(需%d) 被 side=%d 覆盖"
                                " → 本次跨线不计",
                                track.track_id, state.line_side_pending,
                                state.line_side_pending_frames, kLineSideConfirmFrames, raw_side);
                            notes_.push_back(buf);
                        } else if (state.line_side_pending != state.line_side &&
                                   state.line_side_pending_frames < cross_dwell_frames_) {
                            std::snprintf(buf, sizeof(buf),
                                "flip-abandoned track=%d side=%d 已确认但只站住 %d 帧(需%d)"
                                " 就回到 side=%d → 在线上蹭/徘徊，本次跨线不计",
                                track.track_id, state.line_side_pending,
                                state.line_side_pending_frames, cross_dwell_frames_, raw_side);
                            notes_.push_back(buf);
                        }
                    }
                    state.line_side_pending = raw_side;
                    state.line_side_pending_frames = 1;
                }
                const bool confirmed =
                    state.line_side_pending_frames >= kLineSideConfirmFrames;
                // ★"站住"：新的一侧被连续观测 ≥ cross_dwell_frames_ 帧才允许结算。
                //   = kLineSideConfirmFrames 时与历史行为逐位一致（见类头注释）。
                const bool committed =
                    state.line_side_pending_frames >= cross_dwell_frames_;

                if (confirmed && state.line_side < 0) {
                    // 诊断：首次确立所在侧时脚底已离线很远 → 这个 ID "出生"在线的某一侧深处，
                    // 说明它被观测到之前的那段（也就是真正跨线的那几帧）根本没进计数模块。
                    if (std::fabs(line_signed) >= kNoteFarFromLinePx) {
                        char buf[224];
                        std::snprintf(buf, sizeof(buf),
                            "first-side track=%d 首次确立 side=%d 时 signed=%+.0fpx、已有 %d 帧观测"
                            " → 出生在%s，跨线那几帧没被看到，不会计进入类",
                            track.track_id, state.line_side_pending, line_signed,
                            state.obs_frames, state.line_side_pending == 1 ? "线内侧" : "线外侧");
                        notes_.push_back(buf);
                    }
                    state.line_side = state.line_side_pending;   // 首次确立，不发事件
                } else if (committed && state.line_side_pending != state.line_side &&
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

        // ---- M4：延迟封存考察期推进（决定"封存"还是"放弃"）----
        // 到这一步，这个新 ID 已经用**自己的轨迹**给出了答案：
        //   走出 ≥ kSealMinTravel  → 是会动的真实目标 → 正式封存旧 ID（确认这次交接）；
        //   考察期满仍在原地      → 静止目标（误检框 / ID 抖动）→ **放弃封存**，
        //                          旧 ID 保持活跃，继续走它自己的观测路径，一条真实轨迹都不丢。
        // 另有两种"本次交接作废"的情形：旧 ID 自己又活了（说明它根本没终止）、或待封存对象已消失。
        if (state.seal_probation > 0 && state.seal_target_id >= 0) {
            auto oit = states_.find(state.seal_target_id);
            State* old = (oit != states_.end()) ? &oit->second : nullptr;
            char buf[256];
            if (!old) {
                state.seal_probation = 0;
                state.seal_target_id = -1;
            } else if (old->sealed || old->lost_frames == 0) {
                // 旧 ID 又出现在活跃输出里 → 它根本没终止，交接的前提不成立
                std::snprintf(buf, sizeof(buf),
                    "seal-cancelled track=%d 待封存的旧 ID=%d %s → 本次交接作废，旧 ID 保持活跃",
                    track.track_id, state.seal_target_id,
                    old->sealed ? "已被别的 ID 封存" : "又回到活跃输出(它没终止)");
                notes_.push_back(buf);
                state.seal_probation = 0;
                state.seal_target_id = -1;
            } else if (state.travel_px >= kSealMinTravel) {
                std::snprintf(buf, sizeof(buf),
                    "seal-confirmed track=%d 考察期内累计位移 %.0fpx ≥ %.0fpx → 确认是会动的真实目标，"
                    "正式封存旧 ID=%d",
                    track.track_id, state.travel_px, kSealMinTravel, state.seal_target_id);
                notes_.push_back(buf);
                old->sealed = true;
                old->inside_region = false;
                state.seal_probation = 0;
                state.seal_target_id = -1;
            } else if (--state.seal_probation <= 0) {
                std::snprintf(buf, sizeof(buf),
                    "seal-cancelled track=%d 考察 %d 帧内累计位移仅 %.0fpx(<%.0fpx) → 判为静止目标"
                    "(误检框/ID 抖动)，放弃封存旧 ID=%d",
                    track.track_id, kSealProbationFrames, state.travel_px, kSealMinTravel,
                    state.seal_target_id);
                notes_.push_back(buf);
                state.seal_target_id = -1;
            }
        }

    }

    // ---- 消失回补 + 跟丢越线推断（lost 循环）----
    // 跨线事件在"确认翻转"的那一刻就发出，正常跨线无需等 ID 消失。本循环兜两种漏计：
    //   ① 消失回补：脚底已明确跨到线的另一侧，但翻转确认帧数(< kLineSideConfirmFrames)
    //      还没凑够就消失在活跃输出里 → 按"已完成一次跨线"补发一次事件；
    //   ② 跟丢越线推断（本文件 kInfer* 常量）：人正跨线那一瞬间彻底跟丢、且之后再没
    //      有后继 ID 可交接 → 用最后观测的帧间速度外推，判定其是否已完成这次跨线。
    // 两者都只补一次：补发后立即提交 line_side，后续帧 implied 与 line_side 相同，不再重复。
    for (auto it = states_.begin(); it != states_.end();) {
        State& state = it->second;
        if (active_ids.count(it->first)) { ++it; continue; }

        state.lost_frames++;

        // M4：正在"待封存考察期"的 ID **自己失联了** → 无法再验证它的运动性 → 本次交接作废。
        // 考察期的本意就是"看这个新 ID 会不会动"；它自己都没了，就无从谈起，更不该
        // 进一步去封存别人（继承与 on-seal 结算已经发生，那部分是必要的，不受影响）。
        if (state.seal_probation > 0) {
            state.seal_probation = 0;
            if (state.seal_target_id >= 0) {
                char buf[224];
                std::snprintf(buf, sizeof(buf),
                    "seal-cancelled track=%d 在封存考察期内自身失联 → 放弃封存旧 ID=%d",
                    it->first, state.seal_target_id);
                notes_.push_back(buf);
            }
            state.seal_target_id = -1;
        }

        if (line_pts_.size() >= 2 && state.line_active && state.line_side >= 0 &&
            !state.sealed && !state.handover_done &&
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
            } else if (infer_lost_crossing_ &&                    // ★开关（默认关，见类头注释）
                       state.line_side == 1 &&                       // ① 曾确证在线内侧
                       state.line_signed <= kInferNearLinePx &&      // ② 最后观测离线段不远
                       state.line_signed >= -kCrossLineDepth &&      //    (未明确到线外)
                       state.line_vel <= -kInferMinOutwardPx &&      // ③ 仍在朝线外移动
                       state.lost_frames <= kInferMaxLostFrames &&   //    外推窗口有界
                       framesToCross(state.line_signed, state.line_vel)      // ④ 跨线在望
                           <= (float)kInferMaxReachFrames &&
                       state.line_signed + state.line_vel * (float)state.lost_frames
                           <= -kCrossLineDepth) {                    //    外推结果已跨到线外
                // 跨线瞬间跟丢且无后继 ID → 推断其已完成一次离开类跨线（settled=true）。
                state.exited = true;
                state.inside_region = false;
                events.push_back(makeEvent(it->first, leave_type, state.last_foot,
                                           state.region, state.origin_id,
                                           state.exit_via_entry_band, true,
                                           state.lost_frames,
                                           "leave-cross-line-inferred"));
                // 必须把 line_side 与 line_signed 一并推进到线外，保持状态自洽：
                // 否则下一帧上面的 implied 仍从过期的 line_signed(线内)推出 1，与 line_side=0
                // 冲突 → 会在"消失回补/推断"之间逐帧反复横跳(实测已复现)。
                state.line_side = 0;
                state.line_signed = -kCrossLineDepth;
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
