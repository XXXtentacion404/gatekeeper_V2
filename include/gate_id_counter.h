#ifndef GATEKEEPER_GATE_ID_COUNTER_H
#define GATEKEEPER_GATE_ID_COUNTER_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "types.h"
#include "bytetrack/tracker.h"

// 摄像头角色：决定"进入类 / 离开类"两种底层事件如何映射成 IN / OUT / RETRO。
// 两个摄像头的判定逻辑完全对称，角色只负责给事件贴语义标签：
//   - 入口摄像头(201)：进入类 = IN，离开类 = RETRO
//   - 出口摄像头(200)：进入类 = RETRO，离开类 = OUT
enum GateRole {
    GATE_ROLE_ENTRY = 0,  // 入口摄像头(201)
    GATE_ROLE_EXIT  = 1,  // 出口摄像头(200)
};

enum GateEventType {
    GATE_EVENT_IN    = 0,  // 进入
    GATE_EVENT_OUT   = 1,  // 离开
    GATE_EVENT_RETRO = 2,  // 逆行
};

// 脚底点所在区域。出口区域已删除：检测区多边形之外一律是 R0(= 出口)。
enum GateRegion {
    GATE_REGION_OUTSIDE = 0,  // 区域外（= 出口侧）
    GATE_REGION_ENTRY   = 1,  // 入口带（t1 整体沿 x 方向平移 |entry_zone_depth| px 的带）
    GATE_REGION_DETECT  = 2,  // 检测区内（检测区多边形）
};

struct GateCountEvent {
    int track_id = -1;
    int type = GATE_EVENT_IN;
    cv::Point2f foot;
    int region = GATE_REGION_OUTSIDE;  // 事件判定时该 ID 的确认区域
    int origin_id = -1;                // 交接来源 ID（-1 = 本 ID 自己走完全程）
    bool via_entry_band = false;       // 离开是否经过入口带（2→1→0 逆行离开）
    bool settled = false;              // 是否为"轨迹在区域内消亡"的终审补发
    int lost_frames = 0;               // 判定时该 ID 已消失帧数
    std::string reason;                // 判定依据（日志/trace 用）
};

// 每个 ID 的区域标记（req1：区域内/区域外都要有标记）。
struct GateIdMark {
    int track_id = -1;
    int region = GATE_REGION_OUTSIDE;  // 当前确认区域 0/1/2
    bool inside = false;               // 当前是否标记为"在区域内"
    bool entered = false;              // 已发进入类事件
    bool exited = false;               // 已发离开类事件
    bool inherited = false;            // 本 ID 由交接继承而来
    int origin_id = -1;                // 继承来源 ID
    int lost_frames = 0;               // 连续未出现在活跃输出的帧数
    // ---- 跨线几何（只看不改：用于回答"这个人离本条相机的判定线还有多远"）----
    // 现场排查"某台相机全程 0 事件"时必须看这三个量：脚底投影若一直在跨度外
    //（line_active=false），说明人根本没走到这条线覆盖的那段边界上，而不是逻辑漏计。
    float line_signed = 0.0f;          // 脚底到判定线的带符号垂距(px)：>0 = 线内侧
    int line_side = -1;                // -1 未定 / 0 线外侧 / 1 线内侧
    bool line_active = false;          // 脚底投影是否落在判定线跨度内
};

// 跨标记线判定 + 区域标记 + ID 交接的双向闸机计数。
//
// 几何模型（脚底点判定，出口区域已删除）：
//   R0 区域外   ：脚底既不在入口带、也不在检测区多边形内  → 出口侧
//   R1 入口区域 ：脚底在 t1 入口带内（zonePolygon(t1, entry_zone_depth)）
//   R2 检测区域 ：脚底在检测区多边形内（入口带之外）
//   区域优先级：R1 入口带 > R2 检测区 > R0（入口带优先，避免交叠处歧义）
//
// ★ 触发源 = 本相机标记线(t1)的"侧翻转边沿"（不再按 ID 生命周期封顶）★
//   每台相机只对自己画的那条标记线负责，脚底到 t1 的带符号距离
//   line_signed 决定所在侧：>0 线内侧（检测区侧），<0 线外侧（真·外面）。
//   每次**确认的侧翻转**恰好产生一次事件（角色决定标签）：
//     线外 → 线内 = 进入类    201 → IN         200 → RETRO
//     线内 → 线外 = 离开类    201 → RETRO      200 → OUT
//   关键推论：
//     - 同一 ID 可以反复往返，**每跨一次线就计一次**（旧的"每人至多 1 进 1 出"封顶已废弃，
//       它会让单人多次往返被漏计，且交接会把这个封顶一并继承 → 连续漏计）；
//     - 人从检测区**对面那一侧**进出时，在本相机上 line_signed 全程同号，
//       不产生任何翻转 → 本相机 0 事件，杜绝"201 记了本该 200 记的 OUT"这类串台；
//     - 首次观测只"确立所在侧"而不发事件：出生在检测区内的 ID 不会凭空计 IN。
//
// 去抖与鲁棒（防止同一次跨线被重复/抖动计数）：
//   1) 死区滞回：|line_signed| < kCrossLineDepth 的线上抖动带内不改侧；
//   2) 翻转确认：候选侧连续 kLineSideConfirmFrames 帧才确认翻转（并叠加短冷却）；
//   3) 跨度约束：脚底投影须落在 t1 线段跨度内（±kLineSpanMargin），
//      避免"人在闸口侧面远处横向走动"被无限延长线判成跨线；
//   4) 遮挡脚底校正：框底被遮挡抬升时用 per-ID 地面基准冻结，遮挡不产生假翻转；
//   5) 消失回补：已跨到线另一侧、但确认帧数没凑够就消失在活跃输出里的，按"已完成
//      一次跨线"补发一次（settled=true），堵住"跨线瞬间丢帧"的漏计；
//   6) 封存前结算（2026-09-18 修复缺陷 A）：ID 交接封存旧 ID **之前**，先把它
//      "最后一次观测已到线另一侧 ≥ 死区"的证据兑现成事件，再把结算后的所在侧继承给新 ID。
//      不加这一步的话，封存会让这份证据永久作废（`!sealed` 是翻转与回补的硬门），
//      而继承者若只落在死区内就消失 → **整次跨线丢失**（现场 P21：最后观测 signed=-129px
//      却一条事件都没有）。口径与 5) 完全一致，属"三条合法证据"的第 ③ 条，不是推断。
//      ⚠️ 结算必须在"捕获旧侧 / 继承字段"**之前**，否则继承者会带过期侧再计一次（双计）。
//   7) 复活 ID 识别（2026-09-18）：**只有"本模块第一次见到"的 track_id 才有资格做 ID 交接**。
//      被 tracker 重连复活的旧轨迹（见过、但状态已过期）一律不继承、也不封存他人 ——
//      它并不是新出现的人，没有权利终结一条真实轨迹。该门只依赖"见过哪些 ID"的短期记忆，
//      与状态保留期是否够长解耦（现场：复活的 18 继承并封存了真实的人 21）。
//   8) 解封（2026-09-18，M3）：交接的前提是"旧 ID 已终止"。若一个**已被封存**的 ID 此后又连续
//      kUnsealConfirmFrames 帧收到自己的**真实观测**（预测框不算），前提即不成立 → 那次交接
//      是误判（典型：附近假框把真人顶掉，反倒把真人封存）→ 解封并恢复计数资格。
//      解封时把 line_side 与待确认侧一并作废，**绝不拿封存前的旧证据补计数** → 下一次真实
//      跨线才计，天然不与继承者双计。不加这一步的话，被误封 ID 的一切真实轨迹会被
//      `!sealed` 硬门永久静默。
//
// 区域序列(0/1/2) 与 inside_region 标记只用于显示/日志，不再决定事件。
// ID 交接(handover) 只继承 line_side（所在侧）与区域上下文，**不再继承 entered/exited**：
// 只要侧不翻转就不产生事件，此后每一次真实跨线都会重新计数。
class GateIdCounter {
public:
    // 跨线"站住"确认帧数的默认值：与 kLineSideConfirmFrames 相同 ⇒ 默认行为与历史逐位一致。
    static constexpr int kDefaultCrossDwellFrames = 3;

    // 两点直线版（= 旧的 t1）。内部转成 2 点折线，行为与折线版逐位一致。
    GateIdCounter(GateRole role, const Tripwire& entry_line,
                  float entry_zone_depth, const cv::Mat& detection_mask,
                  int cross_dwell_frames = kDefaultCrossDwellFrames);
    // 折线判定线版：区域边缘本身就是判定线，点数不限（≥2），点重合会被剔除。
    // 用 2 个点时与上面的直线版完全等价。
    GateIdCounter(GateRole role, const std::vector<cv::Point2f>& line_pts,
                  float entry_zone_depth, const cv::Mat& detection_mask,
                  int cross_dwell_frames = kDefaultCrossDwellFrames);

    // ---- 跨线"站住"确认（cross_dwell_frames，2026-09-18）----
    // 背景：一次"确认的侧翻转"（连续 kLineSideConfirmFrames 帧脚底在线另一侧）只证明
    //   "脚曾经到过另一边"，不证明"人真的过去了"。现场实测：人在门口线上小幅徘徊
    //   （跨过去 0.75s 又回到原侧）会各计一次离开+进入，日志里表现为
    //   RETRO 之后 752ms 紧跟一个 IN（201 的 IN 3→4），用户判定为误计。
    // 规则：翻转只有在**新的一侧被连续观测 ≥ cross_dwell_frames 帧**之后才结算
    //   （发事件 + 推进 line_side）。若中途脚底回到原来那一侧 → 本次跨线作废：
    //   不发事件、line_side 不变（记 flip-abandoned 注记）。这样"蹭一下再回来"这一对
    //   事件被整体取消，进出计数保持配平。
    // ⚠️ 纯"冷却"解决不了这个问题：冷却期过后人还停在新侧，事件仍会被补发（只是晚一点）。
    //    必须让"没站住"的翻转候选被真正作废。
    // 取值影响：太小 → 徘徊照样计；太大 → 真人快速穿过（在新侧停留不足）会被漏计
    //   （此时可由「消失回补/交接兑付」两条独立证据兜底）。
    // 默认 3（= 历史行为，168 项测试走这条）；板端用更大值，见 src/main_tracking.cpp。

    // ---- 跟丢越线推断开关（★默认关闭★）----
    // 关闭（默认）⇒ 所有 IN/OUT/RETRO 都必须由一个**真实跨线观测**支撑，三条合法证据：
    //   ① 连续 kLineSideConfirmFrames 帧观测到脚底在线的另一侧（≥ 死区 20px）；
    //   ② 两个真实脚底（旧 ID 末帧 → 新 ID 首帧）的连线确实穿过判定线（交接兑付）；
    //   ③ 最后一次观测的脚底已经在线的另一侧 ≥ 死区（消失回补）。
    //   没有观测 ⇒ 不计数（宁可漏计，绝不虚计）。
    // 打开 ⇒ 额外允许"跟丢越线推断"在**跨线在望(≤ kInferMaxReachFrames 帧就越过死区)**时
    //   补发一次离开类。⚠️ 这是一次**没有实时观测依据**的补计数（只是运动趋势外推）。
    //
    // 为什么默认关闭（2026-09-17 用户裁定）：板端 200 出口相机现场出现过"人从区域外进区、
    //   在区内转圈、走到判定线末端附近但没跨出去，跟丢后却被补了一次 OUT"（OUT:6→7）。
    //   校准后的判据（§19：need ≤ 5 帧）虽然已能拒掉那一次，但"没有观测就补计数"这一类
    //   机制本身不被接受 —— 门口被遮挡、跨线瞬间被吞都是**低频**场景，不值得用虚计风险去换。
    // 保留开关（而不是删代码）是为了将来做"推断 + 实时观测"的混合方案时能直接复用机制。
    void enableLostCrossingInference(bool on) { infer_lost_crossing_ = on; }
    bool lostCrossingInferenceEnabled() const { return infer_lost_crossing_; }

    std::vector<GateCountEvent> update(const std::vector<TrackedObject>& tracks);

    // 脚底点所在区域：0=区域外(出口侧) 1=入口带 2=检测区内（供显示/调试）
    int regionOf(const cv::Point2f& pt) const;
    // 到检测区边界的有符号距离(px)：>0 在检测区内，<0 在区域外。
    // 无检测区多边形时回退到"标记线二分"（>0 = 入侧）。
    float regionDistance(const cv::Point2f& pt) const;

    // 每个 ID 的区域标记（req1：区域内/区域外都有标记，画面与日志用）
    std::vector<GateIdMark> marks() const;
    // 当前被标记为"在区域内"的 ID 数量（被交接封存的旧 ID 不计入）
    int insideCount() const;

    // 判定线折线（= 区域边缘）与区域环 —— 供显示/调试直接取用，
    // 保证画面上画的几何与计数内部用的几何**同源**（不会各画各的）。
    const std::vector<cv::Point2f>& judgingLine() const { return line_pts_; }
    const std::vector<cv::Point2f>& zoneRing() const { return entry_zone_; }

    // ---- 诊断注记（只读，不影响任何计数语义）----
    // 每次 update() 清空重填，记录"为什么这个人没被计数/证据为什么被丢"这类关键瞬间：
    //   predicted-frame   ：本帧只有预测框(observed=false) → 计数模块看不到这一帧
    //                       （画面上却有框 → 现场"视频里明明有轨迹却没计数"的头号原因）
    //   first-side        ：某 ID 首次确立所在侧时脚底已离线很远 → 出生在区内/门口观测缺口
    //   cand-abandoned    ：一次跨线候选只观测到 1~2 帧（< kLineSideConfirmFrames）就被覆盖
    //   flip-abandoned    ：跨线候选**已确认**（≥ kLineSideConfirmFrames）但没"站住"
    //                       （< cross_dwell_frames）就被反向覆盖 → 在线上蹭/徘徊，本次不计
    //   sealed-with-pending：交接封存了带着"未确认跨线证据"的旧 ID —— **该证据已在
    //                       封存前兑现**（若注明"质量门未过"则未兑现，这次仍会漏）
    const std::vector<std::string>& notes() const { return notes_; }

    void reset();

private:
    struct State {
        // ---- 事件（仅记录"是否发过"，不再作为触发门限）----
        bool entered = false;
        bool exited  = false;

        // ---- 区域序列状态机（仅用于显示标记/日志）----
        int region = GATE_REGION_OUTSIDE;  // 当前确认区域 0/1/2
        int pending_region = -1;           // 候选区域（待确认）
        int pending_frames = 0;            // 候选区域连续帧数

        // ---- 本相机标记线(t1)跨线判定（角色隔离）----
        // line_signed > 0 = 线内侧(检测区侧)，< 0 = 线外侧(真·外面)。
        float line_signed = 0.0f;      // 最近一次脚底到 t1 的带符号距离(px)
        bool line_outside = false;     // 当前是否"已在 t1 外侧 ≥ kCrossLineDepth"
        bool line_active = false;      // 脚底投影是否落在 t1 线段跨度内

        // ---- 侧翻转边沿检测（唯一触发源）----
        // line_side: -1 未定 / 0 线外侧 / 1 线内侧。
        // 每次"确认的侧翻转"恰好产生一次事件 → 同一 ID 可反复往返计数。
        int line_side = -1;
        int line_side_pending = -1;
        // 候选侧已连续观测的帧数。**不封顶**：≥ kLineSideConfirmFrames 表示"确认翻转"，
        // ≥ cross_dwell_frames_ 才表示"站住了"（可结算）。见类头"跨线站住确认"注释。
        int line_side_pending_frames = 0;
        int line_flip_cooldown = 0;    // 翻转后冷却帧数（防线上抖动连发）

        // ---- 区域标记（req1：区域内/区域外，粘滞）----
        bool inside_region = false;   // 该 ID 当前标记为"在区域内"
        bool ever_in_R2 = false;      // 是否曾进入检测区（粘滞）

        // ---- "清晰在外"（仅用于显示标记翻转 inside_region，不参与触发）----
        bool outside_clear = false;
        float outside_depth = 0.0f;
        int outside_frames = 0;
        bool exit_via_entry_band = false; // 离开前最后一站是否为入口带（逆行离开标记）

        // ---- ID 交接（req3）----
        bool sealed = false;          // 已被新 ID 继承：不再发事件、不再被继承、标记翻转为区域外
        bool inherited = false;       // 本 ID 的状态来自继承
        int origin_id = -1;           // 继承链根 ID
        // 解封计数（M3，2026-09-18）：被封存的 ID 连续收到自己的真实观测帧数。
        // 达到 kUnsealConfirmFrames 即认定"当初那次交接是误判" → 解封并恢复计数资格。
        int unseal_streak = 0;

        // ---- M4：轨迹账本 + 延迟封存（2026-09-18，"用运动性排除静止误检"）----
        // 累计有效位移：锚点法累加（与锚点距离 ≥ kMoveMinStep 才累加并推进锚点）。
        // 只统计 observed=true 的真实帧，且在假框守卫之后 → 被丢弃的帧不污染轨迹。
        // 真人走过去 → 几百 px；静止误检（饮水机）→ 长期停在个位数。
        float travel_px = 0.0f;
        bool has_travel_anchor = false;
        cv::Point2f travel_anchor{0.0f, 0.0f};
        // 延迟封存：本 ID 出生时若继承了旧 ID，**不当场封存它**，而是登记"待封存"并进入考察期；
        // 考察期内本 ID 自己走出 ≥ kSealMinTravel 的轨迹才确认封存，否则放弃（见 update()）。
        int seal_target_id = -1;      // 待封存的旧 ID（-1 = 无）
        int seal_probation = 0;       // 剩余考察帧数
        // 被继承方："跨线账"已结清标记（**只是不再自行补历史，不等于封存**）。
        // 继承一旦发生，这条 ID 当前这次跨线就交给继任者负责了：
        // 它会冻结自己在 lost 循环里的**补发**（-on-loss / 跟丢推断），避免与新 ID 的
        // 交接兑付对同一次跨线各计一次（实测：不冻结会双计）。
        // ⚠️ 它**不冻结正常侧翻转路径** —— 那需要新的真实观测才能触发，属于"新的一次跨线"：
        //    这条 ID 若其实是真人（没有被真正接手），回来照样正常计数。
        // ⚠️ 与 sealed 的区别：sealed 是"已被确认接手 → 彻底静默"；本标记只是"账已移交"。
        bool handover_done = false;

        // ---- 生命周期 / 检测质量 ----
        int lost_frames = 0;          // 连续未出现在活跃输出的帧数
        int last_result_frame_id = -1;
        int last_hit_streak = 0;
        float last_confidence = 0.0f;
        int obs_frames = 0;           // 被接受的"真实观测"帧数（仅诊断注记用）

    // ---- 跟丢越线推断（治"跨线瞬间跟丢且无后继 ID"的整段漏计）----
    // ★默认关闭（infer_lost_crossing_ = false）★：见上方 enableLostCrossingInference() 的说明。
    // 用"最近两次被接受的观测"之间 line_signed 的变化率(px/帧)：>0 朝线内，<0 朝线外。
    // 必须用帧间速度而非平均速度：走到门口后站定再消失时平均速度仍朝外会被误判。
    float line_vel = 0.0f;
    float prev_line_signed = 0.0f;
    bool has_prev_line_signed = false;

        // ---- 脚底点（含遮挡抬升校正基准）----
        cv::Point2f last_foot{0.0f, 0.0f};
        bool has_last_foot = false;
        bool has_ref_foot = false;
        cv::Point2f ref_foot{0.0f, 0.0f};
        bool has_last_inside_foot = false;
        cv::Point2f last_inside_foot{0.0f, 0.0f};
    };

    // ---- 判定线求值（折线版）----
    // 判定线 = 区域边缘折线；"跨过判定线" = 带符号侧发生翻转。
    struct LineEval {
        float signed_dist = 0.0f;   // 到【最近线段所在直线】的带符号垂距：>0 = 线内侧
        bool  in_span = false;      // 是否落在折线跨度内（只有两个端段才做余量约束）
        int   seg = -1;             // 命中的线段下标
    };
    LineEval evalLine(const cv::Point2f& pt) const;
    // 第 seg 段的"内侧"单位法向：右手系法向 (dy,-dx)/L，再按 depth 符号定向。
    cv::Point2f inwardNormal(int seg) const;
    // 折线弧长中点（2 点直线时就是线段中点，与旧实现一致）；也用于法向自校正的探测点。
    cv::Point2f lineMidPoint() const;
    int midSegment() const;
    // 区域环 = 判定线 ∪ 其沿 x 平移 offset 的副本，首尾相接的闭环。
    // offset 的符号 = 内侧法向 x 分量所指的那一侧（保证带落在检测区一侧）。
    std::vector<cv::Point2f> zonePolygon(float offset) const;
    float zoneOffsetX() const;
    bool pointInPolygon(const cv::Point2f& pt,
                        const std::vector<cv::Point2f>& polygon,
                        float tolerance) const;
    GateRole role_;
    // 跨线"站住"确认帧数（≥1）。见类头"跨线站住确认"注释 + 构造参数说明。
    int cross_dwell_frames_ = kDefaultCrossDwellFrames;
    // 跟丢越线推断开关：默认 false（= 只认观测）。见 enableLostCrossingInference()。
    bool infer_lost_crossing_ = false;
    // 判定线：折线（≥2 点）。2 点即旧的直线 t1，此时所有算法都退化成旧实现。
    std::vector<cv::Point2f> line_pts_;
    float entry_depth_;                        // |depth|；<1 视为"无区域"
    float entry_depth_signed_;                 // 带符号深度：内侧法向指向检测区侧
    cv::Mat detection_mask_;
    std::vector<cv::Point2f> entry_zone_;      // 区域环（空 = 无区域）
    std::vector<std::vector<cv::Point>> region_contours_;
    std::map<int, State> states_;
    // ---- 复活 ID 识别（缺陷 B 的结构性防线）----
    // track_id → 最近一次被计数器看到的 update 序号。用来判断"这个 ID 是不是真的第一次
    // 出现"，从而不让被 tracker 重连复活的旧轨迹冒充新生 ID 去继承并封存别人。
    // 与状态保留期无关；超过 kSeenIdHorizonUpdates 没再出现即回收（内存有界）。
    std::map<int, int> last_seen_seq_;
    int update_seq_ = 0;
    std::vector<std::string> notes_;   // 诊断注记（见 notes()）
};

#endif // GATEKEEPER_GATE_ID_COUNTER_H
