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
    GATE_REGION_ENTRY   = 1,  // 入口带（t1 向检测区内延伸 entry_zone_depth 的矩形带）
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
//      一次跨线"补发一次（settled=true），堵住"跨线瞬间丢帧"的漏计。
//
// 区域序列(0/1/2) 与 inside_region 标记只用于显示/日志，不再决定事件。
// ID 交接(handover) 只继承 line_side（所在侧）与区域上下文，**不再继承 entered/exited**：
// 只要侧不翻转就不产生事件，此后每一次真实跨线都会重新计数。
class GateIdCounter {
public:
    GateIdCounter(GateRole role, const Tripwire& entry_line,
                  float entry_zone_depth, const cv::Mat& detection_mask);

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

        // ---- 生命周期 / 检测质量 ----
        int lost_frames = 0;          // 连续未出现在活跃输出的帧数
        int last_result_frame_id = -1;
        int last_hit_streak = 0;
        float last_confidence = 0.0f;

        // ---- 脚底点（含遮挡抬升校正基准）----
        cv::Point2f last_foot{0.0f, 0.0f};
        bool has_ref_foot = false;
        cv::Point2f ref_foot{0.0f, 0.0f};
        bool has_last_inside_foot = false;
        cv::Point2f last_inside_foot{0.0f, 0.0f};
    };

    cv::Point2f inwardNormal(const Tripwire& tw, float depth) const;
    float signedDist(const cv::Point2f& pt, const Tripwire& tw,
                     const cv::Point2f& normal) const;
    std::vector<cv::Point2f> zonePolygon(const Tripwire& tw, float depth) const;
    bool pointInPolygon(const cv::Point2f& pt,
                        const std::vector<cv::Point2f>& polygon,
                        float tolerance) const;
    GateRole role_;
    Tripwire entry_line_;
    float entry_depth_;                        // 入口带深度(>0)；<1 视为"无入口带"
    float entry_depth_signed_;                  // 带符号深度：t1 法向指向检测区侧
    cv::Mat detection_mask_;
    std::vector<cv::Point2f> entry_zone_;      // 入口带矩形（空 = 无入口带）
    std::vector<std::vector<cv::Point>> region_contours_;
    std::map<int, State> states_;
};

#endif // GATEKEEPER_GATE_ID_COUNTER_H
