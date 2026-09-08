#ifndef GATEKEEPER_GATE_ID_COUNTER_H
#define GATEKEEPER_GATE_ID_COUNTER_H

#include <cstdint>
#include <map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "types.h"
#include "bytetrack/tracker.h"

// 摄像头角色：决定"进入类 / 离开类"两种底层事件如何映射成 IN / OUT / RETRO。
// 两个摄像头的判定逻辑完全对称，角色只负责给事件贴语义标签：
//   - 入口摄像头(201)：外→内 = IN，内→外消失 = RETRO
//   - 出口摄像头(200)：外→内 = RETRO，内→外消失 = OUT
enum GateRole {
    GATE_ROLE_ENTRY = 0,  // 入口摄像头(201)
    GATE_ROLE_EXIT  = 1,  // 出口摄像头(200)
};

enum GateEventType {
    GATE_EVENT_IN    = 0,  // 进入
    GATE_EVENT_OUT   = 1,  // 离开
    GATE_EVENT_RETRO = 2,  // 逆行
};

struct GateCountEvent {
    int track_id = -1;
    int type = GATE_EVENT_IN;
    cv::Point2f foot;
};

// 与 ByteTrack ID 绑定的双向闸机计数。
//
// 每个摄像头一条"标记线"(marker line)，把空间分成两侧：
//   - 入侧(检测区内) = signedDist > 0
//   - 出侧(检测区外) = signedDist < 0
//   - 入标记区域 = 标记线向入侧延伸 entry_depth px（默认 100）
//   - 出标记区域 = 标记线向出侧延伸 exit_depth px（默认 50）
//
// 每个 ID 至多产生一次"进入类"事件 + 一次"离开类"事件（1:1 绑定，根治重复计）：
//   - 进入类：脚底从出侧稳定跨越标记线进入入侧（外→内）
//       201 → IN，200 → RETRO
//   - 离开类：脚底曾在入侧稳定出现过，之后跨越到出侧，且该 ID 从活跃输出消失
//       201 → RETRO，200 → OUT
//     离开类与进入类共用同一检测质量门限(entry_quality_ok: conf≥kMinConfidence
//     && hit_streak≥kMinHitStreak)，使两类漏判概率一致、IN/OUT 对称。
//
// 由此自然覆盖"进入检测区→徘徊→逆行穿过标记线消失"的两次计数：
//   201 侧 = IN+1 且 RETRO+1；200 侧 = RETRO+1 且 OUT+1。
//
// 遮挡稳定性由 ByteTrack 保证：遮挡时用卡尔曼预测保持同一 ID 继续输出
// (observed=false 但 track_id 不变)，既不改变侧状态也不计 lost；只有该 ID 连续
// 多帧真正从活跃输出消失，"离开类"事件才结算。
class GateIdCounter {
public:
    GateIdCounter(GateRole role, const Tripwire& marker_line,
                  float entry_depth, float exit_depth);

    std::vector<GateCountEvent> update(const std::vector<TrackedObject>& tracks);

    // 脚底点所在"侧"：0=标记线附近 1=入侧 2=出侧（供显示用）
    int sideOf(const cv::Point2f& pt) const;

    void reset();

private:
    struct State {
        bool entered = false;    // 进入类事件已发射（每人至多一次）
        bool exited = false;     // 离开类事件已发射（每人至多一次）
        bool was_inside = false; // 脚底是否曾在入侧(检测区内)稳定出现过
        bool entry_quality_ok = false; // 进入侧跨越时检测质量是否达标(conf≥kMin & hit≥kMin)
                                       // 离开类用同一门限对称判定，使两类漏判概率一致
        int stable_side = 0;     // 已确认稳定侧：0/1/2
        int pending_side = 0;    // 候选侧（待确认）
        int pending_frames = 0;  // 候选侧连续帧数
        int lost_frames = 0;     // 连续未出现在活跃输出的帧数
        cv::Point2f last_foot{0.0f, 0.0f};
        int last_result_frame_id = -1;
    };

    // 带符号距离：>0 入侧(检测区内)，<0 出侧(检测区外)
    float signedDist(const cv::Point2f& pt) const;

    GateRole role_;
    Tripwire marker_line_;
    float entry_depth_;    // 入标记区域深度(>0)，配置契约保留（判定用符号距离）
    float exit_depth_;     // 出标记区域深度(>0)，配置契约保留（判定用符号距离）
    cv::Point2f inward_n_; // 入侧单位法向量
    std::map<int, State> states_;
};

#endif // GATEKEEPER_GATE_ID_COUNTER_H
