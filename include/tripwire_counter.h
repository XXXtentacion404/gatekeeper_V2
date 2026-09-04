#ifndef GATEKEEPER_TRIPWIRE_COUNTER_H
#define GATEKEEPER_TRIPWIRE_COUNTER_H

#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "types.h"
#include "bytetrack/tracker.h"

enum TripwireEventType {
    TW_EVENT_IN = 0,
    TW_EVENT_OUT = 1,
    TW_EVENT_RETRO = 2,
};

struct TripwireCountEvent {
    int track_id = -1;
    int type = TW_EVENT_IN;
    int side = 0;  // 1=entry, 2=exit
    cv::Point2f foot;
    bool birth_confirmed = false;
    float confidence = 0.0f;
    float movement = 0.0f;
    bool loss_confirmed = false;
};

struct TripwireDebugEvent {
    int track_id = -1;
    std::string reason;
    int raw_region = -1;
    int stable_region = -1;
    int pending_region = -1;
    int pending_frames = 0;
    int approach_side = 0;
    int approach_observations = 0;
    int departure_side = 0;
    int departure_observations = 0;
    bool inside_committed = false;
    int observed_streak = 0;
    int hit_streak = 0;
    int lost_frames = 0;
    float confidence = 0.0f;
    float movement = 0.0f;
    cv::Point2f foot;
};

// Stable region-sequence counter:
// 0->1->2 IN, 0->3->2 RETRO, 2->3->0 OUT, 2->1->0 RETRO.
class TripwireCounter {
public:
    TripwireCounter(const Tripwire& entry_line,
                    const Tripwire& exit_line,
                    float entry_depth,
                    float exit_depth,
                    const cv::Mat& detection_mask);

    std::vector<TripwireCountEvent> update(const std::vector<TrackedObject>& tracks);
    std::vector<TripwireDebugEvent> takeDebugEvents();
    void setDebugEnabled(bool enabled) { debug_enabled_ = enabled; }
    int regionOf(const cv::Point2f& pt) const;
    bool inZone(const cv::Point2f& pt) const { return regionOf(pt) == 2; }

private:
    struct State {
        std::deque<cv::Point2f> trail;
        cv::Point2f last_foot;
        // ---- 遮挡脚底基准（per-ID）----
        // 门口遮挡时检测框底部会从脚底跳到胸口，遮挡结束又落回。ByteTrack
        // 在遮挡期间保持同一 ID，用该 ID 最近一次可靠脚底观测作为地面基准，
        // 即可识别"被抬起"的框并把脚底冻结回地面。
        bool has_ref_foot = false;        // 是否已建立可靠的脚底（地面）基准
        cv::Point2f ref_foot{0.0f, 0.0f}; // 最近一次可靠脚底观测
        int lost_frames = 0;
        int last_result_frame_id = -1;
        int observed_streak = 0;
        int last_hit_streak = 0;
        float last_confidence = 0.0f;
        int64_t last_event_ms = 0;   // 上次计数事件时间，用于事件冷却（抑制抖动连发）

        // ---- ByteTrack ID 与计数的 1:1 绑定 ----
        // 每个 ID 生命周期内最多 1 次进入类事件(IN/RETRO-in) + 1 次离开类事件
        // (OUT/RETRO-out)。ID 切换(遮挡致轨迹死亡+新生)时由 ID 交接继承计数
        // 上下文，保证一个物理的人在整个系统中只贡献一对 IN/OUT。
        int enter_events = 0;        // 已发进入类事件数（0/1）
        int exit_events = 0;         // 已发离开类事件数（0/1）

        int stable_region = 0;
        int pending_region = -1;
        int pending_frames = 0;
        cv::Point2f pending_start_foot;

        int approach_side = 0;
        int approach_observed_frames = 0;
        int approach_gap_frames = 0;
        cv::Point2f approach_foot;
        bool inside_committed = false;
        bool has_inside_foot = false;
        cv::Point2f last_inside_foot;
        int departure_side = 0;
        int departure_observed_frames = 0;
        cv::Point2f departure_foot;
        cv::Point2f departure_side_foot;

        bool baseline_ready = false;
        bool candidate_active = false;
        int candidate_side = 0;
        int candidate_frames = 0;
        int candidate_inward_steps = 0;
        float candidate_last_depth = 0.0f;
        float candidate_conf_sum = 0.0f;
        cv::Point2f candidate_birth_foot;
    };

    cv::Point2f inwardNormal(const Tripwire& tw, float depth) const;
    float signedDist(const cv::Point2f& pt, const Tripwire& tw,
                     const cv::Point2f& normal) const;
    std::vector<cv::Point2f> zonePolygon(const Tripwire& tw, float depth) const;
    bool pointInPolygon(const cv::Point2f& pt,
                        const std::vector<cv::Point2f>& polygon,
                        float tolerance) const;
    bool pointInMask(const cv::Point2f& pt, float tolerance) const;

    Tripwire entry_line_;
    Tripwire exit_line_;
    float entry_depth_;
    float exit_depth_;
    cv::Mat detection_mask_;
    std::vector<cv::Point2f> entry_zone_;
    std::vector<cv::Point2f> exit_zone_;
    std::vector<std::vector<cv::Point>> detection_contours_;
    std::map<int, State> states_;
    bool debug_enabled_ = true;
    std::vector<TripwireDebugEvent> debug_events_;
};

#endif
