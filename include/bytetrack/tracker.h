#ifndef GATE_BYTETRACK_TRACKER_H
#define GATE_BYTETRACK_TRACKER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include "types.h"

struct TrackedObject {
    int track_id;
    cv::Rect box;
    float confidence;
    int class_id;
    std::string class_name;
    cv::Scalar color;
    bool observed = false;       // true only when this frame matched a real detection
    int hit_streak = 0;
    int time_since_update = 0;
    int result_frame_id = -1;
    // Camera acquisition time propagated through asynchronous inference.
    int64_t capture_timestamp_ms = 0;
};

class ByteTrackTracker {
public:
    /**
     * @param track_thresh  低阈值：低于此的检测被丢弃 (default 0.3)
     * @param high_thresh   高阈值：高于此的检测用于创建新track (default 0.45)
     * @param match_thresh  IoU匹配阈值 (default 0.7)
     * @param max_lost      最大丢失帧数后移除track (default 30)
     * @param min_hits      track至少连续命中多少帧才输出 (default 3)
     */
    ByteTrackTracker(float track_thresh = 0.3f,
                     float high_thresh = 0.45f,
                     float match_thresh = 0.35f,
                     int max_lost = 60,
                     int min_hits = 2);
    ~ByteTrackTracker() = default;

    ByteTrackTracker(const ByteTrackTracker&) = delete;
    ByteTrackTracker& operator=(const ByteTrackTracker&) = delete;

    /** 核心更新函数 */
    std::vector<TrackedObject> update(const std::vector<Detection>& detections,
                                     const cv::Mat& frame,
                                     int img_width, int img_height);

    /** 绘制跟踪结果 */
    void drawTracks(cv::Mat& img, const std::vector<TrackedObject>& tracks);

    /** 当前活跃跟踪数 */
    int getTrackCount() const { return (int)active_tracks_.size(); }

private:
    // ========== Kalman 4-状态滤波器 ==========
    struct KalmanFilter {
        cv::KalmanFilter kf;
        
        KalmanFilter() : kf(8, 4, 0, CV_32F) {
            // 状态: [cx, cy, w, h, vx, vy, vw, vh]
            // 测量: [cx, cy, w, h]
            
            // F = [I, I*dt; 0, I]  (dt=1)
            cv::setIdentity(kf.transitionMatrix, cv::Scalar(1));
            for (int i = 0; i < 4; i++)
                kf.transitionMatrix.at<float>(i, i+4) = 1.0f;
            
            // H = [I, 0]
            kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
            for (int i = 0; i < 4; i++)
                kf.measurementMatrix.at<float>(i, i) = 1.0f;
            
            cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(0.1));
            cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(0.5));
            cv::setIdentity(kf.errorCovPost, cv::Scalar::all(2));
        }
        
        void init(const cv::Rect& r) {
            kf.statePost.at<float>(0) = (float)(r.x + r.width/2.0);
            kf.statePost.at<float>(1) = (float)(r.y + r.height/2.0);
            kf.statePost.at<float>(2) = (float)r.width;
            kf.statePost.at<float>(3) = (float)r.height;
            for (int i = 4; i < 8; i++) kf.statePost.at<float>(i) = 0;
        }
        
        cv::Rect predict() {
            cv::Mat s = kf.predict();
            float cx = s.at<float>(0), cy = s.at<float>(1);
            float w = s.at<float>(2), h = s.at<float>(3);
            return cv::Rect((int)(cx - w/2), (int)(cy - h/2), std::max(1, (int)w), std::max(1, (int)h));
        }
        
        void update(const cv::Rect& r) {
            cv::Mat meas(4, 1, CV_32F);
            meas.at<float>(0) = (float)(r.x + r.width/2.0);
            meas.at<float>(1) = (float)(r.y + r.height/2.0);
            meas.at<float>(2) = (float)r.width;
            meas.at<float>(3) = (float)r.height;
            kf.correct(meas);
        }

        /** Mahalanobis距离：检测框 vs Kalman预测位置 */
        float mahalanobisDistance(const cv::Rect& det) const {
            cv::Mat z(4, 1, CV_32F);
            z.at<float>(0) = det.x + det.width/2.0f;
            z.at<float>(1) = det.y + det.height/2.0f;
            z.at<float>(2) = (float)det.width;
            z.at<float>(3) = (float)det.height;
            cv::Mat hx = kf.measurementMatrix * kf.statePre;
            cv::Mat S  = kf.measurementMatrix * kf.errorCovPre *
                         kf.measurementMatrix.t() + kf.measurementNoiseCov;
            cv::Mat S_inv, v = z - hx;
            cv::invert(S, S_inv, cv::DECOMP_SVD);
            cv::Mat r = v.t() * S_inv * v;
            return std::sqrt(std::max(0.0f, r.at<float>(0,0)));
        }
    };
    
    // ========== Track ==========
    struct Track {
        int id;                        // 唯一ID
        KalmanFilter kf;               // Kalman滤波器
        cv::Rect rect;                 // 当前预测/更新后的框
        float score;                   // 置信度
        int time_since_update;         // 未更新帧数
        int hit_streak;                // 连续命中数
        bool is_activated;             // 是否已激活(被confirmed)
        cv::Mat appearance_hist;       // HSV颜色直方图（EMA更新）
        float avg_score = 0;           // 检测置信度滑动平均（稳定度指标）
        int stable_count = 0;          // 连续匹配帧数
        
        Track() : id(-1), score(0), time_since_update(0), hit_streak(0), is_activated(false) {}
        
        void activate(int new_id, const Detection& det) {
            id = new_id;
            rect = det.box;
            score = det.confidence;
            kf.init(det.box);
            time_since_update = 0;
            hit_streak = 1;
            is_activated = false;
        }
        
        void re_activate(const Detection& det) {
            rect = det.box;
            score = det.confidence;
            kf.update(det.box);
            time_since_update = 0;
            hit_streak++;
            is_activated = true;
        }
        
        void predict() {
            rect = kf.predict();
            time_since_update++;
        }
        
        void update(const Detection& det) {
            rect = det.box;
            score = det.confidence;
            kf.update(det.box);
            time_since_update = 0;
            hit_streak++;
            if (hit_streak >= 3) is_activated = true;
        }
    };
    
    // 参数
    float track_thresh_;
    float high_thresh_;
    float match_thresh_;
    int max_lost_;
    int min_hits_;
    int next_id_;
    
    // 存储 (使用list避免迭代器失效)
    std::vector<Track> active_tracks_;
    std::vector<Track> removed_tracks_;  // 只是保留颜色映射用
    
    std::unordered_map<int, cv::Scalar> color_map_;
    cv::RNG rng_;
    
    static float calcIou(const cv::Rect& a, const cv::Rect& b);
    
    /**
     * 匈牙利风格贪心匹配：按IoU降序匹配track和detection
     * @return matches: (track_index, det_index)
     */
    void greedyMatch(const std::vector<Track>& tracks,
                     const std::vector<Detection>& dets,
                     float iou_thresh,
                     std::vector<int>& track_match,   // [track_idx]->det_idx 或 -1
                     std::vector<int>& det_match);    // [det_idx]->track_idx 或 -1

    // ========== 联合评分模型 (NEW) ==========
    /** 计算检测框的HSV外观直方图 */
    cv::Mat computeAppearanceHistogram(const cv::Mat& frame, const cv::Rect& box);

    /** 计算track与detection的联合评分 S = α*IoU + β*Motion + γ*Appearance */
    float calcJointScore(const Track& track, const Detection& det,
                         const cv::Mat& frame, const cv::Mat& det_hist);

    /** 轨迹引导置信度校准：为稳定track的低分匹配detection提升置信度 */
    void calibrateConfidence(const std::vector<Track>& tracks,
                              std::vector<Detection>& det_low,
                              std::vector<Detection>& det_high);

    /** 联合评分贪心匹配 */
    void greedyJointMatch(const std::vector<Track>& tracks,
                          const std::vector<Detection>& dets,
                          float score_thresh,
                          const cv::Mat& frame,
                          const std::vector<cv::Mat>& det_hists,
                          std::vector<int>& track_match,
                          std::vector<int>& det_match);
    
    cv::Scalar getColor(int track_id);
};

#endif // GATE_BYTETRACK_TRACKER_H
