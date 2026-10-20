#ifndef GATE_ITEM_ABANDONMENT_H
#define GATE_ITEM_ABANDONMENT_H

// ============================================================================
// 物品滞留检测（清场双时相）— 与 GateIdCounter 完全解耦
// ----------------------------------------------------------------------------
// ★ 本版已整体重写（2026-10-20）。旧的双 MOG2 + 参考差分实现已删除，
//   完整保留在 commit c67cd6a（远程可直接取回）。
//
// 新路线：清场触发 → 三帧（长期基线 / 入场前 / 离场后）比对 → 变化块 →
//         几何配对（移动）→ 残留表跨会话确认 → 两级告警
//   状态机在 include/change_session.h，本文件只是"相机侧的外观"：
//   ROI 裁剪、人在不在、变化掩膜从哪来、残余块怎么变成告警事件。
//
// 变化掩膜的两个来源：
//   cv       : |before − after| 灰度阈值（骨架，零依赖，先用它跑通全链路）
//   rknn     : Fast BSUV-Net 2.0（9 通道），替换 buildChangeMask() 的实现即可
//
// 详见 文档/FastBSUV-Net2.0_清场双时相比对方案.md
// ============================================================================

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "bytetrack/tracker.h"   // TrackedObject
#include "change_session.h"

// ==================== 配置 ====================
struct AbandonConfig {
    bool enable = true;              // 总开关（0 = update() 立即返回，零开销）

    // —— 检测区裁剪（8cm 小目标的关键：只处理检测区，1:1 不降采样）——
    int use_roi_crop  = 1;
    int roi_margin_px = 24;
    int work_max_w    = 1280;        // 裁剪后宽度超此值才降采样（0 = 永不降采样）
    int work_w        = 960;         // 仅整帧模式用
    int work_h        = 540;

    // —— 标定辅助 ——
    double region_width_cm = 0;      // 检测区横向实际宽度(cm)，0 = 未知

    // —— 清场会话（时序）——
    double t_clear_sec      = 1.5;   // 连续无人多久算"确认清场"
    double t_settle_sec     = 2.0;   // 人走后等多久取 after 帧
    double max_occupied_sec = 300.0; // 有人超此时长 → 放弃本次会话

    // —— 人在不在 ——
    double motion_thresh = 6.0;      // ROI 内相邻帧平均灰度差超此值 → 判为有人（补 YOLO 漏检）

    // —— 变化掩膜（cv 后端）——
    double change_thresh = 25.0;     // |before − after| 灰度阈值
    int    morph_open_px  = 3;       // 开运算：去掉细碎噪点（8cm 目标别调大）
    int    morph_close_px = 5;       // 闭运算：补小目标断裂

    // —— 会话作废守卫 ——
    double max_mean_abs_diff = 18.0; // 全局平均绝对差超此值 → 作废（光照漂移）
    double max_changed_ratio = 0.25; // 变化像素占比超此值 → 作废

    // —— 尺寸门限（全分辨率口径）——
    int    min_item_px   = 45;       // 最小边长
    double min_item_area = 324;      // 最小面积：8cm≈18px 见方≈324px²，别沿用 2000
    double max_area_ratio = 0.03;    // 面积/画面 上限

    // —— 配对与确认 ——
    double pair_dist_px    = 120.0;
    double pair_area_ratio = 0.5;
    int    confirm_rounds  = 2;      // 连续 N 轮清场稳定 → 确认
    int    keep_missed_rounds = 2;

    // —— 两级告警（秒，从首次发现起算）——
    double remind_sec = 15.0;
    double push_sec   = 60.0;

    // —— 推理后端 ——
    int use_network = 0;             // 0 = CV 骨架（absdiff）；1 = RKNN 网络（待接入）
    std::string model_path = "";     // .rknn 路径（use_network=1 时用）

    // —— 兼容字段（旧配置里有，读进来但当前版本不实现）——
    std::string save_dir = "./abandon_evidence";
    double backtrack_sec = 10.0;
    int    seq_fps = 5;
    double sample_every = 1;         // 新架构下每帧都要过状态机，恒为 1
};

// ==================== 状态 ====================
enum class ItemState {
    MOVING = 0,        // 预留（新架构下不会产生）
    STATIC_CANDIDATE,  // 已发现，未跨会话确认
    LEVEL1_REMIND,     // 确认且无人关联 ≥ remind_sec
    LEVEL2_PUSH,       // 确认且无人关联 ≥ push_sec
};

inline const char* itemStateName(ItemState s) {
    switch (s) {
        case ItemState::MOVING:           return "MOVING";
        case ItemState::STATIC_CANDIDATE: return "STATIC";
        case ItemState::LEVEL1_REMIND:    return "L1_REMIND";
        case ItemState::LEVEL2_PUSH:      return "L2_PUSH";
    }
    return "?";
}

// ==================== 事件 ====================
enum class ItemEventType {
    REMIND = 0,
    PUSH,
    CLEAR,
    RESET,
};

inline const char* itemEventName(ItemEventType e) {
    switch (e) {
        case ItemEventType::REMIND: return "REMIND";
        case ItemEventType::PUSH:   return "PUSH";
        case ItemEventType::CLEAR:  return "CLEAR";
        case ItemEventType::RESET:  return "RESET";
    }
    return "?";
}

struct ItemEvent {
    int           item_id = -1;
    ItemEventType type = ItemEventType::CLEAR;
    int           owner_id = -1;      // 新架构不做归属绑定，恒为 -1
    cv::Rect      box;                // 全分辨率框
    double        static_sec = 0;     // 从首次发现至今的时长
    std::string   evidence_dir;       // 当前版本不产出证据序列，恒为空
    std::string   note;               // 人类可读原因
    int64_t       ts_ms = 0;
};

// ==================== 单个滞留物 ====================
// 字段与旧版保持一致，main_tracking 的绘制代码不用改。
struct ItemTrack {
    int       id = -1;
    cv::Point2f centroid{0.f, 0.f};   // 全分辨率
    cv::Rect  box;                    // 全分辨率
    ItemState state = ItemState::STATIC_CANDIDATE;

    int64_t first_seen_ms = 0;
    int64_t last_seen_ms = 0;
    int64_t static_since_ms = 0;
    int64_t unattended_since_ms = 0;  // = first_seen_ms（新架构下无人看管从发现即开始计）
    int     missing_frames = 0;

    int  owner_id = -1;
    int  bind_hits = 0;
    bool owner_left = true;

    bool emitted_remind = false;
    bool emitted_push = false;
    bool evidence_saved = false;
    int64_t last_evidence_ms = 0;

    int rounds = 0;                   // 跨会话确认计数（诊断用）
    bool confirmed = false;           // 已跨会话确认（≥ confirm_rounds）
};

// ==================== 检测器 ====================
class ItemAbandonmentDetector {
public:
    ItemAbandonmentDetector(const AbandonConfig& cfg, std::string cam_id);
    ~ItemAbandonmentDetector();

    ItemAbandonmentDetector(const ItemAbandonmentDetector&) = delete;
    ItemAbandonmentDetector& operator=(const ItemAbandonmentDetector&) = delete;

    // 每帧调用一次（主循环内，必须在 draw() 之前 —— 画上框会污染差分）
    std::vector<ItemEvent> update(const cv::Mat& full_frame,
                                  const std::vector<TrackedObject>& person_tracks,
                                  const cv::Mat& region_mask,
                                  int64_t capture_ms,
                                  int result_frame_id);

    // 注入初始基线（全分辨率 BGR）：省掉开机后等第一次清场的时间。
    // 主要给测试和离线工具用；正常路径靠自动学习。
    bool setRefBackground(const cv::Mat& full_bgr);

    // —— 供显示 / 统计 ——
    const std::vector<ItemTrack>& items() const { return items_; }
    int alarmCount() const;      // L1 + L2
    int level1Count() const;
    int level2Count() const;
    bool warmupDone() const { return session_.hasBaseline(); }

    void setConfig(const AbandonConfig& c);
    const AbandonConfig& config() const { return cfg_; }

    // 诊断：本路相机跑过多少次会话
    int pairCount() const { return session_.pairCount(); }

private:
    void prepareRoi(const cv::Mat& region_mask);
    cv::Rect    toFull(const cv::Rect& r) const;
    cv::Point2f toFull(const cv::Point2f& p) const;

    // 检测区内有没有人：YOLO 框中心落在区内 + ROI 内帧间运动能量
    bool personPresent(const std::vector<TrackedObject>& persons, const cv::Mat& roi_bgr);
    // 变化掩膜：cv = absdiff 阈值；rknn = 网络（待接入）
    bool buildChangeMask(const PairRequest& req, cv::Mat& mask, std::string& backend_note);

    void printCalibration() const;
    void syncTracks(const PairResult& res, int64_t now_ms, std::vector<ItemEvent>& ev);
    void advanceState(int64_t now_ms, std::vector<ItemEvent>& ev);

    AbandonConfig cfg_;
    std::string   cam_id_;
    ChangeSession session_;

    bool    roi_ready_ = false;
    cv::Rect roi_rect_{0, 0, 0, 0};
    double  inv_sx_ = 1.0, inv_sy_ = 1.0;
    int     full_w_ = 0, full_h_ = 0;
    int     last_rid_ = -1;
    bool    calib_printed_ = false;

    cv::Mat roi_bgr_;        // 当前帧 ROI（彩色）
    cv::Mat prev_gray_;      // 上一帧 ROI 灰度（算运动能量）
    cv::Mat gray_before_, gray_after_, diff_buf_, mask_buf_;

    std::vector<ItemTrack> items_;
};

// ==================== 配置加载 ====================
AbandonConfig loadAbandonConfig(const std::string& path, int role);

#endif // GATE_ITEM_ABANDONMENT_H
