#ifndef GATE_ITEM_ABANDONMENT_H
#define GATE_ITEM_ABANDONMENT_H

// ============================================================================
// 物品滞留检测（纯 CV 背景差分 · 不识别物品类别）— 与 GateIdCounter 完全解耦
// ----------------------------------------------------------------------------
// 路线：双 MOG2 背景（长/短）→ static_mask = fg_long & ~fg_short（"静止下来的新物体"）
//       → 连通域 + 尺寸过滤 → 轻量质心关联跟踪 → 复用现有 YOLO people 框做人-物绑定
//       → 四状态机（MOVING / STATIC_CANDIDATE / LEVEL1_REMIND / LEVEL2_PUSH）。
//
// 设计要点：
//   1. 不识别物品：滞留物 = 「新出现并持续静止的前景块」，类别无关，零重训。
//   2. 复用 person 框：主人绑定直接用 main_tracking 已产出的 tracks（people）。
//   3. 不动计数逻辑：GateIdCounter / tripwire_counter 一行不改。
//   4. 不用 NPU：MOG2 是带状态的逐像素统计模型 + 分支控制流，无法表达成 RKNN 静态图。
//      靠「降帧抽样(sample_every) + 降工作分辨率(work_w/h)」把 CPU 压到 ≈4~5%。
//
// 详见 文档/物品滞留检测_模块设计.md（§3 数据结构 / §4 逐帧算法 / §5 状态机）
// ============================================================================

#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>

#include "bytetrack/tracker.h"   // TrackedObject

// ==================== 配置 ====================
struct AbandonConfig {
    bool enable = true;

    // —— 背景差分工作分辨率（越小越快，越小目标越糊）——
    int work_w = 960;
    int work_h = 540;
    int warmup_frames = 60;   // 前 N 个抽样帧只喂背景模型，不检测（避免冷启动全屏前景）
    int sample_every = 4;     // 抽样帧间隔：每 N 帧跑一次背景差分（省 CPU）
    bool use_own_thread = false;  // 预留：独立线程模式。当前实现为同步抽样调用
                                  // （sample_every 已把开销压到 ~4%，无需线程）。
    std::string bg_seed_path = "";  // 可选：标准空场景背景图路径，直接 seed 长期背景
                                    //       → 免 warmup（对应导师"两输入"里的输入①）

    // —— 双背景学习率 ——
    double bg_long_lr = 0.002;   // 慢：把"新静止物"长期留作前景
    double bg_short_lr = 0.05;   // 快：几秒内吸收静止物

    // —— 尺寸门限（8cm 换算见设计文档 §8；现场标定后调）——
    int min_item_px = 45;        // 最小边长(px，全分辨率口径)，任一方向 < 此值丢弃
    double min_item_area = 2000; // 最小面积(px²，全分辨率口径)，双保险
    double max_area_ratio = 0.03;// 面积/画面 上限，防整片误报

    // —— 块跟踪 ——
    float assoc_dist_px = 60.0f;  // 相邻采样帧质心关联距离(全分辨率)
    int item_lost_frames = 10;    // 连续丢失(采样帧数)→判"被拾取/移出"，清除告警
    float reset_dist_px = 10.0f;  // 采样帧间位移 > 此值 → 移动，重置计时
    float smooth_alpha = 0.5f;    // 质心/框指数平滑系数（1=不平滑）

    // —— 人-物绑定 ——
    float safe_radius_px = 120.0f;  // 人-物"看管"半径（物品质心 ↔ 人框中心）
    int bind_confirm_frames = 5;    // 持续靠近 N 个采样帧才确认归属
    float person_iou_reject = 0.30f;// 静止块与 person 框的覆盖比 > 此值 → 判为"站着的人"，丢弃
                                    // （取 max(IoU, 交面积/块面积)，blob ⊂ 大框时 IoU 会偏小）

    // —— 两级告警（秒）——
    double remind_sec = 15.0;   // 一级：提醒乘客（可配）
    double push_sec = 60.0;     // 二级：推送监控中心/保安（默认 60s）

    // —— 证据序列（滞留前的人-物关联图片序列）——
    double backtrack_sec = 10.0;   // 回溯窗口时长
    int seq_fps = 5;               // 序列采样帧率（证据帧写入间隔 = 1/seq_fps 秒）
    std::string save_dir = "./abandon_evidence";
};

// ==================== 状态 ====================
enum class ItemState {
    MOVING = 0,        // 刚出现 / 正在被移动，计时清零
    STATIC_CANDIDATE,  // 已静止，但有人看管或无告警：计时中
    LEVEL1_REMIND,     // 无人关联静止 ≥ remind_sec：提醒乘客
    LEVEL2_PUSH,       // 无人关联静止 ≥ push_sec ：推送监控中心
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
    REMIND = 0,  // 一级：提醒乘客
    PUSH,        // 二级：推送监控中心
    CLEAR,       // 清除（主人返回 / 被拾取 / 被移走）
    RESET,       // 移动重置（位移超阈值，计时清零）
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
    int           owner_id = -1;        // 关联的人（若有）
    cv::Rect      box;                  // 全分辨率框
    double        static_sec = 0;       // 触发时"无人关联静止"时长
    std::string   evidence_dir;         // CLEAR/PUSH 时附带的证据序列目录
    std::string   note;                 // 人类可读原因（主人返回 / 被拾取 / 主人离开…）
    int64_t       ts_ms = 0;
};

// ==================== 单个滞留物轨迹 ====================
struct ItemTrack {
    int       id = -1;
    cv::Point2f centroid{0.f, 0.f};  // 平滑后质心（**全分辨率**坐标）
    cv::Rect  box;                   // 平滑后外接框（**全分辨率**坐标）
    ItemState state = ItemState::MOVING;

    int64_t first_seen_ms = 0;
    int64_t last_seen_ms = 0;
    int64_t static_since_ms = 0;      // 转静止时刻
    int64_t unattended_since_ms = 0;  // "无人关联"起点（0 = 当前有人看管）
    int     missing_frames = 0;       // 连续未匹配（采样帧数）

    int  owner_id = -1;               // 绑定的人 track_id（-1=未绑定）
    int  bind_hits = 0;               // 持续靠近帧数（绑定确认用）
    bool owner_left = false;          // 主人是否已离开

    bool emitted_remind = false;      // 一级告警已发（防重复）
    bool emitted_push = false;        // 二级告警已发
    bool evidence_saved = false;      // 证据序列已落盘

    int64_t last_evidence_ms = 0;     // 上次记证据帧的时间（按 seq_fps 节流）

    // 证据帧环（小图：滞留前的人-物关联序列）
    std::deque<std::pair<int64_t, cv::Mat>> evidence;
};

// ==================== 检测器 ====================
class ItemAbandonmentDetector {
public:
    ItemAbandonmentDetector(const AbandonConfig& cfg, std::string cam_id);
    ~ItemAbandonmentDetector();

    ItemAbandonmentDetector(const ItemAbandonmentDetector&) = delete;
    ItemAbandonmentDetector& operator=(const ItemAbandonmentDetector&) = delete;

    // 每帧调用一次（主循环内）。内部按 sample_every 抽样，非抽样帧立即返回空。
    //   full_frame   : 全分辨率 BGR 帧
    //   person_tracks: 现有 YOLO people 跟踪结果（只用于人-物绑定 + 排除站着的人）
    //   region_mask  : 监控区域 ROI（全分辨率 uchar；空 = 不限制区域）
    //   capture_ms   : 采集时间戳（毫秒）
    //   result_frame_id : 结果帧号（去重：同一 rid 只处理一次）
    std::vector<ItemEvent> update(const cv::Mat& full_frame,
                                  const std::vector<TrackedObject>& person_tracks,
                                  const cv::Mat& region_mask,
                                  int64_t capture_ms,
                                  int result_frame_id);

    // —— 供显示 / 统计 ——
    const std::vector<ItemTrack>& items() const { return items_; }
    int alarmCount() const;                 // 处于告警态(L1/L2)的滞留物数
    int level1Count() const;                // L1 数量
    int level2Count() const;                // L2 数量
    bool warmupDone() const { return frame_count_ >= cfg_.warmup_frames; }

    void setConfig(const AbandonConfig& c) { cfg_ = c; }
    const AbandonConfig& config() const { return cfg_; }

private:
    // —— 内部步骤 ——
    void seedBackgroundFromImage(const std::string& path);
    void extractStaticBlobs(const std::vector<TrackedObject>& persons,
                            const cv::Mat& region_mask,
                            std::vector<cv::Rect>& out_blobs,
                            std::vector<cv::Point2f>& out_centroids,
                            std::vector<int>& out_areas);
    void associateAndUpdate(const std::vector<cv::Rect>& blobs,
                            const std::vector<cv::Point2f>& centroids,
                            const std::vector<int>& areas,
                            int64_t now_ms,
                            std::vector<ItemEvent>& ev);
    void bindOwners(const std::vector<TrackedObject>& persons,
                    const cv::Mat& full_frame,
                    int64_t now_ms,
                    std::vector<ItemEvent>& ev);
    void advanceState(int64_t now_ms, std::vector<ItemEvent>& ev);
    void pushEvidence(ItemTrack& it, const cv::Mat& full_frame, int64_t now_ms);
    void saveEvidenceSequence(ItemTrack& it, int64_t center_ms);
    bool overlapsPerson(const cv::Rect& box,
                        const std::vector<TrackedObject>& persons) const;
    static double dist(const cv::Point2f& a, const cv::Point2f& b);

    // —— 状态 ——
    AbandonConfig cfg_;
    std::string   cam_id_;

    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_long_;
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_short_;

    int   call_count_ = 0;      // update() 调用次数（含非抽样帧）
    int   frame_count_ = 0;     // 实际跑过差分的帧数（warmup 判定用）
    int   next_item_id_ = 1;
    int   last_result_frame_id_ = -1;
    int64_t seq_ = 0;           // 证据目录序号

    std::vector<ItemTrack> items_;

    // 复用缓冲，避免每帧分配
    cv::Mat gray_small_, fg_long_, fg_short_, mask_combined_;
    cv::Mat kernel_open_, kernel_close_;
    double  scale_x_ = 1.0, scale_y_ = 1.0;   // 全分辨率 → 工作分辨率
    int     full_w_ = 0, full_h_ = 0;         // 最近一帧的全分辨率尺寸
    bool    seed_tried_ = false;              // 是否已尝试过 bg_seed_path
    bool    bg_seeded_ = false;               // 背景是否已由标准图建立（可跳过 warmup）
};

// ==================== 配置加载 ====================
// 解析 key=value 文本；支持 "roleN.xxx=value" 按相机角色覆盖（N = role 值）。
// 文件不存在 → 全部用默认值（并打印一行提示），不视为错误。
AbandonConfig loadAbandonConfig(const std::string& path, int role);

#endif // GATE_ITEM_ABANDONMENT_H
