// ============================================================================
// 物品滞留检测实现（纯 CV 背景差分 · 不识别物品类别）
// 详细设计与参数含义见 include/item_abandonment.h 与 文档/物品滞留检测_模块设计.md
//
// 三种检测模式（cfg_.mode）：
//   DOUBLE_MOG2  : fg_long & ~fg_short                —— 抗光照，小目标弱，有吸收窗口上限
//   REF_BG_DIFF  : |当前 - 空场景标准图| > 阈值        —— 导师"两输入"方案，小目标最敏感
//   HYBRID(默认) : ref_mask & ~fg_short               —— 参考差分给灵敏度 + 短背景给"已静止"语义
//
// 检测区处理（cfg_.use_roi_crop）：只把检测区多边形的 bbox 送进差分并尽量保持 1:1 分辨率
// （而不是把整帧缩到 960x540）—— 这是 8cm 小目标能不能做到的关键，能把目标像素数翻倍。
// ============================================================================

#include "item_abandonment.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

// 建目录：POSIX mkdir 带 mode，Windows(MSVC/MinGW) 只有单参数版本。
// 不用 <filesystem> 是为了避免板端老 g++ 需要额外链 libstdc++fs。
#ifdef _WIN32
    #include <direct.h>
    #define GATE_MKDIR(p) ::_mkdir(p)
#else
    #include <sys/stat.h>
    #include <sys/types.h>
    #define GATE_MKDIR(p) ::mkdir((p), 0755)
#endif

// ==================== 小工具 ====================
static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }
static inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

static inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// 逐级创建目录（已存在不算错）
static void mkdirP(const std::string& path) {
    if (path.empty()) return;
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || path[i] == '\\' || i + 1 == path.size()) {
            if (cur == "/" || cur == "./" || cur == "\\" || cur == "../") continue;
            GATE_MKDIR(cur.c_str());
        }
    }
}

// 形态学核边长强制为奇数（0 表示不用该操作）
static int oddPx(int v) { return v <= 0 ? 0 : (v | 1); }

// ==================== 构造 / 析构 ====================
ItemAbandonmentDetector::ItemAbandonmentDetector(const AbandonConfig& cfg, std::string cam_id)
    : cfg_(cfg), cam_id_(std::move(cam_id)) {
    // history 只在 learningRate < 0（自动）时决定更新速度；这里一律显式给 lr（见 cfg_），
    // history 主要用于高斯分量的淘汰与新分量生成阈值。
    // detectShadows=false：不把阴影标为 127 灰度，掩膜只剩 0/255，取前景无需阈值化，更快。
    bg_long_  = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
    bg_short_ = cv::createBackgroundSubtractorMOG2(60,  16.0, false);

    // 形态学核：小目标必须用小核（5x5 开运算会把 8cm 细结构直接吃掉）
    const int op = oddPx(cfg_.morph_open_px);
    const int cl = oddPx(cfg_.morph_close_px);
    if (op > 0) kernel_open_  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(op, op));
    if (cl > 0) kernel_close_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cl, cl));

    static const char* kModeName[] = {"DOUBLE_MOG2", "REF_BG_DIFF", "HYBRID"};
    const int mi = clampi(cfg_.mode, 0, 2);
    std::cout << "[ITEM] " << cam_id_ << " 滞留检测器已创建: enable=" << (cfg_.enable ? 1 : 0)
              << " mode=" << kModeName[mi]
              << " ROI裁剪=" << (cfg_.use_roi_crop ? 1 : 0)
              << " sample_every=" << cfg_.sample_every
              << " min_item_px=" << cfg_.min_item_px
              << " morph=" << op << "/" << cl
              << " remind=" << cfg_.remind_sec << "s push=" << cfg_.push_sec << "s"
              << (cfg_.ref_bg_path.empty() ? "" : (" ref_bg=" + cfg_.ref_bg_path))
              << std::endl;
    if (mi != (int)AbandonMode::DOUBLE_MOG2 && cfg_.ref_bg_path.empty()) {
        std::cerr << "[ITEM] " << cam_id_ << " ⚠️ mode=" << kModeName[mi]
                  << " 但 ref_bg_path 为空 → 参考基线不可用，本相机将退回 DOUBLE_MOG2。"
                     "请在 abandon_config.txt 填标准空场景背景图路径"
                     "（或用 setRefBackground() 注入）。\n";
    }
    if (cfg_.push_sec > 120.0 && cfg_.mode == (int)AbandonMode::DOUBLE_MOG2)
        std::cout << "[ITEM] " << cam_id_
                  << " 注意: DOUBLE_MOG2 下 push_sec=" << cfg_.push_sec
                  << "s 较长。有效检测窗口上限 ≈ 1/bg_long_lr 个抽样帧"
                     "（默认 0.002 × sample_every=4 @20FPS ≈ 100s）——超过后长背景也会把"
                     "静止物吸收入背景，滞留物从 static_mask 消失。"
                     "需要更长门槛请调小 bg_long_lr，或改用 REF_BG_DIFF/HYBRID 模式（无上限）。"
                  << std::endl;
}

ItemAbandonmentDetector::~ItemAbandonmentDetector() = default;

// ==================== 检测区 ROI 裁剪 ====================
void ItemAbandonmentDetector::prepareRoi(const cv::Mat& region_mask) {
    roi_ready_ = true;
    if (!cfg_.use_roi_crop) {
        roi_rect_ = cv::Rect(0, 0, full_w_, full_h_);
        std::cout << "[ITEM] " << cam_id_ << " 检测区裁剪=关闭 → 处理整帧 "
                  << full_w_ << "x" << full_h_ << std::endl;
        return;
    }
    if (region_mask.empty()) {
        roi_rect_ = cv::Rect(0, 0, full_w_, full_h_);
        std::cout << "[ITEM] " << cam_id_ << " 无检测区掩膜 → 处理整帧" << std::endl;
        return;
    }
    std::vector<cv::Point> nz;
    cv::findNonZero(region_mask, nz);
    if (nz.empty()) {
        roi_rect_ = cv::Rect(0, 0, full_w_, full_h_);
        return;
    }
    cv::Rect bb = cv::boundingRect(nz);
    const int m = std::max(0, cfg_.roi_margin_px);
    bb.x -= m; bb.y -= m; bb.width += 2 * m; bb.height += 2 * m;
    bb &= cv::Rect(0, 0, full_w_, full_h_);
    if (bb.width <= 0 || bb.height <= 0) bb = cv::Rect(0, 0, full_w_, full_h_);
    roi_rect_ = bb;
    std::cout << "[ITEM] " << cam_id_ << " 检测区 ROI 裁剪: " << bb.width << "x" << bb.height
              << " @(" << bb.x << "," << bb.y << ")  占整帧 "
              << (100.0 * bb.area() / ((double)full_w_ * full_h_)) << "%"
              << " → 小目标可用像素数不变（不降采样）" << std::endl;
}

cv::Rect ItemAbandonmentDetector::toFull(const cv::Rect& r) const {
    return cv::Rect(roi_rect_.x + (int)std::lround(r.x * inv_sx_),
                    roi_rect_.y + (int)std::lround(r.y * inv_sy_),
                    (int)std::lround(r.width * inv_sx_),
                    (int)std::lround(r.height * inv_sy_));
}
cv::Point2f ItemAbandonmentDetector::toFull(const cv::Point2f& p) const {
    return cv::Point2f((float)(roi_rect_.x + p.x * inv_sx_),
                       (float)(roi_rect_.y + p.y * inv_sy_));
}

// ==================== 标准背景图（导师"两输入"的输入①）====================
// 两条入口共用：seedFromRefImage()（读文件）与 setRefBackground()（外部注入）。
bool ItemAbandonmentDetector::applyRefBackground(const cv::Mat& full_bgr) {
    if (full_bgr.empty() || full_bgr.cols != full_w_ || full_bgr.rows != full_h_) return false;

    // 与逐帧处理走完全相同的 ROI 裁剪 + 分辨率，保证像素级可比
    cv::Mat roi_norm = full_bgr;
    if (cfg_.use_roi_crop) {
        cv::Rect r = roi_rect_ & cv::Rect(0, 0, full_bgr.cols, full_bgr.rows);
        if (r.width > 0 && r.height > 0) roi_norm = full_bgr(r);
    }
    cv::Mat small = roi_norm, gray;
    if (!gray_small_.empty() &&
        (roi_norm.cols != gray_small_.cols || roi_norm.rows != gray_small_.rows)) {
        cv::resize(roi_norm, small, gray_small_.size(), 0, 0, cv::INTER_AREA);
    }
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);

    ref_bg_gray_ = gray.clone();     // 参考差分基线
    ref_bg_ready_ = true;

    // 同时把 MOG2 直接设成这张背景 → 免 warmup，开机即检测。
    // ⚠️ MOG2::apply 只有 (image, fgmask, learningRate) 一个重载，learningRate 不能省掉
    //    fgmask 参数（写 apply(gray, 1.0) 会把 1.0 当 OutputArray 而编译失败）。
    cv::Mat dummy;
    for (int i = 0; i < 3; ++i) {
        bg_long_->apply(ref_bg_gray_, dummy, 1.0);
        bg_short_->apply(ref_bg_gray_, dummy, 1.0);
    }
    bg_seeded_ = true;
    return true;
}

bool ItemAbandonmentDetector::setRefBackground(const cv::Mat& full_bgr) {
    if (full_bgr.empty()) return false;
    if (full_w_ <= 0) {                 // 还没跑过 update() → 先建立尺寸与 ROI（整帧）
        full_w_ = full_bgr.cols;
        full_h_ = full_bgr.rows;
        if (!roi_ready_) prepareRoi(cv::Mat());
    }
    if (!applyRefBackground(full_bgr)) return false;
    std::cout << "[ITEM] " << cam_id_ << " 已注入标准背景图 → 参考差分基线就绪("
              << ref_bg_gray_.cols << "x" << ref_bg_gray_.rows << ")，跳过 warmup" << std::endl;
    return true;
}

void ItemAbandonmentDetector::seedFromRefImage() {
    if (cfg_.ref_bg_path.empty()) return;   // 走 MOG2 warmup 冷启动

    cv::Mat img = cv::imread(cfg_.ref_bg_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ITEM] " << cam_id_ << " 背景标准图读取失败: " << cfg_.ref_bg_path
                  << " → 退回 DOUBLE_MOG2 冷启动（warmup_frames=" << cfg_.warmup_frames << "）"
                  << std::endl;
        return;
    }
    if (img.cols != full_w_ || img.rows != full_h_) {
        std::cout << "[ITEM] " << cam_id_ << " 标准图尺寸 " << img.cols << "x" << img.rows
                  << " 与视频 " << full_w_ << "x" << full_h_ << " 不一致 → 已缩放对齐" << std::endl;
        cv::Mat tmp;
        cv::resize(img, tmp, cv::Size(full_w_, full_h_), 0, 0, cv::INTER_AREA);
        img = tmp;
    }
    if (!applyRefBackground(img)) {
        std::cerr << "[ITEM] " << cam_id_ << " 标准背景图应用失败" << std::endl;
        return;
    }
    std::cout << "[ITEM] " << cam_id_ << " 已载入标准背景图 → 参考差分基线就绪("
              << ref_bg_gray_.cols << "x" << ref_bg_gray_.rows << ")，跳过 warmup" << std::endl;
}

// 光照/场景突变时刷新参考基线
void ItemAbandonmentDetector::refreshRefBg(const cv::Mat& gray) {
    ref_bg_gray_ = gray.clone();
    ref_bg_ready_ = true;
    std::cerr << "[ITEM] " << cam_id_ << " ⚠️ 连续 " << cfg_.ref_light_change_frames
              << " 帧大面积变化（光照/相机变动）→ 已用当前帧刷新参考背景。"
                 "注意：若此刻画面里确有滞留物，它会被写进基线、之后不再告警。\n";
}

// ==================== 按模式生成 static mask ====================
void ItemAbandonmentDetector::buildStaticMask() {
    const int mode = clampi(cfg_.mode, 0, 2);

    // MOG2 始终喂：DOUBLE_MOG2 靠它取前景；REF/HYBRID 靠它的短背景提供"已静止"语义
    bg_long_->apply(gray_small_, fg_long_, cfg_.bg_long_lr);
    bg_short_->apply(gray_small_, fg_short_, cfg_.bg_short_lr);

    if (mode == (int)AbandonMode::DOUBLE_MOG2) {
        cv::Mat not_short;
        cv::bitwise_not(fg_short_, not_short);
        cv::bitwise_and(fg_long_, not_short, mask_combined_);
    } else {
        if (!ref_bg_ready_ || ref_bg_gray_.size() != gray_small_.size()) {
            mask_combined_.create(gray_small_.size(), CV_8UC1);
            mask_combined_.setTo(0);
            return;
        }
        // 参考背景差分：|当前 − 空场景标准图| > 固定阈值（阈值可压得很低 → 抓得住 8cm）
        cv::absdiff(gray_small_, ref_bg_gray_, diff_buf_);
        cv::threshold(diff_buf_, ref_mask_, cfg_.ref_diff_thresh, 255, cv::THRESH_BINARY);
        if (mode == (int)AbandonMode::REF_BG_DIFF) {
            ref_mask_.copyTo(mask_combined_);
        } else {
            // HYBRID：参考差分给灵敏度；~fg_short 给"静止下来"语义（压掉走动的人/还在动的物体）
            cv::Mat not_short;
            cv::bitwise_not(fg_short_, not_short);
            cv::bitwise_and(ref_mask_, not_short, mask_combined_);
        }
    }

    if (!kernel_open_.empty())
        cv::morphologyEx(mask_combined_, mask_combined_, cv::MORPH_OPEN, kernel_open_);
    if (!kernel_close_.empty())
        cv::morphologyEx(mask_combined_, mask_combined_, cv::MORPH_CLOSE, kernel_close_);
}

// ==================== 标定自检 ====================
void ItemAbandonmentDetector::printCalibration() const {
    if (roi_rect_.width <= 0 || gray_small_.empty()) return;
    const double src_w = cfg_.use_roi_crop ? (double)roi_rect_.width : (double)full_w_;
    const double zoom = gray_small_.cols / std::max(1.0, src_w);
    std::cout << "[ITEM] " << cam_id_ << " 标定自检: ROI " << roi_rect_.width << "x" << roi_rect_.height
              << "px → 处理图 " << gray_small_.cols << "x" << gray_small_.rows
              << "px (缩放 " << zoom << "x)" << std::endl;

    if (cfg_.region_width_cm <= 0) {
        std::cout << "[ITEM]   未填 region_width_cm → 无法自动校核 8cm。现场做法：在检测区放"
                     "已知尺寸物体（A4 纸横放 29.7cm），量它在**处理图**里的像素宽，"
                     "px_per_cm = 像素宽/29.7，则 8cm ≈ 8×px_per_cm。\n";
        return;
    }
    const double px_per_cm = gray_small_.cols / cfg_.region_width_cm;
    const double px8 = 8.0 * px_per_cm;
    const double min_cm = cfg_.min_item_px / px_per_cm;
    std::cout << "[ITEM]   检测区宽 " << cfg_.region_width_cm << "cm → px_per_cm=" << px_per_cm
              << " → 8cm ≈ " << px8 << "px;  min_item_px=" << cfg_.min_item_px
              << " 相当于 " << min_cm << "cm";
    if (cfg_.min_item_px > px8) {
        std::cout << "  ⚠️ 比 8cm 大 → 8cm 目标会被尺寸门限滤掉！建议 min_item_px ≤ "
                  << (int)px8;
    } else {
        std::cout << "  ✓ 满足 8cm 指标";
    }
    std::cout << std::endl;
    if (px8 < 15.0) {
        std::cout << "[ITEM]   ⚠️ 8cm 仅 " << px8
                  << "px (<15px)：纯 CV 可靠检测困难。可提高 work_max_w（减少降采样）、"
                     "缩小检测区视野/把相机装近，或换更高分辨率相机。\n";
    } else if (px8 < 20.0) {
        std::cout << "[ITEM]   ⚠️ 8cm 约 " << px8
                  << "px (15~20px)：勉强可检测，建议 morph_open_px=0 并适当降低 "
                     "ref_diff_thresh（如 15~20）。\n";
    }
}

// ==================== 主入口 ====================
std::vector<ItemEvent> ItemAbandonmentDetector::update(const cv::Mat& full_frame,
                                                       const std::vector<TrackedObject>& person_tracks,
                                                       const cv::Mat& region_mask,
                                                       int64_t capture_ms,
                                                       int result_frame_id) {
    std::vector<ItemEvent> ev;
    if (!cfg_.enable) return ev;
    if (full_frame.empty() || full_frame.type() != CV_8UC3) return ev;

    // 同一结果帧只处理一次（主循环可能因超时重入）
    if (result_frame_id >= 0) {
        if (result_frame_id == last_result_frame_id_) return ev;
        last_result_frame_id_ = result_frame_id;
    }

    // ---- 抽样：每 sample_every 次调用跑一次差分 ----
    // 滞留是秒级事件，抽样到 5fps（sample_every=4 @20FPS）完全够；CPU 直接省 4 倍。
    if (cfg_.sample_every < 1) cfg_.sample_every = 1;
    if (++call_count_ % cfg_.sample_every != 0) return ev;

    full_w_ = full_frame.cols;
    full_h_ = full_frame.rows;
    const int64_t now_ms = capture_ms;

    // ---- 1) 检测区 ROI（只算一次，掩膜是固定的）----
    if (!roi_ready_) prepareRoi(region_mask);
    cv::Rect roi = roi_rect_;
    if (roi.width <= 0 || roi.height <= 0 || roi.x + roi.width > full_w_ ||
        roi.y + roi.height > full_h_) {
        roi = cv::Rect(0, 0, full_w_, full_h_);
    }

    // ---- 2) 裁 ROI → 灰度（尽量 1:1，不损失小目标像素）----
    if (cfg_.use_roi_crop) {
        const cv::Mat roi_src = full_frame(roi);
        double s = 1.0;
        if (cfg_.work_max_w > 0 && roi.width > cfg_.work_max_w)
            s = (double)cfg_.work_max_w / roi.width;
        if (s < 0.999) {
            cv::resize(roi_src, small_color_,
                       cv::Size(std::max(1, (int)std::lround(roi.width * s)),
                                std::max(1, (int)std::lround(roi.height * s))),
                       0, 0, cv::INTER_AREA);
        } else {
            small_color_ = roi_src;   // 浅拷贝，只读
        }
        inv_sx_ = inv_sy_ = 1.0 / s;
    } else {
        const int ww = cfg_.work_w > 0 ? cfg_.work_w : full_w_;
        const int wh = cfg_.work_h > 0 ? cfg_.work_h : full_h_;
        cv::resize(full_frame, small_color_, cv::Size(ww, wh), 0, 0, cv::INTER_AREA);
        inv_sx_ = (double)full_w_ / ww;
        inv_sy_ = (double)full_h_ / wh;
    }
    cv::cvtColor(small_color_, gray_small_, cv::COLOR_BGR2GRAY);

    // ---- 3) 标准图 seed / 冷启动 ----
    if (!seed_tried_) {
        seed_tried_ = true;
        seedFromRefImage();
        // 参考模式若拿不到基线，mask 会恒为 0（静默不工作）—— 必须明确退回，不能沉默失败。
        if (clampi(cfg_.mode, 0, 2) != (int)AbandonMode::DOUBLE_MOG2 && !ref_bg_ready_) {
            std::cerr << "[ITEM] " << cam_id_
                      << " 参考背景基线不可用 → 本相机退回 DOUBLE_MOG2 冷启动\n";
            cfg_.mode = (int)AbandonMode::DOUBLE_MOG2;
        }
    }
    if (!calib_printed_) {
        printCalibration();
        calib_printed_ = true;
    }
    ++frame_count_;
    const bool warming = (!bg_seeded_ && frame_count_ <= cfg_.warmup_frames);

    // ---- 4) 生成 static mask（内部同时喂两个 MOG2）----
    buildStaticMask();
    if (warming) return ev;   // 冷启动期只喂背景，不检测

    // ---- 5) 光照/场景突变守卫（参考差分专用）----
    if (clampi(cfg_.mode, 0, 2) != (int)AbandonMode::DOUBLE_MOG2 && !ref_mask_.empty()) {
        const double ratio =
            (double)cv::countNonZero(ref_mask_) / (double)ref_mask_.total();
        if (ratio > cfg_.ref_light_change_ratio) {
            if (++light_change_streak_ >= std::max(1, cfg_.ref_light_change_frames)) {
                refreshRefBg(gray_small_);   // 连续多帧大面积变化 = 光照/相机变了
                light_change_streak_ = 0;
            }
            return ev;   // 本帧不出候选（避免整片误报）
        }
        light_change_streak_ = 0;
    }

    // ---- 6) 连通域 + 过滤 → 候选静止块 ----
    std::vector<cv::Rect> blobs;
    std::vector<cv::Point2f> centroids;
    std::vector<int> areas;
    extractStaticBlobs(person_tracks, region_mask, blobs, centroids, areas);

    // ---- 7) 块关联到已有轨迹（贪心最近质心）----
    associateAndUpdate(blobs, centroids, areas, now_ms, ev);

    // ---- 8) 人-物绑定（用现有 person 框）----
    bindOwners(person_tracks, full_frame, now_ms, ev);

    // ---- 9) 状态推进（阈值用"无人关联静止时长"）----
    advanceState(now_ms, ev);

    return ev;
}

// ==================== 候选静止块提取 ====================
void ItemAbandonmentDetector::extractStaticBlobs(const std::vector<TrackedObject>& persons,
                                                 const cv::Mat& region_mask,
                                                 std::vector<cv::Rect>& out_blobs,
                                                 std::vector<cv::Point2f>& out_centroids,
                                                 std::vector<int>& out_areas) {
    out_blobs.clear(); out_centroids.clear(); out_areas.clear();
    if (mask_combined_.empty()) return;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_combined_, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double full_area = (double)full_w_ * (double)full_h_;
    for (const auto& c : contours) {
        // 面积按全分辨率口径换算，保证与配置的 min_item_area 同一量纲
        const double area_full = cv::contourArea(c) * inv_sx_ * inv_sy_;
        if (area_full < cfg_.min_item_area) continue;

        cv::Rect bf = toFull(cv::boundingRect(c));
        bf &= cv::Rect(0, 0, full_w_, full_h_);
        if (bf.width <= 0 || bf.height <= 0) continue;
        // 最小边长（8cm 门限的像素口径）
        if (std::min(bf.width, bf.height) < cfg_.min_item_px) continue;
        // 面积上限：防整片前景（光照突变）误报
        if (full_area > 0 && (double)bf.area() / full_area > cfg_.max_area_ratio) continue;

        const cv::Point2f cen(bf.x + bf.width * 0.5f, bf.y + bf.height * 0.5f);

        // ROI 是矩形 bbox，检测区是多边形：矩形内仍可能有区外部分，用掩膜做精确判定
        if (!region_mask.empty()) {
            const int cx = clampi((int)cen.x, 0, region_mask.cols - 1);
            const int cy = clampi((int)cen.y, 0, region_mask.rows - 1);
            if (region_mask.at<uchar>(cy, cx) == 0) continue;
        }

        // 排除"站着不动的人"：参考差分同样会把静止的人判成新前景块
        if (overlapsPerson(bf, persons)) continue;

        out_blobs.push_back(bf);
        out_centroids.push_back(cen);
        out_areas.push_back((int)area_full);
    }
}

// ==================== 关联 + 移动重置 + 拾取清除 ====================
void ItemAbandonmentDetector::associateAndUpdate(const std::vector<cv::Rect>& blobs,
                                                 const std::vector<cv::Point2f>& centroids,
                                                 const std::vector<int>& /*areas*/,
                                                 int64_t now_ms,
                                                 std::vector<ItemEvent>& ev) {
    const size_t nb = blobs.size();
    const size_t ni = items_.size();
    std::vector<int> blob2item(nb, -1);
    std::vector<int> item2blob(ni, -1);

    // 收集候选对，按距离升序贪心匹配（块数量少，O(n²) 足够）
    std::vector<std::tuple<double, int, int>> cand;
    for (size_t b = 0; b < nb; ++b) {
        for (size_t k = 0; k < ni; ++k) {
            const double d = dist(centroids[b], items_[k].centroid);
            if (d <= (double)cfg_.assoc_dist_px) cand.emplace_back(d, (int)b, (int)k);
        }
    }
    std::sort(cand.begin(), cand.end(),
              [](const std::tuple<double, int, int>& a, const std::tuple<double, int, int>& b) {
                  return std::get<0>(a) < std::get<0>(b);
              });
    for (const auto& c : cand) {
        const int b = std::get<1>(c), k = std::get<2>(c);
        if (blob2item[b] >= 0 || item2blob[k] >= 0) continue;
        blob2item[b] = k;
        item2blob[k] = b;
    }

    const float a = clampf(cfg_.smooth_alpha, 0.05f, 1.0f);

    // ---- 已匹配：更新质心/框，判"是否在移动"----
    for (size_t k = 0; k < ni; ++k) {
        if (item2blob[k] < 0) continue;
        ItemTrack& it = items_[k];
        const int b = item2blob[k];
        const double disp = dist(centroids[b], it.centroid);

        it.centroid.x += (centroids[b].x - it.centroid.x) * a;
        it.centroid.y += (centroids[b].y - it.centroid.y) * a;
        const cv::Rect& nbb = blobs[b];
        it.box.x      = (int)std::lround(it.box.x      + (nbb.x      - it.box.x)      * a);
        it.box.y      = (int)std::lround(it.box.y      + (nbb.y      - it.box.y)      * a);
        it.box.width  = (int)std::lround(it.box.width  + (nbb.width  - it.box.width)  * a);
        it.box.height = (int)std::lround(it.box.height + (nbb.height - it.box.height) * a);
        if (it.box.width <= 0)  it.box.width = 1;
        if (it.box.height <= 0) it.box.height = 1;

        it.missing_frames = 0;
        it.last_seen_ms = now_ms;

        // 位移超阈值 = 被移动 → 重置计时（"移动位置，重置计时"）
        if (disp > (double)cfg_.reset_dist_px) {
            const bool was_static = (it.state != ItemState::MOVING) || it.unattended_since_ms != 0
                                    || it.emitted_remind || it.emitted_push;
            if (was_static) {
                ItemEvent e;
                e.item_id = it.id;
                e.type = ItemEventType::RESET;
                e.owner_id = it.owner_id;
                e.box = it.box;
                e.static_sec = 0;
                e.note = "位移重置";
                e.ts_ms = now_ms;
                ev.push_back(e);
            }
            it.state = ItemState::MOVING;
            it.static_since_ms = now_ms;
            it.unattended_since_ms = 0;
            it.owner_id = -1;
            it.bind_hits = 0;
            it.owner_left = false;
            it.emitted_remind = false;
            it.emitted_push = false;
            it.evidence_saved = false;
            it.last_evidence_ms = 0;
            it.evidence.clear();
        }
    }

    // ---- 未匹配的块 → 新轨迹 ----
    for (size_t b = 0; b < nb; ++b) {
        if (blob2item[b] >= 0) continue;
        ItemTrack it;
        it.id = next_item_id_++;
        it.centroid = centroids[b];
        it.box = blobs[b];
        it.state = ItemState::MOVING;
        it.first_seen_ms = now_ms;
        it.last_seen_ms = now_ms;
        it.static_since_ms = now_ms;
        // 新块先按"无人关联"起算；若旁边有人，bindOwners 会在本帧立刻清零
        it.unattended_since_ms = now_ms;
        items_.push_back(it);
    }

    // ---- 未匹配的轨迹 → 连续丢失超限 = 被拾取/移出 ----
    // 用重建法而不是边遍历边 erase：上面刚 push 过新轨迹，items_ 与 item2blob 长度
    // 已不对齐，原地 erase 会让后续下标全部错位（历史踩坑）。
    {
        std::vector<ItemTrack> kept;
        kept.reserve(items_.size());
        for (size_t k = 0; k < items_.size(); ++k) {
            ItemTrack& it = items_[k];
            const bool matched = (k < item2blob.size() && item2blob[k] >= 0);
            if (matched) { kept.push_back(std::move(it)); continue; }
            ++it.missing_frames;
            if (it.missing_frames > cfg_.item_lost_frames) {
                if (it.state != ItemState::MOVING || it.emitted_remind || it.emitted_push) {
                    ItemEvent e;
                    e.item_id = it.id;
                    e.type = ItemEventType::CLEAR;
                    e.owner_id = it.owner_id;
                    e.box = it.box;
                    e.static_sec = it.unattended_since_ms > 0
                                 ? (now_ms - it.unattended_since_ms) / 1000.0 : 0.0;
                    e.note = "被拾取/移出";
                    e.ts_ms = now_ms;
                    ev.push_back(e);
                }
                continue;   // 丢弃该轨迹
            }
            kept.push_back(std::move(it));
        }
        items_.swap(kept);
    }
}

// ==================== 人-物绑定 ====================
void ItemAbandonmentDetector::bindOwners(const std::vector<TrackedObject>& persons,
                                         const cv::Mat& full_frame,
                                         int64_t now_ms,
                                         std::vector<ItemEvent>& ev) {
    for (auto& it : items_) {
        // 找最近的人（人物框中心 ↔ 物品质心）
        const TrackedObject* best = nullptr;
        double best_d = 1e18;
        for (const auto& p : persons) {
            const cv::Point2f pc(p.box.x + p.box.width * 0.5f, p.box.y + p.box.height * 0.5f);
            const double d = dist(pc, it.centroid);
            if (d < best_d) { best_d = d; best = &p; }
        }

        const bool attended = (best != nullptr) && (best_d <= (double)cfg_.safe_radius_px);
        if (attended) {
            ++it.bind_hits;
            it.owner_left = false;
            if (it.bind_hits >= cfg_.bind_confirm_frames || it.owner_id < 0)
                it.owner_id = best->track_id;

            // 有人看管 → 计时暂停/清零（对应"无人持续关联才计时"）
            it.unattended_since_ms = 0;
            pushEvidence(it, full_frame, now_ms);

            // 主人返回 → 清除告警
            if (it.state == ItemState::LEVEL1_REMIND || it.state == ItemState::LEVEL2_PUSH) {
                ItemEvent e;
                e.item_id = it.id;
                e.type = ItemEventType::CLEAR;
                e.owner_id = best->track_id;
                e.box = it.box;
                e.note = "主人返回";
                e.ts_ms = now_ms;
                ev.push_back(e);
                it.emitted_remind = false;
                it.emitted_push = false;
                it.evidence_saved = false;
            }
            if (it.state != ItemState::MOVING) it.state = ItemState::STATIC_CANDIDATE;
        } else {
            it.bind_hits = 0;
            // 从"有人看管"变为"无人看管" → 开始计时 + 保存滞留前的人-物序列
            if (it.unattended_since_ms == 0) {
                it.unattended_since_ms = now_ms;
                if (it.owner_id >= 0) it.owner_left = true;
                saveEvidenceSequence(it, now_ms);
            }
        }
    }
}

// ==================== 状态推进 ====================
void ItemAbandonmentDetector::advanceState(int64_t now_ms, std::vector<ItemEvent>& ev) {
    for (auto& it : items_) {
        if (it.unattended_since_ms == 0) {   // 有人看管中
            if (it.state != ItemState::MOVING) it.state = ItemState::STATIC_CANDIDATE;
            continue;
        }
        const double sec = (now_ms - it.unattended_since_ms) / 1000.0;

        if (sec >= cfg_.push_sec && !it.emitted_push) {
            it.state = ItemState::LEVEL2_PUSH;
            it.emitted_push = true;
            it.emitted_remind = true;        // 二级已发，一级不再补发
            ItemEvent e;
            e.item_id = it.id;
            e.type = ItemEventType::PUSH;
            e.owner_id = it.owner_id;
            e.box = it.box;
            e.static_sec = sec;
            e.note = "无人关联滞留";
            e.ts_ms = now_ms;
            ev.push_back(e);
        } else if (sec >= cfg_.remind_sec && !it.emitted_remind) {
            it.state = ItemState::LEVEL1_REMIND;
            it.emitted_remind = true;
            ItemEvent e;
            e.item_id = it.id;
            e.type = ItemEventType::REMIND;
            e.owner_id = it.owner_id;
            e.box = it.box;
            e.static_sec = sec;
            e.note = "无人关联滞留";
            e.ts_ms = now_ms;
            ev.push_back(e);
        } else if (sec < cfg_.remind_sec) {
            it.state = ItemState::STATIC_CANDIDATE;
        }
        // 否则：处于 [remind, push) 且一级已发 → 保持 LEVEL1_REMIND；push 已发 → 保持 LEVEL2_PUSH
    }
}

// ==================== 证据帧 ====================
void ItemAbandonmentDetector::pushEvidence(ItemTrack& it, const cv::Mat& full_frame, int64_t now_ms) {
    if (cfg_.seq_fps <= 0) return;
    const int64_t interval = (int64_t)(1000.0 / cfg_.seq_fps);
    if (it.last_evidence_ms != 0 && (now_ms - it.last_evidence_ms) < interval) return;
    it.last_evidence_ms = now_ms;

    // 存小图（320 宽）控制内存：50 帧 ≈ 8MB/物品
    const int ew = 320;
    const int eh = full_frame.rows * ew / std::max(1, full_frame.cols);
    cv::Mat small;
    cv::resize(full_frame, small, cv::Size(ew, eh > 0 ? eh : 180), 0, 0, cv::INTER_AREA);
    it.evidence.emplace_back(now_ms, small);

    const size_t max_frames = (size_t)std::max(1.0, cfg_.backtrack_sec * cfg_.seq_fps);
    while (it.evidence.size() > max_frames) it.evidence.pop_front();
}

void ItemAbandonmentDetector::saveEvidenceSequence(ItemTrack& it, int64_t center_ms) {
    if (it.evidence_saved) return;
    it.evidence_saved = true;   // 每个物品只落一次（避免反复触发反复写盘）
    if (it.evidence.empty()) {
        std::cout << "[ITEM] " << cam_id_ << " item=" << it.id
                  << " 无证据帧可保存（该物品出现时主人已不在场）" << std::endl;
        return;
    }
    const int64_t from_ms = center_ms - (int64_t)(cfg_.backtrack_sec * 1000.0);

    char dir[512];
    snprintf(dir, sizeof(dir), "%s/%s_item%d_%lld",
             cfg_.save_dir.c_str(), cam_id_.c_str(), it.id, (long long)center_ms);
    mkdirP(cfg_.save_dir);
    mkdirP(dir);

    int k = 0;
    for (const auto& f : it.evidence) {
        if (f.first < from_ms || f.first > center_ms) continue;
        char fn[640];
        snprintf(fn, sizeof(fn), "%s/%03d.jpg", dir, k++);
        cv::imwrite(fn, f.second);
    }
    std::cout << "[ITEM] " << cam_id_ << " item=" << it.id
              << " 已保存滞留前人-物关联序列 " << k << " 帧 → " << dir << std::endl;
}

// ==================== 工具 ====================
double ItemAbandonmentDetector::dist(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = (double)a.x - b.x, dy = (double)a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool ItemAbandonmentDetector::overlapsPerson(const cv::Rect& box,
                                             const std::vector<TrackedObject>& persons) const {
    if (box.area() <= 0) return false;
    for (const auto& p : persons) {
        const cv::Rect inter = box & p.box;
        const double ia = (double)inter.area();
        if (ia <= 0) continue;
        const double uni = (double)box.area() + (double)p.box.area() - ia;
        const double iou = uni > 0 ? ia / uni : 0.0;
        if (iou > (double)cfg_.person_iou_reject) return true;
    }
    return false;
}

// ==================== 统计 ====================
int ItemAbandonmentDetector::alarmCount() const {
    int n = 0;
    for (const auto& it : items_)
        if (it.state == ItemState::LEVEL1_REMIND || it.state == ItemState::LEVEL2_PUSH) ++n;
    return n;
}
int ItemAbandonmentDetector::level1Count() const {
    int n = 0;
    for (const auto& it : items_) if (it.state == ItemState::LEVEL1_REMIND) ++n;
    return n;
}
int ItemAbandonmentDetector::level2Count() const {
    int n = 0;
    for (const auto& it : items_) if (it.state == ItemState::LEVEL2_PUSH) ++n;
    return n;
}

// ==================== 配置加载 ====================
namespace {

// 把一条 "key=value" 应用到配置；key 已去掉 roleN. 前缀
bool applyAbandonKV(AbandonConfig& c, const std::string& key, const std::string& val) {
    auto i = [&](int def) { try { return std::stoi(val); } catch (...) { return def; } };
    auto d = [&](double def) { try { return std::stod(val); } catch (...) { return def; } };
    auto b = [&]() { return !(val == "0" || val == "false" || val == "no" || val == "off"); };

    if (key == "enable")            c.enable = b();
    else if (key == "mode") {
        if (val == "mog2" || val == "double_mog2" || val == "0")      c.mode = 0;
        else if (val == "ref" || val == "ref_bg_diff" || val == "1")  c.mode = 1;
        else if (val == "hybrid" || val == "2")                       c.mode = 2;
    }
    // bg_seed_path 是旧名，等价于 ref_bg_path（兼容老配置）
    else if (key == "ref_bg_path" || key == "bg_seed_path") c.ref_bg_path = val;
    else if (key == "use_roi_crop")  c.use_roi_crop = i(c.use_roi_crop);
    else if (key == "roi_margin_px") c.roi_margin_px = i(c.roi_margin_px);
    else if (key == "work_max_w")    c.work_max_w = i(c.work_max_w);
    else if (key == "ref_diff_thresh") c.ref_diff_thresh = d(c.ref_diff_thresh);
    else if (key == "ref_light_change_ratio") c.ref_light_change_ratio = d(c.ref_light_change_ratio);
    else if (key == "ref_light_change_frames") c.ref_light_change_frames = i(c.ref_light_change_frames);
    else if (key == "morph_open_px")  c.morph_open_px = i(c.morph_open_px);
    else if (key == "morph_close_px") c.morph_close_px = i(c.morph_close_px);
    else if (key == "region_width_cm") c.region_width_cm = d(c.region_width_cm);
    else if (key == "work_w")       c.work_w = i(c.work_w);
    else if (key == "work_h")       c.work_h = i(c.work_h);
    else if (key == "warmup_frames")c.warmup_frames = i(c.warmup_frames);
    else if (key == "sample_every") c.sample_every = i(c.sample_every);
    else if (key == "use_own_thread") c.use_own_thread = b();
    else if (key == "bg_long_lr")   c.bg_long_lr = d(c.bg_long_lr);
    else if (key == "bg_short_lr")  c.bg_short_lr = d(c.bg_short_lr);
    else if (key == "min_item_px")  c.min_item_px = i(c.min_item_px);
    else if (key == "min_item_area")c.min_item_area = d(c.min_item_area);
    else if (key == "max_area_ratio") c.max_area_ratio = d(c.max_area_ratio);
    else if (key == "assoc_dist_px")c.assoc_dist_px = (float)d(c.assoc_dist_px);
    else if (key == "item_lost_frames") c.item_lost_frames = i(c.item_lost_frames);
    else if (key == "reset_dist_px")c.reset_dist_px = (float)d(c.reset_dist_px);
    else if (key == "smooth_alpha") c.smooth_alpha = (float)d(c.smooth_alpha);
    else if (key == "safe_radius_px") c.safe_radius_px = (float)d(c.safe_radius_px);
    else if (key == "bind_confirm_frames") c.bind_confirm_frames = i(c.bind_confirm_frames);
    else if (key == "person_iou_reject") c.person_iou_reject = (float)d(c.person_iou_reject);
    else if (key == "remind_sec")   c.remind_sec = d(c.remind_sec);
    else if (key == "push_sec")     c.push_sec = d(c.push_sec);
    else if (key == "backtrack_sec")c.backtrack_sec = d(c.backtrack_sec);
    else if (key == "seq_fps")      c.seq_fps = i(c.seq_fps);
    else if (key == "save_dir")     c.save_dir = val;
    else return false;
    return true;
}

}  // namespace

AbandonConfig loadAbandonConfig(const std::string& path, int role) {
    AbandonConfig c;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cout << "[ITEM] 未找到 " << path << " → 滞留检测使用内置默认值（enable="
                  << (c.enable ? 1 : 0) << "）" << std::endl;
        return c;
    }
    const std::string role_prefix = "role" + std::to_string(role) + ".";
    std::vector<std::pair<std::string, std::string>> role_kv;  // 角色覆盖，最后统一应用
    std::string line;
    while (std::getline(f, line)) {
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';') continue;
        const size_t eq = t.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(t.substr(0, eq));
        const std::string val = trim(t.substr(eq + 1));
        if (key.empty()) continue;

        if (key.rfind(role_prefix, 0) == 0) {          // roleN.xxx → 覆盖
            role_kv.emplace_back(key.substr(role_prefix.size()), val);
        } else if (key.rfind("role", 0) == 0 && key.find('.') != std::string::npos) {
            continue;                                   // 别的角色的覆盖，忽略
        } else {
            applyAbandonKV(c, key, val);
        }
    }
    for (const auto& kv : role_kv) applyAbandonKV(c, kv.first, kv.second);

    static const char* kModeName[] = {"DOUBLE_MOG2", "REF_BG_DIFF", "HYBRID"};
    std::cout << "[ITEM] " << path << " 已加载 (role=" << role << "): enable=" << (c.enable ? 1 : 0)
              << " mode=" << kModeName[clampi(c.mode, 0, 2)]
              << " roi_crop=" << c.use_roi_crop
              << " min_item_px=" << c.min_item_px << " min_area=" << c.min_item_area
              << " remind=" << c.remind_sec << "s push=" << c.push_sec << "s"
              << " save_dir=" << c.save_dir << std::endl;
    return c;
}
