#include "tracker.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

namespace {
// ---- 遮挡重连（二次关联）参数 ----
// 搜索窗半径 = max(kReidBaseRadiusPx, kReidBoxRatio * 框最长边)
//             + kReidPerLostPx * 已失联帧数，上限 kReidMaxRadiusPx。
// 失联越久、人可能走得越远，窗口随之放大；但始终以"人不可能瞬移"为约束。
constexpr float kReidBaseRadiusPx = 60.0f;
constexpr float kReidBoxRatio     = 0.75f;
constexpr float kReidPerLostPx    = 8.0f;
constexpr float kReidMaxRadiusPx  = 260.0f;
// 外观相关性硬门限（HSV 2D 直方图 HISTCMP_CORREL）。同一人不同帧普遍 >0.5，
// 不同衣着的人普遍 <0.35；低于门限一律不关联，宁可让它新建 ID（避免 ID 盗用）。
constexpr float kReidMinAppearance = 0.35f;
// 综合评分：外观权重最高（遮挡重现身最可靠的线索），距离/运动作约束。
constexpr float kReidWDist   = 0.20f;
constexpr float kReidWMotion = 0.25f;
constexpr float kReidWAppear = 0.55f;
constexpr float kReidScoreThresh = 0.45f;
// 帧为空/直方图不可用时的保守兜底（只能靠距离+运动，门限更高）
constexpr float kReidScoreThreshNoAppear = 0.60f;
}

ByteTrackTracker::ByteTrackTracker(float track_thresh, float high_thresh,
                                 float match_thresh, int max_lost, int min_hits)
    : track_thresh_(track_thresh),
      high_thresh_(high_thresh),
      match_thresh_(match_thresh),
      max_lost_(max_lost),
      min_hits_(min_hits),
      next_id_(0),
      rng_((uint64)std::time(nullptr)) {}

float ByteTrackTracker::calcIou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    if (inter.area() <= 0) return 0.0f;
    float ua = (float)(a.area() + b.area() - inter.area());
    return ua > 0 ? (float)inter.area() / ua : 0.0f;
}

void ByteTrackTracker::greedyMatch(
    const std::vector<Track>& tracks,
    const std::vector<Detection>& dets,
    float iou_thresh,
    std::vector<int>& track_match,
    std::vector<int>& det_match) {
    
    track_match.assign(tracks.size(), -1);
    det_match.assign(dets.size(), -1);
    if (tracks.empty() || dets.empty()) return;
    
    struct Pair { float iou; int ti; int di; };
    std::vector<Pair> pairs;
    for (size_t i = 0; i < tracks.size(); i++)
        for (size_t j = 0; j < dets.size(); j++) {
            float iou = calcIou(tracks[i].rect, dets[j].box);
            if (iou >= iou_thresh)
                pairs.push_back({iou, (int)i, (int)j});
        }
    
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.iou > b.iou; });
    
    std::vector<bool> t_used(tracks.size(), false);
    std::vector<bool> d_used(dets.size(), false);
    for (auto& p : pairs) {
        if (!t_used[p.ti] && !d_used[p.di]) {
            track_match[p.ti] = p.di;
            det_match[p.di] = p.ti;
            t_used[p.ti] = true;
            d_used[p.di] = true;
        }
    }
}

// ==================== 联合评分模型 ====================

cv::Mat ByteTrackTracker::computeAppearanceHistogram(const cv::Mat& frame, const cv::Rect& box) {
    cv::Rect valid = box & cv::Rect(0, 0, frame.cols, frame.rows);
    if (valid.area() <= 0) return cv::Mat();
    cv::Mat patch = frame(valid);
    cv::Mat hsv;
    cv::cvtColor(patch, hsv, cv::COLOR_BGR2HSV);
    int h_bins = 16, s_bins = 16;
    int histSize[] = {h_bins, s_bins};
    float h_range[] = {0, 180}, s_range[] = {0, 256};
    const float* ranges[] = {h_range, s_range};
    int channels[] = {0, 1};
    cv::Mat hist;
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

float ByteTrackTracker::calcJointScore(const Track& track, const Detection& det,
                                        const cv::Mat& frame, const cv::Mat& det_hist) {
    // 1. IoU 评分 (0~1)
    float s_iou = calcIou(track.rect, det.box);

    // 2. 运动一致性评分：马氏距离 → (0,1]
    float mahal = track.kf.mahalanobisDistance(det.box);
    float s_motion = 1.0f / (1.0f + mahal);   // Mahal=0 → 1.0, Mahal=10 → 0.09

    // 3. 外观相似度评分：直方图相关性 [0,1]
    float s_appear = 0.0f;
    if (!track.appearance_hist.empty() && !det_hist.empty()) {
        s_appear = (float)cv::compareHist(track.appearance_hist, det_hist, cv::HISTCMP_CORREL);
        s_appear = std::max(0.0f, s_appear);
    }

    // 4. 置信度项：高置信度检测本身更可信
    float s_conf = det.confidence;

    // 权重：IoU主导 + 运动约束 + 外观辅助 + 置信度偏好
    // 遮挡重现身靠外观重识别降 ID 切换，故 γ 0.20→0.25（α 0.40→0.35 补足，和仍 1.0）
    const float alpha = 0.35f;   // IoU
    const float beta  = 0.25f;   // Motion
    const float gamma = 0.25f;   // Appearance
    const float delta = 0.15f;   // Confidence

    return alpha * s_iou + beta * s_motion + gamma * s_appear + delta * s_conf;
}

void ByteTrackTracker::calibrateConfidence(const std::vector<Track>& tracks,
                                            std::vector<Detection>& det_low,
                                            std::vector<Detection>& det_high) {
    for (const auto& t : tracks) {
        // 只对已确认的稳定 track 做校准（连续匹配 ≥5 帧，最近有更新）
        if (!t.is_activated) continue;
        if (t.time_since_update > 8) continue;
        if (t.stable_count < 5) continue;

        for (int j = (int)det_low.size() - 1; j >= 0; j--) {
            float mahal = t.kf.mahalanobisDistance(det_low[j].box);

            // 马氏距离 < 3.0（~99.7% 置信区间内）→ 运动高度一致，置信度可信
            if (mahal < 3.0f) {
                // 根据运动匹配度动态提升置信度
                float boost = 1.0f / (1.0f + mahal);            // (0, 1]
                float boosted_conf = det_low[j].confidence
                                   + (high_thresh_ - det_low[j].confidence) * boost * 0.6f;
                boosted_conf = std::min(boosted_conf, 0.95f);

                if (boosted_conf >= high_thresh_) {
                    Detection calib = det_low[j];
                    calib.confidence = boosted_conf;
                    det_high.push_back(calib);
                    det_low.erase(det_low.begin() + j);
                }
            }
        }
    }
}

void ByteTrackTracker::greedyJointMatch(
    const std::vector<Track>& tracks,
    const std::vector<Detection>& dets,
    float score_thresh,
    const cv::Mat& frame,
    const std::vector<cv::Mat>& det_hists,
    std::vector<int>& track_match,
    std::vector<int>& det_match) {

    track_match.assign(tracks.size(), -1);
    det_match.assign(dets.size(), -1);
    if (tracks.empty() || dets.empty()) return;

    struct Pair { float score; int ti; int di; };
    std::vector<Pair> pairs;
    for (size_t i = 0; i < tracks.size(); i++)
        for (size_t j = 0; j < dets.size(); j++) {
            // 快速 IoU 预过滤：IoU=0 的直接跳过，避免昂贵的直方图比较
            float iou_fast = calcIou(tracks[i].rect, dets[j].box);
            if (iou_fast < 0.05f) continue;
            // IoU 很高的直接给高分，跳过 Mahalanobis + 直方图
            float s;
            if (iou_fast > 0.7f) {
                s = 0.7f * iou_fast + 0.3f * dets[j].confidence;
            } else {
                s = calcJointScore(tracks[i], dets[j], frame, det_hists[j]);
            }
            if (s >= score_thresh)
                pairs.push_back({s, (int)i, (int)j});
        }

    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.score > b.score; });

    std::vector<bool> t_used(tracks.size(), false);
    std::vector<bool> d_used(dets.size(), false);
    for (auto& p : pairs) {
        if (!t_used[p.ti] && !d_used[p.di]) {
            track_match[p.ti] = p.di;
            det_match[p.di] = p.ti;
            t_used[p.ti] = true;
            d_used[p.di] = true;
        }
    }
}

// ==================== 遮挡重连（二次关联） ====================

void ByteTrackTracker::reassociateLost(
    const std::vector<Track>& tracks,
    const std::vector<int>& free_track_idx,
    const std::vector<Detection>& dets,
    const std::vector<cv::Mat>& det_hists,
    std::vector<int>& track_match,
    std::vector<int>& det_match) {

    track_match.assign(tracks.size(), -1);
    det_match.assign(dets.size(), -1);
    if (free_track_idx.empty() || dets.empty()) return;

    struct Pair { float score; int ti; int di; };
    std::vector<Pair> pairs;

    for (int ti : free_track_idx) {
        if (ti < 0 || ti >= (int)tracks.size()) continue;
        const Track& t = tracks[ti];
        // time_since_update 在本轮 predict() 后已自增：1 = 本帧刚失配，
        // 2 及以上 = 已连续失联 (time_since_update-1) 帧。
        const int lost = std::max(0, t.time_since_update - 1);
        const float base = std::max(kReidBaseRadiusPx,
                                    kReidBoxRatio * (float)std::max(t.rect.width,
                                                                    t.rect.height));
        const float radius = std::min(kReidMaxRadiusPx,
                                      base + kReidPerLostPx * (float)lost);
        const cv::Point2f tc(t.rect.x + t.rect.width * 0.5f,
                             t.rect.y + t.rect.height * 0.5f);

        for (size_t di = 0; di < dets.size(); di++) {
            const Detection& d = dets[di];
            const cv::Point2f dc(d.box.x + d.box.width * 0.5f,
                                 d.box.y + d.box.height * 0.5f);
            const float dist = (float)cv::norm(dc - tc);
            if (dist > radius) continue;

            // 距离评分：窗口内线性衰减（窗口边缘 = 0），窗口本身已按失联帧数放大
            const float s_dist = std::max(0.0f, 1.0f - dist / radius);
            const float mahal = t.kf.mahalanobisDistance(d.box);
            const float s_motion = 1.0f / (1.0f + mahal);                 // (0,1]

            float s_appear = -1.0f;                                       // <0 = 无外观证据
            if (di < det_hists.size() && !t.appearance_hist.empty() &&
                !det_hists[di].empty()) {
                s_appear = std::max(0.0f,
                    (float)cv::compareHist(t.appearance_hist, det_hists[di],
                                           cv::HISTCMP_CORREL));
                if (s_appear < kReidMinAppearance) continue;              // 外观不像，放弃
            } else if (lost == 0) {
                continue;   // 无外观证据且本帧才失配 → 不冒险建关联
            }

            float score = 0.0f;
            float thresh = kReidScoreThreshNoAppear;
            if (s_appear >= 0.0f) {
                score = kReidWDist * s_dist + kReidWMotion * s_motion +
                        kReidWAppear * s_appear;
                thresh = kReidScoreThresh;
            } else {
                score = 0.5f * s_dist + 0.5f * s_motion;
            }
            if (score >= thresh) pairs.push_back({score, ti, (int)di});
        }
    }

    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.score > b.score; });

    std::vector<bool> t_used(tracks.size(), false);
    std::vector<bool> d_used(dets.size(), false);
    for (auto& p : pairs) {
        if (!t_used[p.ti] && !d_used[p.di]) {
            track_match[p.ti] = p.di;
            det_match[p.di] = p.ti;
            t_used[p.ti] = true;
            d_used[p.di] = true;
        }
    }
}

// ==================== 核心更新流程 ====================

std::vector<TrackedObject> ByteTrackTracker::update(
    const std::vector<Detection>& detections,
    const cv::Mat& frame,
    int img_width, int img_height) {

    (void)img_width;
    (void)img_height;

    // ====== Step 0: 分离高低置信度检测 ======
    std::vector<Detection> det_high, det_low;
    for (const auto& d : detections) {
        if (d.confidence >= high_thresh_)
            det_high.push_back(d);
        else if (d.confidence >= track_thresh_)
            det_low.push_back(d);
    }

    // ====== Step 1: Kalman 预测所有活跃 track ======
    for (auto& t : active_tracks_)
        t.predict();

    // ====== Step 1.5（NEW）: 轨迹引导置信度校准 ======
    // 对稳定 track，将马氏距离匹配度高的低分检测提升置信度并纳入高置信度池
    if (!det_low.empty() && !frame.empty())
        calibrateConfidence(active_tracks_, det_low, det_high);

    // ====== Step 2: 预计算高置信度检测的外观直方图 ======
    std::vector<cv::Mat> det_hists_high(det_high.size());
    if (!frame.empty()) {
        for (size_t j = 0; j < det_high.size(); j++)
            det_hists_high[j] = computeAppearanceHistogram(frame, det_high[j].box);
    }

    // ====== Step 2: 第一轮联合评分匹配 — 活跃track vs 高置信度检测 ======
    std::vector<int> t_match1, d_match1;
    if (!frame.empty() && !det_high.empty()) {
        greedyJointMatch(active_tracks_, det_high, 0.35f, frame,
                         det_hists_high, t_match1, d_match1);
    } else {
        // 无 frame 时回退纯 IoU
        greedyMatch(active_tracks_, det_high, match_thresh_, t_match1, d_match1);
    }

    std::vector<bool> t_matched1(active_tracks_.size(), false);
    for (size_t i = 0; i < t_match1.size(); i++)
        if (t_match1[i] >= 0) t_matched1[i] = true;

    // ====== Step 3: 第二轮匹配 — 未匹配track vs 低置信度检测 ======
    std::vector<int> t2_indices;
    for (size_t i = 0; i < t_matched1.size(); i++)
        if (!t_matched1[i]) t2_indices.push_back((int)i);

    // 只有存在未匹配 track 时才计算低分检测直方图，避免浪费
    std::vector<cv::Mat> det_hists_low;
    bool need_second_match = !t2_indices.empty() && !det_low.empty() && !frame.empty();
    if (need_second_match) {
        det_hists_low.resize(det_low.size());
        for (size_t j = 0; j < det_low.size(); j++)
            det_hists_low[j] = computeAppearanceHistogram(frame, det_low[j].box);
    }

    std::vector<Track> t2_tracks;
    for (int idx : t2_indices)
        t2_tracks.push_back(active_tracks_[idx]);

    std::vector<int> t_match2, d_match2;
    if (need_second_match) {
        greedyJointMatch(t2_tracks, det_low, 0.30f, frame,
                         det_hists_low, t_match2, d_match2);
    } else {
        greedyMatch(t2_tracks, det_low, match_thresh_, t_match2, d_match2);
    }

    std::vector<bool> t_matched2(active_tracks_.size(), false);
    for (size_t k = 0; k < t_match2.size(); k++)
        if (t_match2[k] >= 0) t_matched2[t2_indices[k]] = true;

    // ====== Step 3.5（NEW）: 遮挡重连 — 未匹配 track vs 未匹配高分检测 ======
    // 抑制遮挡导致的 ID 切换：主通道(iou<0.05 预过滤)在"外推框与真实框不重叠"
    // 时会直接失配，这里用搜索窗 + 运动/外观做二次关联，把同一个人的检测挂回旧 ID。
    std::vector<int> t_match3(active_tracks_.size(), -1);   // 值 = det_high 下标
    std::vector<bool> t_matched3(active_tracks_.size(), false);
    std::vector<bool> d_matched3(det_high.size(), false);
    if (!frame.empty() && !det_high.empty()) {
        std::vector<int> free_tracks;
        for (size_t i = 0; i < active_tracks_.size(); i++)
            if (!t_matched1[i] && !t_matched2[i]) free_tracks.push_back((int)i);

        std::vector<int> free_dets;                 // det_high 下标
        for (size_t j = 0; j < det_high.size(); j++)
            if (d_match1[j] < 0) free_dets.push_back((int)j);

        if (!free_tracks.empty() && !free_dets.empty()) {
            std::vector<Detection> fdets;
            std::vector<cv::Mat> fhists;
            fdets.reserve(free_dets.size());
            fhists.reserve(free_dets.size());
            for (int j : free_dets) {
                fdets.push_back(det_high[j]);
                fhists.push_back(det_hists_high[j]);
            }
            std::vector<int> tm, dm;
            reassociateLost(active_tracks_, free_tracks, fdets, fhists, tm, dm);
            for (size_t k = 0; k < dm.size(); k++) {
                if (dm[k] < 0) continue;
                const int ti = dm[k];
                t_match3[ti] = free_dets[k];
                t_matched3[ti] = true;
                d_matched3[free_dets[k]] = true;
            }
        }
    }

    // ====== Step 4: 更新匹配上的track（含外观直方图EMA） ======
    for (size_t i = 0; i < active_tracks_.size(); i++) {
        const Detection* det_ptr = nullptr;
        cv::Mat det_hist;

        if (t_match1[i] >= 0) {
            det_ptr = &det_high[t_match1[i]];
            det_hist = det_hists_high[t_match1[i]];
        } else if (t_matched3[i]) {
            det_ptr = &det_high[t_match3[i]];
            det_hist = det_hists_high[t_match3[i]];
        } else if (t_matched2[i]) {
            for (size_t k = 0; k < t2_indices.size(); k++) {
                if (t2_indices[k] == (int)i && t_match2[k] >= 0) {
                    det_ptr = &det_low[t_match2[k]];
                    if (t_match2[k] < (int)det_hists_low.size())
                        det_hist = det_hists_low[t_match2[k]];
                    break;
                }
            }
        }

        if (det_ptr) {
            active_tracks_[i].update(*det_ptr);
            // EMA 更新外观直方图
            if (!det_hist.empty()) {
                if (active_tracks_[i].appearance_hist.empty()) {
                    active_tracks_[i].appearance_hist = det_hist.clone();
                } else {
                    cv::addWeighted(active_tracks_[i].appearance_hist, 0.75f,
                                    det_hist, 0.25f, 0,
                                    active_tracks_[i].appearance_hist);
                }
            }
            // 更新稳定度
            active_tracks_[i].stable_count++;
            active_tracks_[i].avg_score = 0.9f * active_tracks_[i].avg_score
                                        + 0.1f * det_ptr->confidence;
        }
    }

    // ====== Step 5: 构建新活跃track列表 ======
    std::vector<Track> new_active;

    for (size_t i = 0; i < active_tracks_.size(); i++) {
        if (t_matched1[i] || t_matched2[i] || t_matched3[i]) {
            new_active.push_back(active_tracks_[i]);
        }
    }

    for (size_t i = 0; i < active_tracks_.size(); i++) {
        if (!t_matched1[i] && !t_matched2[i] && !t_matched3[i]) {
            if (active_tracks_[i].time_since_update < max_lost_) {
                new_active.push_back(active_tracks_[i]);
            }
        }
    }

    // ====== Step 6: 未匹配的高置信度检测创建新track ======
    // 注意：只有"主通道 + 低分通道 + 遮挡重连"三路都没匹配上的检测才会新建 ID。
    std::vector<bool> d_used_high(det_high.size(), false);
    for (size_t i = 0; i < d_match1.size(); i++)
        if (d_match1[i] >= 0) d_used_high[i] = true;
    for (size_t j = 0; j < d_matched3.size(); j++)
        if (d_matched3[j]) d_used_high[j] = true;

    for (size_t j = 0; j < det_high.size(); j++) {
        if (!d_used_high[j]) {
            Track nt;
            nt.activate(++next_id_, det_high[j]);
            // 初始化外观直方图
            if (!det_hists_high[j].empty())
                nt.appearance_hist = det_hists_high[j].clone();
            nt.stable_count = 1;
            nt.avg_score = det_high[j].confidence;
            new_active.push_back(nt);
        }
    }

    active_tracks_ = std::move(new_active);

    // ====== Step 7: 生成输出 ======
    std::vector<TrackedObject> results;
    for (auto& t : active_tracks_) {
        if (t.time_since_update > 3) continue;
        if (!t.is_activated) continue;

        TrackedObject obj;
        obj.track_id = t.id;
        obj.box = t.rect;
        obj.confidence = t.score;
        obj.class_id = 0;
        obj.class_name = "people";
        obj.color = getColor(t.id);
        obj.observed = (t.time_since_update == 0);
        obj.hit_streak = t.hit_streak;
        obj.time_since_update = t.time_since_update;
        results.push_back(obj);
    }

    return results;
}

void ByteTrackTracker::drawTracks(cv::Mat& img, const std::vector<TrackedObject>& tracks) {
    for (const auto& t : tracks) {
        cv::rectangle(img, t.box, t.color, 2);
        std::string label = "P#" + std::to_string(t.track_id);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::rectangle(img, cv::Point(t.box.x, t.box.y - ts.height - 5),
                      cv::Point(t.box.x + ts.width, t.box.y), t.color, cv::FILLED);
        cv::putText(img, label, cv::Point(t.box.x, t.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
}

cv::Scalar ByteTrackTracker::getColor(int track_id) {
    auto it = color_map_.find(track_id);
    if (it != color_map_.end()) return it->second;
    cv::Scalar color((int)rng_.uniform(64, 255),
                     (int)rng_.uniform(64, 255),
                     (int)rng_.uniform(64, 255));
    color_map_[track_id] = color;
    return color;
}
