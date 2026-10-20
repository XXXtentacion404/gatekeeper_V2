// 清场双时相会话 — 实现。设计说明见 include/change_session.h 与
// 文档/FastBSUV-Net2.0_清场双时相比对方案.md

#include "change_session.h"

#include <algorithm>
#include <cmath>

namespace {

inline int64_t sec2ms(double s) { return (int64_t)(s * 1000.0 + 0.5); }

inline double dist2(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = a.x - b.x, dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// 在 box 内求 |a - b| 的平均值（灰度）。box 会被裁到图像范围内。
double meanAbsDiffInBox(const cv::Mat& a, const cv::Mat& b, const cv::Rect& box) {
    const cv::Rect r = box & cv::Rect(0, 0, a.cols, a.rows);
    if (r.width <= 0 || r.height <= 0) return 0.0;
    const cv::Mat da = a(r), db = b(r);
    cv::Mat d;
    cv::absdiff(da, db, d);
    return cv::mean(d)[0];   // 三通道取平均，足够做方向判据
}

}  // namespace

// ==================== 每帧推进 ====================
bool ChangeSession::pushFrame(const cv::Mat& frame_bgr, bool person_present,
                              int64_t now_ms, PairRequest& out) {
    if (frame_bgr.empty()) return false;

    // 尺寸变了（相机重连 / ROI 变化）→ 全部状态作废，否则会拿两张不同尺寸的图比
    if ((has_base_ && base_.size() != frame_bgr.size()) ||
        (has_before_ && before_.size() != frame_bgr.size())) {
        reset();
    }

    if (phase_ == ScenePhase::CLEAR) {
        if (person_present) {
            // —— 有人进入：冻结"入场前帧" = 上一个清场帧 ——
            if (!last_clear_.empty()) {
                before_ = last_clear_.clone();
                has_before_ = true;
                before_ms_ = last_clear_ms_;
            } else {
                has_before_ = false;   // 开机就有人，这次会话没有可比对的起点
            }
            occupied_since_ms_ = now_ms;
            clear_since_ms_ = 0;
            phase_ = ScenePhase::OCCUPIED;
            return false;
        }

        // —— 持续清场 ——
        if (clear_since_ms_ == 0) clear_since_ms_ = now_ms;

        // 基线吸收：只有在"确认清场"且"上一轮没发现新增"时才更新。
        // ⚠️ 这是防"包被吸进背景永久漏报"的第一道闸门。
        // ⚠️ 第二道闸门：残留物所在的矩形区域一律不吸收，保留旧基线的内容。
        //    否则一个被留下的包会在下一轮就变成"场景的一部分"，之后再也检不出来。
        if (allow_base_update_ &&
            (now_ms - clear_since_ms_) >= sec2ms(cfg_.t_clear_sec)) {
            if (!has_base_) {
                base_ = frame_bgr.clone();
            } else {
                cv::Mat nb = frame_bgr.clone();
                const cv::Rect full(0, 0, nb.cols, nb.rows);
                for (const auto& r : residuals_) {
                    const cv::Rect rr = r.box & full & cv::Rect(0, 0, base_.cols, base_.rows);
                    if (rr.width > 0 && rr.height > 0) base_(rr).copyTo(nb(rr));
                }
                base_ = nb;
            }
            base_ms_ = now_ms;
            has_base_ = true;
        }

        last_clear_ = frame_bgr.clone();
        last_clear_ms_ = now_ms;
        return false;
    }

    // ==================== OCCUPIED ====================
    if (!person_present) {
        if (clear_since_ms_ == 0) clear_since_ms_ = now_ms;

        // 有人时也持续刷新"上一个清场帧"，供下一次入场冻结用；
        // 但绝不拿它更新基线。
        if ((now_ms - clear_since_ms_) >= sec2ms(cfg_.t_settle_sec)) {
            if (has_before_ && has_base_ && frame_bgr.size() == before_.size()) {
                out.id = next_pair_id_++;
                out.base = base_;                 // 浅拷贝，cv::Mat 自带引用计数
                out.before = before_;
                out.after = frame_bgr;
                out.before_ms = before_ms_;
                out.after_ms = now_ms;

                phase_ = ScenePhase::CLEAR;
                pair_count_++;
                last_pair_ms_ = now_ms;
                has_before_ = false;              // 一次入场只结一次账
                last_clear_ = frame_bgr.clone();
                last_clear_ms_ = now_ms;
                return true;
            }
            // 没有可比对的起点（开机就有人）→ 直接回到清场，不产生会话
            phase_ = ScenePhase::CLEAR;
            has_before_ = false;
            last_clear_ = frame_bgr.clone();
            last_clear_ms_ = now_ms;
            return false;
        }
        return false;
    }

    // 仍然有人
    clear_since_ms_ = 0;
    last_clear_.release();

    // 超时保护：有人久站（看手机 / 睡着）永远等不到清场。
    // 骨架阶段选择"放弃本次会话"而不是强制比对 —— 画面里有人时强行比对，
    // 那个人本身就会变成一个巨大的"新增块"。等真正清场后重新记起点。
    if ((now_ms - occupied_since_ms_) >= sec2ms(cfg_.max_occupied_sec)) {
        has_before_ = false;
        occupied_since_ms_ = now_ms;
    }
    return false;
}

// ==================== 结算一次会话 ====================
PairResult ChangeSession::resolve(const PairRequest& req, const cv::Mat& change_mask,
                                  int64_t now_ms) {
    PairResult res;
    res.pair_id = req.id;

    if (change_mask.empty() || req.before.empty() || req.after.empty() || req.base.empty()) {
        res.discarded = true;
        res.reason = "empty-input";
        return res;
    }
    if (change_mask.size() != req.after.size()) {
        res.discarded = true;
        res.reason = "size-mismatch";
        return res;
    }

    // 光照 / 抖动守卫：宁可漏一次，也不要拿两张不可比的图比出满屏候选
    std::string why;
    if (discardIfUnstable(req.before, req.after, why)) {
        res.discarded = true;
        res.reason = why;
        allow_base_update_ = false;   // 场景不可信，基线也不要动
        return res;
    }

    extractBlobs(change_mask, res.blobs);

    // ★ 顺序很重要：先纯几何配对，再给"剩下的孤立块"判方向。
    //   物体被移动时会同时产生"这块没了 + 那块多了"，两者在差分里是对称的，
    //   若先判号再配对，两块会拿到同一个符号而永远配不上。
    pairBlobs(res.blobs);

    std::vector<ChangeBlob> keep;
    keep.reserve(res.blobs.size());
    for (auto& b : res.blobs) {
        if (b.paired) { keep.push_back(b); continue; }   // 移动，保留记录但不判号
        b.sign = decideSign(req.before, req.after, req.base, b.box);
        if (b.sign != 0) keep.push_back(b);
    }
    res.blobs.swap(keep);

    updateResiduals(res.blobs, req.after, req.base, now_ms, res);

    // 基线能否吸收：本轮没有"新增的未配对块"，也没有新登记残留 → 场景是干净的
    bool dirty = false;
    for (const auto& b : res.blobs)
        if (b.sign > 0 && !b.paired) { dirty = true; break; }
    if (!res.new_residuals.empty()) dirty = true;
    allow_base_update_ = !dirty;

    return res;
}

// ==================== 会话作废判定 ====================
bool ChangeSession::discardIfUnstable(const cv::Mat& before, const cv::Mat& after,
                                      std::string& why) const {
    if (before.size() != after.size()) { why = "before-after-size"; return true; }

    cv::Mat d;
    cv::absdiff(before, after, d);
    const double mad = cv::mean(d)[0];
    if (mad > cfg_.max_mean_abs_diff) {
        why = "global-shift:" + std::to_string((int)(mad + 0.5));
        return true;
    }

    // 变化像素占比：整屏都在变 → 光照/相机突变，不是物体
    cv::Mat g;
    cv::cvtColor(d, g, cv::COLOR_BGR2GRAY);
    const int changed = cv::countNonZero(g > 30);
    const double ratio = (double)changed / (double)(g.cols * g.rows);
    if (ratio > cfg_.max_changed_ratio) {
        why = "changed-ratio:" + std::to_string((int)(ratio * 100)) + "%";
        return true;
    }
    return false;
}

// ==================== 连通域提块 ====================
void ChangeSession::extractBlobs(const cv::Mat& mask, std::vector<ChangeBlob>& out) const {
    out.clear();

    cv::Mat bin;
    if (mask.type() != CV_8UC1) cv::cvtColor(mask, bin, cv::COLOR_BGR2GRAY);
    else bin = mask;
    cv::Mat bin2 = (bin > 0);   // 归一成 0/1，耐得住调用方给 0/255 或 0/1

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(bin2, labels, stats, centroids, 8, CV_32S);

    int idx = 0;
    for (int i = 1; i < n; ++i) {   // 0 = 背景
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (area < cfg_.min_area_px) continue;
        if (w < cfg_.min_side_px || h < cfg_.min_side_px) continue;

        ChangeBlob b;
        b.id = idx++;
        b.box = cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                         w, h);
        b.centroid = cv::Point2f((float)centroids.at<double>(i, 0),
                                 (float)centroids.at<double>(i, 1));
        b.area = area;
        out.push_back(b);
    }
}

// ==================== 方向判定 ====================
int ChangeSession::decideSign(const cv::Mat& before, const cv::Mat& after,
                              const cv::Mat& base, const cv::Rect& box) const {
    // 与长期基线比：after 离基线更远 = 这地方多了东西；before 离基线更远 = 少了东西
    const double mad_after  = meanAbsDiffInBox(after,  base, box);
    const double mad_before = meanAbsDiffInBox(before, base, box);
    if (mad_after - mad_before > cfg_.sign_mad_margin) return +1;
    if (mad_before - mad_after > cfg_.sign_mad_margin) return -1;
    return 0;
}

// ==================== 正负配对 ====================
void ChangeSession::pairBlobs(std::vector<ChangeBlob>& blobs) const {
    // 纯几何配对，不看符号：两块面积相近、质心相近 → 判为"同一个物体换了位置"。
    // 贪心 + 就近原则：按质心距离升序尝试，配过的块不再参与。
    const double max_d2 = cfg_.pair_dist_px * cfg_.pair_dist_px;

    struct Cand { int a, b; double d2; };
    std::vector<Cand> cands;
    for (size_t i = 0; i < blobs.size(); ++i) {
        for (size_t j = i + 1; j < blobs.size(); ++j) {
            const double d2 = dist2(blobs[i].centroid, blobs[j].centroid);
            if (d2 > max_d2) continue;
            const double lo = std::min(blobs[i].area, blobs[j].area);
            const double hi = std::max(blobs[i].area, blobs[j].area);
            if (hi <= 0 || lo / hi < cfg_.pair_area_ratio) continue;
            cands.push_back({(int)i, (int)j, d2});
        }
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand& x, const Cand& y) { return x.d2 < y.d2; });

    for (const auto& c : cands) {
        if (blobs[c.a].paired || blobs[c.b].paired) continue;
        blobs[c.a].paired = true;
        blobs[c.a].paired_with = blobs[c.b].id;
        blobs[c.b].paired = true;
        blobs[c.b].paired_with = blobs[c.a].id;
    }
}

// ==================== 残留表 ====================
void ChangeSession::updateResiduals(const std::vector<ChangeBlob>& blobs,
                                    const cv::Mat& after, const cv::Mat& base,
                                    int64_t now_ms, PairResult& res) {
    const double max_d2 = cfg_.pair_dist_px * cfg_.pair_dist_px;
    std::vector<bool> touched(residuals_.size(), false);

    // 先处理"消失"（被取走）：反过来把对应残留物删掉
    for (const auto& b : blobs) {
        if (b.sign >= 0 || b.paired) continue;
        int best = -1;
        double best_d2 = max_d2;
        for (size_t i = 0; i < residuals_.size(); ++i) {
            if (touched[i]) continue;
            const double d2 = dist2(b.centroid, residuals_[i].centroid);
            if (d2 < best_d2) { best_d2 = d2; best = (int)i; }
        }
        if (best >= 0) {
            touched[best] = true;
            res.cleared.push_back(residuals_[best].id);
            residuals_.erase(residuals_.begin() + best);
            touched.erase(touched.begin() + best);
        }
    }

    // 再处理"新增"：能对上残留表 → 累加轮数；对不上 → 新登记
    for (const auto& b : blobs) {
        if (b.sign <= 0 || b.paired) continue;

        int best = -1;
        double best_d2 = max_d2;
        for (size_t i = 0; i < residuals_.size(); ++i) {
            if (touched[i]) continue;
            const double d2 = dist2(b.centroid, residuals_[i].centroid);
            if (d2 < best_d2) { best_d2 = d2; best = (int)i; }
        }

        if (best >= 0) {
            ResidualItem& r = residuals_[best];
            touched[best] = true;
            r.box = b.box;
            r.centroid = b.centroid;
            r.last_seen_ms = now_ms;
            r.missed = 0;
            r.rounds++;
            if (!r.confirmed && r.rounds >= cfg_.confirm_rounds) {
                r.confirmed = true;
                res.confirmed.push_back(r.id);
            }
        } else {
            ResidualItem r;
            r.id = next_residual_id_++;
            r.box = b.box;
            r.centroid = b.centroid;
            r.first_seen_ms = now_ms;
            r.last_seen_ms = now_ms;
            r.rounds = 1;
            r.missed = 0;
            r.confirmed = (cfg_.confirm_rounds <= 1);
            residuals_.push_back(r);
            res.new_residuals.push_back(r.id);
            if (r.confirmed) res.confirmed.push_back(r.id);
            touched.push_back(true);
        }
    }

    // 没被任何变化块匹配到的残留物：先回头确认它还在不在。
    // ★ 这一步不能省。遗留物一直不动时，本轮的 before/after 差分是空的（两边都有它），
    //   若直接按"没匹配到"处理，两轮之后它就会被当成已被取走删掉，
    //   然后在基线更新里被吸收 —— 变成永久漏报。
    for (size_t i = residuals_.size(); i-- > 0;) {
        if (i < touched.size() && touched[i]) continue;

        const double mad = meanAbsDiffInBox(after, base, residuals_[i].box);
        if (mad > cfg_.sign_mad_margin) {
            residuals_[i].missed = 0;
            residuals_[i].last_seen_ms = now_ms;
            residuals_[i].rounds++;
            if (!residuals_[i].confirmed && residuals_[i].rounds >= cfg_.confirm_rounds) {
                residuals_[i].confirmed = true;
                res.confirmed.push_back(residuals_[i].id);
            }
            continue;
        }

        residuals_[i].missed++;
        if (residuals_[i].missed > cfg_.keep_missed_rounds) {
            res.cleared.push_back(residuals_[i].id);
            residuals_.erase(residuals_.begin() + i);
        }
    }
}

// ==================== 查询 ====================
int ChangeSession::confirmedCount() const {
    int n = 0;
    for (const auto& r : residuals_) if (r.confirmed) n++;
    return n;
}

const ResidualItem* ChangeSession::findResidual(int id) const {
    for (const auto& r : residuals_) if (r.id == id) return &r;
    return nullptr;
}

// ==================== 注入基线 ====================
void ChangeSession::seedBaseline(const cv::Mat& roi_bgr, int64_t now_ms) {
    if (roi_bgr.empty()) return;
    base_ = roi_bgr.clone();
    base_ms_ = now_ms;
    has_base_ = true;
    last_clear_ = roi_bgr.clone();
    last_clear_ms_ = now_ms;
    allow_base_update_ = true;
}

void ChangeSession::reset() {
    phase_ = ScenePhase::CLEAR;
    clear_since_ms_ = 0;
    occupied_since_ms_ = 0;
    has_base_ = false;
    base_ms_ = 0;
    base_.release();
    has_before_ = false;
    before_ms_ = 0;
    before_.release();
    last_clear_.release();
    last_clear_ms_ = 0;
    allow_base_update_ = true;
    residuals_.clear();
    next_residual_id_ = 1;
    // 注意：pair_count_ / last_pair_ms_ / next_pair_id_ 不清，
    // 它们是"这一路相机跑过多少次会话"的统计，reset 不该把统计抹掉。
}
