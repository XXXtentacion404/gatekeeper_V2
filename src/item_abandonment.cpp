// 物品滞留检测 — 清场双时相版本的相机侧外观
// 状态机在 change_session.cpp，本文件负责：ROI 裁剪、判"有没有人"、
// 产出变化掩膜、把残留表同步成 ItemTrack / ItemEvent。
//
// 旧的双 MOG2 + 参考差分实现已整体删除，完整版本见 commit c67cd6a。

#include "item_abandonment.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

}  // namespace

// ==================== 配置加载 ====================
AbandonConfig loadAbandonConfig(const std::string& path, int role) {
    AbandonConfig c;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[ITEM] 配置文件不存在: " << path << " → 全部使用内置默认值\n";
        return c;
    }

    // key=value；支持 "roleN." 前缀按相机角色覆盖。两遍扫描：
    // 先应用无前缀的公共值，再用 role{role}. 覆盖，保证覆盖优先生效。
    auto apply = [&c](const std::string& k, const std::string& v) {
        try {
            if      (k == "enable")              c.enable = std::stoi(v) != 0;
            else if (k == "use_roi_crop")        c.use_roi_crop = std::stoi(v);
            else if (k == "roi_margin_px")       c.roi_margin_px = std::stoi(v);
            else if (k == "work_max_w")          c.work_max_w = std::stoi(v);
            else if (k == "work_w")              c.work_w = std::stoi(v);
            else if (k == "work_h")              c.work_h = std::stoi(v);
            else if (k == "region_width_cm")     c.region_width_cm = std::stod(v);
            else if (k == "t_clear_sec")         c.t_clear_sec = std::stod(v);
            else if (k == "t_settle_sec")        c.t_settle_sec = std::stod(v);
            else if (k == "max_occupied_sec")    c.max_occupied_sec = std::stod(v);
            else if (k == "motion_thresh")       c.motion_thresh = std::stod(v);
            else if (k == "change_thresh")       c.change_thresh = std::stod(v);
            else if (k == "morph_open_px")       c.morph_open_px = std::stoi(v);
            else if (k == "morph_close_px")      c.morph_close_px = std::stoi(v);
            else if (k == "max_mean_abs_diff")   c.max_mean_abs_diff = std::stod(v);
            else if (k == "max_changed_ratio")   c.max_changed_ratio = std::stod(v);
            else if (k == "min_item_px")         c.min_item_px = std::stoi(v);
            else if (k == "min_item_area")       c.min_item_area = std::stod(v);
            else if (k == "max_area_ratio")      c.max_area_ratio = std::stod(v);
            else if (k == "pair_dist_px")        c.pair_dist_px = std::stod(v);
            else if (k == "pair_area_ratio")     c.pair_area_ratio = std::stod(v);
            else if (k == "confirm_rounds")      c.confirm_rounds = std::stoi(v);
            else if (k == "keep_missed_rounds")  c.keep_missed_rounds = std::stoi(v);
            else if (k == "remind_sec")          c.remind_sec = std::stod(v);
            else if (k == "push_sec")            c.push_sec = std::stod(v);
            else if (k == "use_network")         c.use_network = std::stoi(v);
            else if (k == "model_path")          c.model_path = v;
            else if (k == "save_dir")            c.save_dir = v;
            else if (k == "backtrack_sec")       c.backtrack_sec = std::stod(v);
            else if (k == "seq_fps")             c.seq_fps = std::stoi(v);
            // 其余键（mode / ref_bg_path / bg_*_lr / safe_radius_px …）属旧版参数，
            // 新架构不再使用，静默忽略以免旧配置文件报错。
        } catch (...) {
            std::cerr << "[ITEM] 配置项解析失败: " << k << "=" << v << "\n";
        }
    };

    struct KV { std::string k, v; };
    std::vector<KV> common, override_;
    const std::string prefix = "role" + std::to_string(role) + ".";

    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trim(line.substr(0, eq));
        std::string v = trim(line.substr(eq + 1));
        if (k.rfind(prefix, 0) == 0) override_.push_back({k.substr(prefix.size()), v});
        else if (k.rfind("role", 0) == 0 && k.find('.') != std::string::npos) continue;  // 别的角色
        else common.push_back({k, v});
    }

    for (const auto& kv : common) apply(kv.k, kv.v);
    for (const auto& kv : override_) apply(kv.k, kv.v);

    std::cout << "[ITEM] 配置: enable=" << c.enable
              << " roi_crop=" << c.use_roi_crop
              << " t_clear=" << c.t_clear_sec << "s"
              << " t_settle=" << c.t_settle_sec << "s"
              << " change_thresh=" << c.change_thresh
              << " backend=" << (c.use_network ? "rknn" : "cv")
              << std::endl;
    return c;
}

// ==================== 构造 ====================
ItemAbandonmentDetector::ItemAbandonmentDetector(const AbandonConfig& cfg, std::string cam_id)
    : cfg_(cfg), cam_id_(std::move(cam_id)) {
    setConfig(cfg);
}

ItemAbandonmentDetector::~ItemAbandonmentDetector() = default;

void ItemAbandonmentDetector::setConfig(const AbandonConfig& c) {
    cfg_ = c;

    // 把外观层的参数翻译成状态机参数。两处都有尺寸门限，
    // 这里以配置里的 min_item_area / min_item_px 为准，避免两边不一致。
    ChangeSessionConfig sc;
    sc.t_clear_sec        = c.t_clear_sec;
    sc.t_settle_sec       = c.t_settle_sec;
    sc.max_occupied_sec   = c.max_occupied_sec;
    sc.max_mean_abs_diff  = c.max_mean_abs_diff;
    sc.max_changed_ratio  = c.max_changed_ratio;
    sc.min_area_px        = (int)std::lround(c.min_item_area);
    sc.min_side_px        = c.min_item_px;
    sc.pair_dist_px       = c.pair_dist_px;
    sc.pair_area_ratio    = c.pair_area_ratio;
    sc.confirm_rounds     = c.confirm_rounds;
    sc.keep_missed_rounds = c.keep_missed_rounds;
    session_.setConfig(sc);

    if (c.enable) {
        std::cout << "[ITEM] " << cam_id_ << " 清场双时相: T_clear=" << c.t_clear_sec
                  << "s T_settle=" << c.t_settle_sec << "s 确认=" << c.confirm_rounds
                  << "轮 差分阈值=" << c.change_thresh
                  << " 后端=" << (c.use_network ? "rknn(未接入→退化为cv)" : "cv")
                  << std::endl;
    }
}

// ==================== ROI ====================
void ItemAbandonmentDetector::prepareRoi(const cv::Mat& region_mask) {
    roi_ready_ = true;
    if (!cfg_.use_roi_crop || region_mask.empty()) {
        roi_rect_ = cv::Rect(0, 0, full_w_, full_h_);
        std::cout << "[ITEM] " << cam_id_ << " 检测区裁剪=关闭 → 处理整帧 "
                  << full_w_ << "x" << full_h_ << std::endl;
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

    std::cout << "[ITEM] " << cam_id_ << " 检测区 ROI: " << bb.width << "x" << bb.height
              << " @(" << bb.x << "," << bb.y << ")  占整帧 "
              << (100.0 * bb.area() / ((double)full_w_ * full_h_)) << "%"
              << " → 小目标像素数不变（不降采样）" << std::endl;
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

// ==================== 标定自检 ====================
void ItemAbandonmentDetector::printCalibration() const {
    if (roi_rect_.width <= 0 || roi_bgr_.empty()) return;
    std::cout << "[ITEM] " << cam_id_ << " 标定自检: 处理图 " << roi_bgr_.cols << "x"
              << roi_bgr_.rows << "px (缩放 " << (1.0 / inv_sx_) << "x)" << std::endl;
    if (cfg_.region_width_cm <= 0) {
        std::cout << "[ITEM]   未填 region_width_cm → 无法自动校核 8cm。"
                     "现场做法：A4 纸横放(29.7cm)量像素宽，px_per_cm = 像素宽/29.7。\n";
        return;
    }
    const double px_per_cm = roi_bgr_.cols / cfg_.region_width_cm;
    const double px8 = 8.0 * px_per_cm;
    std::cout << "[ITEM]   检测区宽 " << cfg_.region_width_cm << "cm → px_per_cm="
              << px_per_cm << " → 8cm ≈ " << px8 << "px;  min_item_px=" << cfg_.min_item_px
              << " / min_item_area=" << (int)cfg_.min_item_area;
    if (px8 < 15.0)
        std::cout << "  ⚠️ 8cm 仅 " << px8 << "px (<15px)：纯 CV 可靠检测困难，"
                     "需提高分辨率或缩小视野";
    std::cout << std::endl;
}

// ==================== 判"有没有人" ====================
bool ItemAbandonmentDetector::personPresent(const std::vector<TrackedObject>& persons,
                                            const cv::Mat& roi_bgr) {
    // ① 有 YOLO 人框中心落在检测区（ROI）内
    bool by_box = false;
    for (const auto& p : persons) {
        const cv::Point2f c(p.box.x + p.box.width * 0.5f, p.box.y + p.box.height * 0.5f);
        if (roi_rect_.contains(cv::Point((int)c.x, (int)c.y))) { by_box = true; break; }
    }

    // ② ROI 内帧间运动能量。必须保留：YOLO 漏掉一个蹲着的人时，
    //    只看框数会误判"清场了"，然后那个人本身就被报成遗留物。
    bool by_motion = false;
    cv::Mat g;
    cv::cvtColor(roi_bgr, g, cv::COLOR_BGR2GRAY);
    if (!prev_gray_.empty() && prev_gray_.size() == g.size()) {
        cv::Mat d;
        cv::absdiff(prev_gray_, g, d);
        by_motion = cv::mean(d)[0] > cfg_.motion_thresh;
    }
    prev_gray_ = g;

    return by_box || by_motion;
}

// ==================== 变化掩膜 ====================
bool ItemAbandonmentDetector::buildChangeMask(const PairRequest& req, cv::Mat& mask,
                                              std::string& backend_note) {
    if (req.before.empty() || req.after.empty()) return false;

    // —— 后端 1：CV 骨架（absdiff + 阈值 + 形态学）——
    // 这里将来整体替换成 Fast BSUV-Net 2.0（9 通道 → 1 通道掩膜），
    // 输入输出约定不变：before/after 同尺寸 RGB，输出同尺寸 0/255。
    cv::cvtColor(req.before, gray_before_, cv::COLOR_BGR2GRAY);
    cv::cvtColor(req.after, gray_after_, cv::COLOR_BGR2GRAY);
    cv::absdiff(gray_before_, gray_after_, diff_buf_);
    cv::threshold(diff_buf_, mask, cfg_.change_thresh, 255, cv::THRESH_BINARY);

    if (cfg_.morph_open_px > 1) {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(cfg_.morph_open_px, cfg_.morph_open_px));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k);
    }
    if (cfg_.morph_close_px > 1) {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(cfg_.morph_close_px, cfg_.morph_close_px));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, k);
    }

    backend_note = cfg_.use_network ? "network-not-ready(fallback:cv)" : "cv";
    return true;
}

// ==================== 残留表 → ItemTrack ====================
void ItemAbandonmentDetector::syncTracks(const PairResult& res, int64_t now_ms,
                                         std::vector<ItemEvent>& ev) {
    const auto& rs = session_.residuals();

    // ① 会话已经清掉的（被取走 / 连续缺席）→ 发 CLEAR
    for (int cid : res.cleared) {
        for (size_t i = 0; i < items_.size(); ++i) {
            if (items_[i].id != cid) continue;
            ItemEvent e;
            e.item_id = cid;
            e.type = ItemEventType::CLEAR;
            e.box = items_[i].box;
            e.static_sec = (now_ms - items_[i].first_seen_ms) / 1000.0;
            e.note = "taken-away-or-gone";
            e.ts_ms = now_ms;
            ev.push_back(e);
            items_.erase(items_.begin() + i);
            break;
        }
    }

    // ② 残留表为准：有则更新，无则新建
    for (const auto& r : rs) {
        ItemTrack* hit = nullptr;
        for (auto& it : items_) if (it.id == r.id) { hit = &it; break; }

        if (!hit) {
            ItemTrack t;
            t.id = r.id;
            t.first_seen_ms = r.first_seen_ms;
            t.static_since_ms = r.first_seen_ms;
            t.unattended_since_ms = r.first_seen_ms;
            items_.push_back(t);
            hit = &items_.back();
        }
        hit->box = toFull(r.box);
        hit->centroid = toFull(r.centroid);
        hit->last_seen_ms = r.last_seen_ms;
        hit->rounds = r.rounds;
        hit->confirmed = r.confirmed;
        hit->missing_frames = 0;
        if (!r.confirmed) hit->state = ItemState::STATIC_CANDIDATE;
    }
}

// ==================== 状态推进 / 两级告警 ====================
void ItemAbandonmentDetector::advanceState(int64_t now_ms, std::vector<ItemEvent>& ev) {
    for (auto& it : items_) {
        const double age_sec = (now_ms - it.first_seen_ms) / 1000.0;

        ItemEventType want = ItemEventType::REMIND;
        bool fire = false;

        if (it.confirmed && age_sec >= cfg_.push_sec) {
            it.state = ItemState::LEVEL2_PUSH;
            if (!it.emitted_push) { it.emitted_push = true; fire = true; want = ItemEventType::PUSH; }
        } else if (it.confirmed && age_sec >= cfg_.remind_sec) {
            it.state = ItemState::LEVEL1_REMIND;
            if (!it.emitted_remind) { it.emitted_remind = true; fire = true; want = ItemEventType::REMIND; }
        } else {
            it.state = ItemState::STATIC_CANDIDATE;
        }

        if (!fire) continue;
        ItemEvent e;
        e.item_id = it.id;
        e.type = want;
        e.box = it.box;
        e.static_sec = age_sec;
        e.note = it.confirmed ? "confirmed" : "unconfirmed";
        e.ts_ms = now_ms;
        ev.push_back(e);
    }
}

// ==================== 每帧入口 ====================
std::vector<ItemEvent> ItemAbandonmentDetector::update(const cv::Mat& full_frame,
                                                       const std::vector<TrackedObject>& person_tracks,
                                                       const cv::Mat& region_mask,
                                                       int64_t capture_ms,
                                                       int result_frame_id) {
    std::vector<ItemEvent> ev;
    if (!cfg_.enable || full_frame.empty()) return ev;
    if (result_frame_id >= 0 && result_frame_id == last_rid_) return ev;   // 同一帧只处理一次
    last_rid_ = result_frame_id;

    full_w_ = full_frame.cols;
    full_h_ = full_frame.rows;
    if (!roi_ready_) prepareRoi(region_mask);

    const cv::Rect rr = roi_rect_ & cv::Rect(0, 0, full_w_, full_h_);
    if (rr.width <= 0 || rr.height <= 0) return ev;

    // —— 取 ROI，非必要不降采样（8cm 小目标的命根子）——
    cv::Mat roi = full_frame(rr);
    double s = 1.0;
    if (cfg_.work_max_w > 0 && roi.cols > cfg_.work_max_w)
        s = (double)cfg_.work_max_w / roi.cols;
    if (s < 0.999) {
        cv::resize(roi, roi_bgr_, cv::Size(std::max(1, (int)std::lround(roi.cols * s)),
                                           std::max(1, (int)std::lround(roi.rows * s))),
                   0, 0, cv::INTER_AREA);
        inv_sx_ = (double)rr.width / roi_bgr_.cols;
        inv_sy_ = (double)rr.height / roi_bgr_.rows;
    } else {
        roi_bgr_ = roi;
        inv_sx_ = inv_sy_ = 1.0;
    }

    if (!calib_printed_) { printCalibration(); calib_printed_ = true; }

    // —— 状态机推进 ——
    const bool person = personPresent(person_tracks, roi_bgr_);

    PairRequest req;
    if (session_.pushFrame(roi_bgr_, person, capture_ms, req)) {
        cv::Mat mask;
        std::string backend;
        if (buildChangeMask(req, mask, backend)) {
            PairResult res = session_.resolve(req, mask, capture_ms);
            if (res.discarded) {
                std::cerr << "[ITEM] " << cam_id_ << " 会话#" << res.pair_id
                          << " 作废: " << res.reason << "\n";
            } else {
                int np = 0, nm = 0;
                for (const auto& b : res.blobs) {
                    if (b.paired) ++nm;
                    else if (b.sign > 0) ++np;
                }
                std::cout << "[ITEM] " << cam_id_ << " 会话#" << res.pair_id
                          << "(" << backend << ") 新增=" << np << " 移动=" << nm
                          << " 清除=" << res.cleared.size()
                          << " 确认=" << res.confirmed.size()
                          << " 残留总数=" << session_.residuals().size() << std::endl;
                syncTracks(res, capture_ms, ev);
            }
        } else {
            std::cerr << "[ITEM] " << cam_id_ << " 变化掩膜生成失败，本次会话跳过\n";
        }
    }

    advanceState(capture_ms, ev);
    return ev;
}

// ==================== 注入基线 ====================
bool ItemAbandonmentDetector::setRefBackground(const cv::Mat& full_bgr) {
    if (full_bgr.empty()) return false;
    if (full_w_ <= 0) {
        full_w_ = full_bgr.cols;
        full_h_ = full_bgr.rows;
        if (!roi_ready_) prepareRoi(cv::Mat());
    }
    const cv::Rect rr = roi_rect_ & cv::Rect(0, 0, full_bgr.cols, full_bgr.rows);
    if (rr.width <= 0 || rr.height <= 0) return false;

    cv::Mat roi = full_bgr(rr);
    if (cfg_.work_max_w > 0 && roi.cols > cfg_.work_max_w) {
        const double s = (double)cfg_.work_max_w / roi.cols;
        cv::Mat small;
        cv::resize(roi, small, cv::Size(std::max(1, (int)std::lround(roi.cols * s)),
                                        std::max(1, (int)std::lround(roi.rows * s))),
                   0, 0, cv::INTER_AREA);
        roi = small;
    }
    roi_bgr_ = roi.clone();
    session_.seedBaseline(roi_bgr_, 0);
    std::cout << "[ITEM] " << cam_id_ << " 已注入初始基线 (" << roi_bgr_.cols << "x"
              << roi_bgr_.rows << ")" << std::endl;
    return true;
}

// ==================== 统计 ====================
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

int ItemAbandonmentDetector::alarmCount() const { return level1Count() + level2Count(); }
