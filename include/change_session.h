#ifndef GATE_CHANGE_SESSION_H
#define GATE_CHANGE_SESSION_H

// ============================================================================
// 清场双时相会话（Clear-Scene Dual-Temporal Session）— 物品滞留检测的核心骨架
// ----------------------------------------------------------------------------
// 解决的问题：纯背景差分拿一张"几周前拍的空景图"当基线，光照一变就整屏误报，
// 场景里新放个常驻物就永久误报。本模块把基线换成"上一次清场时的真实画面"，
// 时间跨度从"几天"缩到"几分钟"，光照漂移、摆位变动全部自动跟上。
//
// 三层时间（本模块维护，正好凑成网络的 9 通道输入）：
//   base_   : 长期基线 —— 这场地本来长什么样。只在"清场且上一轮无新增"时更新，
//             绝不自动吸收来历不明的静止块（否则人一走开包就被吸进基线，永久漏报）。
//   before_ : 入场前帧 —— 本次有人进入前、最后一次清场时的画面。
//   after_  : 离场后帧 —— 本次人走光、稳定 t_settle 之后的画面。
//
// ★ 立论支点：清场 = 画面里没有人的手，此刻检测到的静止块只可能是"放在那儿"的。
//   这一步从物理上排除了"拎着 vs 放下"的歧义，不需要时序模型也不需要人框绑定。
//
// 两阶段接口（推理后端可换）：
//   ① pushFrame() 每帧喂入 → 返回 true 表示"凑齐了一次会话"，带出三帧
//   ② resolve()   把变化掩膜交回来 → 出候选块、正负配对、残留表增删
//   纯 CV 骨架（absdiff+阈值）和以后换成 RKNN 网络，都只换 ① 与 ② 之间的那一步。
//
// 详见 文档/FastBSUV-Net2.0_清场双时相比对方案.md
// ============================================================================

#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// ==================== 场景相位 ====================
enum class ScenePhase {
    CLEAR    = 0,   // 检测区内无人、无运动
    OCCUPIED = 1,   // 检测区内有人
};

inline const char* scenePhaseName(ScenePhase p) {
    return p == ScenePhase::CLEAR ? "CLEAR" : "OCCUPIED";
}

// ==================== 配置 ====================
struct ChangeSessionConfig {
    // —— 时序（秒）——
    double t_clear_sec      = 1.5;   // 连续无人多久才算"确认清场"（防抖）
    double t_settle_sec     = 2.0;   // 人走后等多久才取 after 帧（残影/曝光回调）
    double max_occupied_sec = 300.0; // 有人滞留超此时长 → 强制结一次会话（防永远等不到清场）

    // —— 会话作废（宁可漏一次，也不要拿两张不可比的图比出满屏候选）——
    double max_mean_abs_diff   = 18.0;  // before/after 全局平均绝对差(灰度级) 超此值 → 作废
    double max_changed_ratio   = 0.25;  // 变化像素占比超此值 → 作废（光照/相机突变）

    // —— 尺寸过滤（全分辨率口径，与 min_item_px / min_item_area 同量纲）——
    int min_area_px = 300;   // 8cm 物体 ≈ 18px 见方 ≈ 324px²，别设太大
    int min_side_px = 15;    // 最小边长

    // —— 正负号判定 ——
    double sign_mad_margin = 8.0;  // MAD 差超过此值才判方向（否则视为同级噪声）

    // —— 正负配对（新增块 + 消失块成对 = 物体被移动，不是遗留）——
    double pair_dist_px    = 120.0; // 两块质心距离上限（ROI 宽约 1100px，60 偏小）
    double pair_area_ratio = 0.5;  // 面积比下限（小/大，1.0=要求等大）

    // —— 残留表 ——
    int confirm_rounds = 2;   // 连续 N 轮清场都稳定存在 → 确认（跨会话确认）
    int keep_missed_rounds = 2; // 连续 N 轮没匹配上才认为残留物消失
};

// ==================== 候选变化块 ====================
struct ChangeBlob {
    int         id = -1;        // 本轮内编号
    cv::Rect    box;            // 全分辨率坐标
    cv::Point2f centroid{0.f, 0.f};  // 全分辨率坐标
    int         area = 0;       // 全分辨率 px²
    int         sign = 0;       // +1 = 新增（以前没有）/ -1 = 消失（本来有）
    bool        paired = false; // 已与反号块配对 → 判为"移动"，不是遗留
    int         paired_with = -1;
};

// ==================== 残留表条目 ====================
// 语义：这块东西在场景里已经存在、且我们知道它的来历（上一轮判出来的）。
// 基线更新时要跳过它 —— 否则它会被吸进基线，之后就永远检不出来了。
struct ResidualItem {
    int         id = -1;
    cv::Rect    box;
    cv::Point2f centroid{0.f, 0.f};
    int64_t     first_seen_ms = 0;
    int64_t     last_seen_ms  = 0;
    int         rounds = 0;      // 连续被观测到的会话轮数
    int         missed = 0;      // 连续未匹配轮数
    bool        confirmed = false;  // rounds >= confirm_rounds
};

// ==================== 一次比对的请求 ====================
// 三帧都是 ROI 裁剪后的彩色图（不降采样），尺寸一致。
struct PairRequest {
    int64_t id = 0;          // 会话序号（回调时带回来，防止异步错配）
    cv::Mat base;            // 长期基线 BGR
    cv::Mat before;          // 入场前 BGR
    cv::Mat after;           // 离场后 BGR
    int64_t before_ms = 0;
    int64_t after_ms  = 0;
};

// ==================== 一次比对的结果 ====================
struct PairResult {
    int64_t     pair_id = 0;
    bool        discarded = false;   // 本次会话作废（光照/抖动/尺寸不符）
    std::string reason;              // 作废原因（人类可读，进日志）

    std::vector<ChangeBlob> blobs;          // 过滤 + 配对后的全部候选块
    std::vector<int>        new_residuals;  // 本轮新登记的残留物 id
    std::vector<int>        confirmed;      // 本轮跨会话确认的残留物 id（可报警）
    std::vector<int>        cleared;        // 本轮消失的残留物 id（被取走）
};

// ==================== 会话状态机 ====================
class ChangeSession {
public:
    ChangeSession() = default;
    explicit ChangeSession(const ChangeSessionConfig& cfg) : cfg_(cfg) {}

    void setConfig(const ChangeSessionConfig& c) { cfg_ = c; }
    const ChangeSessionConfig& config() const { return cfg_; }

    // 每帧调用一次。
    //   frame_bgr     : 当前帧的 ROI 彩色图（与 base/before/after 同尺寸）
    //   person_present: 检测区内是否有人（由调用方用 YOLO 框 + 前景能量合成）
    //   now_ms        : 采集时间戳
    //   out           : 凑齐一次会话时填充三帧，返回 true；否则返回 false
    // 返回 true 后，调用方必须尽快算完掩膜并调 resolve()，否则这三帧会过期。
    bool pushFrame(const cv::Mat& frame_bgr, bool person_present,
                   int64_t now_ms, PairRequest& out);

    // 把变化掩膜（uchar，0=没变 / 255=变了）交回来，出本轮结果。
    // 掩膜尺寸必须与 req.after 一致；不一致时按"作废"处理并返回。
    PairResult resolve(const PairRequest& req, const cv::Mat& change_mask, int64_t now_ms);

    // —— 查询 ——
    ScenePhase phase() const { return phase_; }
    bool  hasBaseline() const { return has_base_; }
    int64_t baseMs() const { return base_ms_; }
    int64_t lastPairMs() const { return last_pair_ms_; }
    int   pairCount() const { return pair_count_; }
    const std::vector<ResidualItem>& residuals() const { return residuals_; }

    // 已确认的残留物数量（供 OSD / 统计）
    int confirmedCount() const;
    // 取某条残留物（不存在返回 nullptr）
    const ResidualItem* findResidual(int id) const;

    // 直接注入基线（跳过开机后等第一次清场）。尺寸不符会在下一帧被 reset。
    void seedBaseline(const cv::Mat& roi_bgr, int64_t now_ms);

    // 清空全部状态（切场景 / 重连相机时用）
    void reset();

private:
    // 连续 components 提候选块（含尺寸过滤）
    void extractBlobs(const cv::Mat& mask, std::vector<ChangeBlob>& out) const;
    // 纯几何配对：面积相近 + 质心相近的两块 = 同一个物体换了位置（不看符号）
    void pairBlobs(std::vector<ChangeBlob>& blobs) const;
    // 判孤立块的方向：与长期基线的平均绝对差，after 更远 = 新增，before 更远 = 消失
    int  decideSign(const cv::Mat& before, const cv::Mat& after,
                    const cv::Mat& base, const cv::Rect& box) const;
    // 残留表增删改。after/base 用于"本轮没有变化块时回头确认残留物还在不在"
    void updateResiduals(const std::vector<ChangeBlob>& blobs,
                         const cv::Mat& after, const cv::Mat& base,
                         int64_t now_ms, PairResult& res);
    // 会话作废判定（光照 / 抖动）
    bool discardIfUnstable(const cv::Mat& before, const cv::Mat& after,
                           std::string& why) const;

    ChangeSessionConfig cfg_;

    ScenePhase phase_ = ScenePhase::CLEAR;
    int64_t    clear_since_ms_    = 0;   // 本轮 CLEAR 起点（0 = 未开始计）
    int64_t    occupied_since_ms_ = 0;
    int64_t    last_pair_ms_      = 0;
    int        pair_count_        = 0;
    int64_t    next_pair_id_      = 1;

    bool    has_base_   = false;
    int64_t base_ms_    = 0;
    cv::Mat base_;

    bool    has_before_ = false;
    int64_t before_ms_  = 0;
    cv::Mat before_;

    // 上一个清场帧（入场那一刻冻结成 before_）
    cv::Mat last_clear_;
    int64_t last_clear_ms_ = 0;

    // 上一次清场后是否"干净"（没有新增块）—— 只有干净才允许基线吸收
    bool allow_base_update_ = true;

    std::vector<ResidualItem> residuals_;
    int next_residual_id_ = 1;
};

#endif // GATE_CHANGE_SESSION_H
