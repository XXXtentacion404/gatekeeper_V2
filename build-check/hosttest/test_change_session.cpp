// ============================================================================
// 清场双时相会话 — 宿主侧回归测试（纯 OpenCV core/imgproc，不依赖 YOLO / RKNN）
// ----------------------------------------------------------------------------
// 用合成的"空场 / 有人 / 有遗留物 / 光照突变"帧序列驱动 ChangeSession，
// 覆盖 8 个场景。CV 后端（absdiff + 阈值）在这里由测试自己模拟 ——
// 真实工程里这一步将来换成 RKNN 网络，状态机部分一行不用改。
//
// 编译：见 run_host_tests.ps1 的 Invoke-Suite 'ChangeSession ...'
// ============================================================================

#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "change_session.h"

// ==================== 断言框架 ====================
static int g_pass = 0, g_fail = 0;
static std::string g_case;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (cond) { ++g_pass; }                                                 \
        else {                                                                  \
            ++g_fail;                                                           \
            std::printf("  [FAIL] %s :: %s\n", g_case.c_str(), (msg));          \
        }                                                                       \
    } while (0)

#define CASE(name) do { g_case = (name); std::printf("\n== %s\n", (name)); } while (0)

// ==================== 合成场景 ====================
static const int kW = 320, kH = 240;

static cv::Mat emptyScene() {
    return cv::Mat(kH, kW, CV_8UC3, cv::Scalar(110, 110, 110));
}

// 在空场上画若干个灰色实心块（模拟"放在地上的东西"）
static cv::Mat sceneWith(std::initializer_list<cv::Rect> objs, int val = 200) {
    cv::Mat f = emptyScene();
    for (const auto& r : objs) cv::rectangle(f, r, cv::Scalar(val, val, val), cv::FILLED);
    return f;
}

// 光照突变：整体抬亮
static cv::Mat brighten(const cv::Mat& f, int delta) {
    cv::Mat o;
    f.convertTo(o, -1, 1.0, delta);
    return o;
}

// ==================== 驱动器 ====================
struct Drv {
    ChangeSession s;
    int64_t t = 0;
    PairRequest req;
    int pairs = 0;

    explicit Drv(const ChangeSessionConfig& c = ChangeSessionConfig()) : s(c) {}

    // 喂一帧；凑齐会话就立刻用"absdiff + 阈值"算掩膜并结算
    bool step(const cv::Mat& frame, bool person, PairResult& res) {
        t += 100;   // 100ms/帧
        if (!s.pushFrame(frame, person, t, req)) return false;

        cv::Mat g1, g2, d, m;
        cv::cvtColor(req.before, g1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(req.after, g2, cv::COLOR_BGR2GRAY);
        cv::absdiff(g1, g2, d);
        cv::threshold(d, m, 25, 255, cv::THRESH_BINARY);

        res = s.resolve(req, m, t);
        ++pairs;
        return true;
    }
};

// 跑一次完整会话：清场 → 有人 → 人走光（用 after 画面）
// 返回 true 表示凑齐了一次会话
static bool runSession(Drv& d, const cv::Mat& idle, const cv::Mat& after,
                       PairResult& res, int personFrames = 5) {
    // 1) 清场（够久，让基线建立）
    for (int i = 0; i < 25; ++i) {
        PairResult tmp;
        if (d.step(idle, false, tmp)) { res = tmp; return true; }
    }
    // 2) 有人
    for (int i = 0; i < personFrames; ++i) {
        PairResult tmp;
        if (d.step(idle, true, tmp)) { res = tmp; return true; }
    }
    // 3) 人走光（t_settle 需要 20 帧，给 60 帧余量）
    for (int i = 0; i < 60; ++i) {
        PairResult tmp;
        if (d.step(after, false, tmp)) { res = tmp; return true; }
    }
    return false;
}

// ==================== 场景 ====================
int main() {
    const cv::Rect kBag(140, 120, 24, 24);        // 8cm 量级的小目标
    const cv::Rect kChairA(40, 40, 48, 48);
    const cv::Rect kChairB(114, 114, 48, 48);     // 与 A 的质心距 ≈ 74px（< pair_dist 120）

    // ---------------------------------------------------------------- T1
    {
        CASE("T1 空场 -> 有人 -> 走光，场景无变化：不应产生任何候选");
        Drv d;
        PairResult res;
        const bool fired = runSession(d, emptyScene(), emptyScene(), res);
        CHECK(fired, "应当凑齐一次会话");
        CHECK(res.blobs.empty(), "无变化时不应有候选块");
        CHECK(res.new_residuals.empty(), "不应登记残留物");
        CHECK(d.s.residuals().empty(), "残留表应为空");
        CHECK(!res.discarded, "不应被判为作废");
    }

    // ---------------------------------------------------------------- T2
    {
        CASE("T2 放下一个包：产生 1 个新增块，登记残留，第 1 轮不确认");
        Drv d;
        PairResult res;
        const bool fired = runSession(d, emptyScene(), sceneWith({kBag}), res);
        CHECK(fired, "应当凑齐一次会话");
        CHECK(res.blobs.size() == 1, "应恰好 1 个候选块");
        if (res.blobs.size() == 1) {
            CHECK(res.blobs[0].sign == +1, "方向应为新增(+1)");
            CHECK(!res.blobs[0].paired, "孤立块不应被配对");
        }
        CHECK(res.new_residuals.size() == 1, "应登记 1 个残留物");
        CHECK(res.confirmed.empty(), "第 1 轮不应确认（confirm_rounds=2）");
        CHECK(d.s.residuals().size() == 1, "残留表应有 1 条");
    }

    // ---------------------------------------------------------------- T3
    {
        CASE("T3 包一直不动：下一轮差分是空的，但残留物必须被确认而不是被删掉");
        Drv d;
        PairResult r1, r2;
        runSession(d, emptyScene(), sceneWith({kBag}), r1);
        const bool fired2 = runSession(d, sceneWith({kBag}), sceneWith({kBag}), r2);
        CHECK(fired2, "第二轮也应凑齐会话");
        CHECK(r2.blobs.empty(), "两边都有包 -> 差分为空 -> 无候选块");
        CHECK(d.s.residuals().size() == 1, "残留物不能因为本轮无变化就被删掉");
        CHECK(r2.confirmed.size() == 1, "第 2 轮应跨会话确认");
        CHECK(d.s.confirmedCount() == 1, "确认计数应为 1");
    }

    // ---------------------------------------------------------------- T4
    {
        CASE("T4 椅子从 A 挪到 B：两块纯几何配对被吃掉，不报警");
        Drv d;
        PairResult res;
        // 基线阶段椅子上就在 A（常驻物），所以 idle 用 chairA
        const bool fired = runSession(d, sceneWith({kChairA}), sceneWith({kChairB}), res);
        CHECK(fired, "应当凑齐一次会话");
        CHECK(res.blobs.size() == 2, "应检出 2 个变化块（A 消失 + B 出现）");
        int paired = 0;
        for (const auto& b : res.blobs) if (b.paired) ++paired;
        CHECK(paired == 2, "两块都应与对方配对");
        CHECK(res.new_residuals.empty(), "移动不是遗留，不应登记残留物");
        CHECK(res.confirmed.empty(), "不应确认任何残留物");
    }

    // ---------------------------------------------------------------- T5
    {
        CASE("T5 包被取走：判定为消失(-1)，从残留表清除，不报警");
        Drv d;
        PairResult r1, r2, r3;
        runSession(d, emptyScene(), sceneWith({kBag}), r1);          // 放下
        runSession(d, sceneWith({kBag}), sceneWith({kBag}), r2);     // 静止确认
        CHECK(d.s.residuals().size() == 1, "取走前应有 1 条残留");

        // 清场帧里还有包 -> 进入前，走光后包没了
        const bool fired = runSession(d, sceneWith({kBag}), emptyScene(), r3);
        CHECK(fired, "应当凑齐一次会话");
        CHECK(r3.blobs.size() == 1, "应检出 1 个变化块");
        if (r3.blobs.size() == 1) {
            CHECK(r3.blobs[0].sign == -1, "方向应为消失(-1)");
        }
        CHECK(r3.cleared.size() == 1, "应从残留表清除");
        CHECK(d.s.residuals().empty(), "残留表应被清空");
    }

    // ---------------------------------------------------------------- T6
    {
        CASE("T6 光照突变（整屏 +40）：本次会话作废，不产生候选");
        Drv d;
        PairResult res;
        const bool fired = runSession(d, emptyScene(), brighten(emptyScene(), 40), res);
        CHECK(fired, "仍应凑齐会话（作废是结算阶段的判断）");
        CHECK(res.discarded, "应被判为作废");
        CHECK(res.reason.find("global-shift") != std::string::npos, "作废原因应指向全局亮度漂移");
        CHECK(res.blobs.empty(), "作废时不应产生候选块");
    }

    // ---------------------------------------------------------------- T7
    {
        CASE("T7 有人久站超过 max_occupied：放弃本次会话，不产生候选");
        ChangeSessionConfig c;
        c.max_occupied_sec = 3.0;
        Drv d(c);
        PairResult res;

        // 先建立基线
        for (int i = 0; i < 25; ++i) { PairResult t; d.step(emptyScene(), false, t); }
        // 有人站 5 秒（50 帧），超过 3 秒上限
        for (int i = 0; i < 50; ++i) { PairResult t; d.step(emptyScene(), true, t); }
        // 再走光
        bool fired = false;
        for (int i = 0; i < 60; ++i) {
            PairResult t;
            if (d.step(sceneWith({kBag}), false, t)) { res = t; fired = true; break; }
        }
        CHECK(!fired, "超时后应放弃本次会话，等下一次入场重新记账");
    }

    // ---------------------------------------------------------------- T8
    {
        CASE("T8 残留保护：包留下后基线不得吸收它所在区域");
        Drv d;
        PairResult r1, r2;
        runSession(d, emptyScene(), sceneWith({kBag}), r1);
        // 再跑 5 轮"包一直不动"的会话，期间基线在持续更新
        for (int i = 0; i < 5; ++i)
            runSession(d, sceneWith({kBag}), sceneWith({kBag}), r2);

        // 若基线被污染，下面的"取走"会判成新增(+1)；没被污染才会判成消失(-1)
        PairResult r3;
        const bool fired = runSession(d, sceneWith({kBag}), emptyScene(), r3);
        CHECK(fired, "应当凑齐会话");
        CHECK(r3.blobs.size() == 1 && r3.blobs[0].sign == -1,
              "基线若吸收过包，这里会判成新增 —— 说明残留保护失效");
        CHECK(d.s.residuals().empty(), "取走后残留表应清空");
    }

    // ---------------------------------------------------------------- 汇总
    std::printf("\n----------------------------------------\n");
    std::printf("ChangeSession: %d passed, %d failed\n", g_pass, g_fail);
    std::printf("----------------------------------------\n");
    return g_fail == 0 ? 0 : 1;
}
