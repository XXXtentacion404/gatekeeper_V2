// 临时验证：检查新版 gate_id_counter 的两个新增机制
//   M1 = 交接跨线兑付 (handover-segment-enter/leave)
//   M2 = 假 outside 框守卫 (kMaxUnexplainedJumpPx)
#include "gate_id_counter.h"

#include <cstdio>
#include <vector>

static cv::Mat makeMask() {
    cv::Mat mask = cv::Mat::zeros(1080, 1920, CV_8UC1);
    std::vector<cv::Point> poly{{600, 300}, {1400, 300}, {1400, 900}, {600, 900}};
    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255));
    return mask;
}
static Tripwire line() { return Tripwire(600.f, 300.f, 600.f, 900.f); }
static TrackedObject mk(int id, float fx, float fy) {
    const int w = 100, h = 200;
    TrackedObject t;
    t.track_id = id; t.observed = true; t.confidence = 0.8f; t.hit_streak = 12;
    t.class_id = 0; t.class_name = "people";
    t.box = cv::Rect((int)(fx - w / 2.0f), (int)(fy - h * 0.92f), w, h);
    return t;
}
static const char* tn(int t) {
    return t == GATE_EVENT_IN ? "IN" : (t == GATE_EVENT_OUT ? "OUT" : "RETRO");
}

struct H {
    GateIdCounter c;
    int rid = 0, in = 0, out = 0, retro = 0, payoff = 0;
    H() : c(GATE_ROLE_ENTRY, line(), 100.0f, makeMask()) {}
    void run(int n, std::vector<TrackedObject> ts) {
        for (int i = 0; i < n; i++) {
            rid++;
            for (auto& x : ts) x.result_frame_id = rid;
            for (const auto& e : c.update(ts)) {
                if (e.type == GATE_EVENT_IN) in++;
                else if (e.type == GATE_EVENT_OUT) out++;
                else retro++;
                if (e.reason.rfind("handover-segment", 0) == 0) payoff++;
                std::printf("      f%3d tid=%-3d %-5s origin=%-3d settled=%d reason=%s\n",
                            rid, e.track_id, tn(e.type), e.origin_id, (int)e.settled,
                            e.reason.c_str());
            }
        }
    }
};

int main() {
    // ---- G. 首次跨线即换 ID（entered/exited 仍为 false）→ 兑付应立刻生效 ----
    std::printf("--- G 首次跨线换 ID（entered=false）---\n");
    {
        H h;
        h.run(3, {mk(80, 550, 600)});   // 线外，确立 side=0
        h.run(2, {});                    // 跟丢 2 帧（在 180px/30 帧窗口内）
        h.run(1, {mk(81, 660, 600)});   // 新 ID 首帧已在线内，连线 550→660 真穿过 t1
        std::printf("   => IN=%d RETRO=%d  兑付事件=%d  (期望 IN=1, 兑付=1)\n",
                    h.in, h.retro, h.payoff);
    }

    // ---- H. 同一几何，但 entered 已为 true（人已走过一趟）→ 兑付被 !entered 挡住 ----
    std::printf("--- H 第二趟跨线换 ID（entered=true）---\n");
    {
        H h;
        h.run(3, {mk(80, 550, 600)});   // 线外 side=0
        h.run(3, {mk(80, 660, 600)});   // 第 1 趟进 → IN=1, entered=true
        h.run(3, {mk(80, 550, 600)});   // 第 1 趟出 → RETRO=1, exited=true
        h.run(2, {});                    // 跟丢
        h.run(1, {mk(81, 660, 600)});   // 第 2 趟跨进，但换了 ID
        h.run(4, {mk(81, 700, 600)});   // 新 ID 继续在区内
        std::printf("   => IN=%d RETRO=%d  兑付事件=%d  (期望 IN=2, 兑付=1)\n",
                    h.in, h.retro, h.payoff);
    }

    // ---- B. 假 outside 框守卫：框跳到区域外、但连线并未穿过 t1 → 应被忽略 ----
    std::printf("--- B 假 outside 框守卫 ---\n");
    {
        H h;
        h.run(3, {mk(80, 550, 600)});
        h.run(3, {mk(80, 660, 600)});   // IN=1
        h.run(4, {mk(80, 800, 600)});   // 进到检测区深处
        h.run(1, {mk(80, 1500, 600)});  // 假框：跳到区域外，但仍在线内侧 → 应被忽略
        h.run(6, {mk(80, 550, 600)});   // 真正的跨出
        std::printf("   => IN=%d RETRO=%d  (期望 IN=1 RETRO=1)\n", h.in, h.retro);
    }

    // ---- C. 跟丢且【没有任何后继 ID】→ 现场 P12 情形，看是否仍漏 ----
    std::printf("--- C 跟丢且无后继 ID（现场 P12 情形）---\n");
    {
        H h;
        h.run(3, {mk(80, 550, 600)});
        h.run(3, {mk(80, 660, 600)});   // IN=1
        h.run(4, {mk(80, 640, 600)});   // 停在入口带内侧，然后消失
        h.run(200, {});                  // 再没有任何 ID 出现
        std::printf("   => IN=%d RETRO=%d  (期望 RETRO=1；若=0 则该漏计仍未修)\n",
                    h.in, h.retro);
    }
    return 0;
}
