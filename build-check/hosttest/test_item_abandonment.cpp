// ============================================================================
// 宿主单元测试：物品滞留检测状态机（ItemAbandonmentDetector）
// ----------------------------------------------------------------------------
// 不依赖真实视频：用合成帧（纯色背景 + 纯色块）驱动 MOG2 双背景，逐帧推进
// 虚拟时间（capture_ms），断言事件序列。覆盖 文档/物品滞留检测_模块设计.md §11 用例。
//
// 编译/运行（宿主 OpenCV 只 build 了 core/imgproc/video，故本文件自带
// imread/imwrite 桩来替代缺失的 imgcodecs）：
//   g++ -std=gnu++17 -O1 -w -I<opencv>/build/include -Iinclude -Iinclude/bytetrack \
//       test_item_abandonment.cpp src/item_abandonment.cpp \
//       <opencv-build>/lib/libopencv_video4100.a \
//       <opencv-build>/lib/libopencv_imgproc4100.a \
//       <opencv-build>/lib/libopencv_core4100.a \
//       <opencv-build>/3rdparty/lib/libzlib.a -lstdc++ -lm -lpthread -static
// ============================================================================

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// ---- imgcodecs 桩（必须在包含 opencv 头之后、使用之前提供定义）----
#include <opencv2/opencv.hpp>
namespace cv {
Mat imread(const String&, int) { return Mat(); }
bool imwrite(const String&, InputArray, const std::vector<int>&) { return true; }
}  // namespace cv

#include "item_abandonment.h"

// ==================== 测试框架 ====================
static int g_pass = 0, g_fail = 0;
static std::string g_case;

static void check(bool cond, const std::string& msg) {
    if (cond) { ++g_pass; }
    else { ++g_fail; std::cout << "      [FAIL] " << msg << "\n"; }
}

// ==================== 合成帧工具 ====================
static const int    kW = 640, kH = 360;          // 全分辨率
static const cv::Scalar kBg(200, 200, 200);      // 背景灰

static cv::Mat emptyFrame() { return cv::Mat(kH, kW, CV_8UC3, kBg); }

// 在帧上画一个"新出现的静止物"（深色实心块）
static cv::Mat frameWithItem(cv::Rect r) {
    cv::Mat f = emptyFrame();
    cv::rectangle(f, r, cv::Scalar(20, 20, 20), cv::FILLED);
    return f;
}

static TrackedObject person(int id, cv::Rect box) {
    TrackedObject t;
    t.track_id = id;
    t.box = box;
    t.class_id = 0;
    t.class_name = "people";
    t.confidence = 0.9f;
    t.observed = true;
    return t;
}

// ==================== 驱动器 ====================
struct Harness {
    AbandonConfig cfg;
    std::unique_ptr<ItemAbandonmentDetector> det;
    int64_t now_ms = 1'000'000;
    int     rid = 0;
    std::vector<ItemEvent> all;

    Harness() {
        // 为测试加速：每帧都跑差分、warmup 短、背景学习率拉大、阈值缩短到秒级
        cfg.enable = true;
        cfg.work_w = 320;
        cfg.work_h = 180;
        cfg.warmup_frames = 8;
        cfg.sample_every = 1;
        cfg.bg_long_lr = 0.001;
        cfg.bg_short_lr = 0.5;
        cfg.min_item_px = 40;
        cfg.min_item_area = 1500;
        cfg.max_area_ratio = 0.05;
        cfg.assoc_dist_px = 60.0f;
        cfg.item_lost_frames = 3;
        cfg.reset_dist_px = 10.0f;
        cfg.smooth_alpha = 0.5f;
        cfg.safe_radius_px = 120.0f;
        cfg.bind_confirm_frames = 3;
        cfg.person_iou_reject = 0.30f;
        cfg.remind_sec = 1.0;
        cfg.push_sec = 3.0;
        cfg.backtrack_sec = 2.0;
        cfg.seq_fps = 5;
        cfg.save_dir = "./_tmp_item_evidence";
    }

    void start(const std::string& cam) {
        det = std::make_unique<ItemAbandonmentDetector>(cfg, cam);
    }

    // 推一帧（虚拟时间 +step_ms）
    std::vector<ItemEvent> push(const cv::Mat& f,
                                const std::vector<TrackedObject>& persons = {},
                                int64_t step_ms = 100) {
        now_ms += step_ms;
        auto ev = det->update(f, persons, cv::Mat(), now_ms, ++rid);
        for (auto& e : ev) all.push_back(e);
        return ev;
    }

    int count(ItemEventType t) const {
        int n = 0;
        for (auto& e : all) if (e.type == t) ++n;
        return n;
    }
    bool sawNote(const std::string& note) const {
        for (auto& e : all) if (e.note == note) return true;
        return false;
    }
};

// ==================== 用例 ====================

// T1: 放包 → 走开 → 不取 → 先 L1 提醒、后 L2 推送；静止时长跨过阈值
static void test_l1_l2_flow() {
    std::cout << "  T1 放包→离开→L1→L2\n";
    Harness h; h.start("t1");
    for (int i = 0; i < 10; ++i) h.push(emptyFrame());          // warmup
    const cv::Rect item(100, 200, 60, 60);
    std::vector<ItemEvent> last;
    for (int i = 0; i < 60; ++i) last = h.push(frameWithItem(item));
    check(h.det->items().size() == 1, "应恰好建立 1 条物品轨迹");
    check(h.count(ItemEventType::REMIND) == 1, "应恰好 1 次一级提醒");
    check(h.count(ItemEventType::PUSH) == 1, "应恰好 1 次二级推送");
    // PUSH 的静止时长必须 >= push_sec
    for (auto& e : h.all)
        if (e.type == ItemEventType::PUSH)
            check(e.static_sec >= h.cfg.push_sec - 0.2,
                  "PUSH 触发时 static_sec(" + std::to_string(e.static_sec) + ") 应 >= push_sec");
    check(h.det->level2Count() == 1, "level2Count 应为 1");
}

// T2: L1 之后被移动（渐进位移）→ RESET，计时清零；旧告警不应继续累积
static void test_move_reset() {
    std::cout << "  T2 移动重置\n";
    Harness h; h.start("t2");
    for (int i = 0; i < 10; ++i) h.push(emptyFrame());
    cv::Rect item(100, 200, 60, 60);
    for (int i = 0; i < 20; ++i) h.push(frameWithItem(item));   // 触发 L1(1s @ 100ms/帧 = 10 帧)
    check(h.count(ItemEventType::REMIND) >= 1, "移动前应先出现一级提醒");
    const int push_before = h.count(ItemEventType::PUSH);
    // 每帧右移 20px（>reset_dist 10, <assoc_dist 60）→ 判为"被移动"
    for (int i = 0; i < 6; ++i) { item.x += 20; h.push(frameWithItem(item)); }
    check(h.count(ItemEventType::RESET) >= 1, "渐进位移应触发 RESET");
    check(h.count(ItemEventType::PUSH) == push_before, "移动后不应继续累积到 PUSH");
    // 移动后不计时：继续静止观察 15 帧（1.5s > remind_sec 1.0s）→ 应重新从 0 计时，
    // 但由于静止 1.5s 又会触发一次 REMIND，所以这里只断言"RESET 之后又出现新 REMIND"
    // 说明计时确实被清零重来。
    const int remind_before = h.count(ItemEventType::REMIND);
    for (int i = 0; i < 15; ++i) h.push(frameWithItem(item));
    check(h.count(ItemEventType::REMIND) > remind_before, "重置后重新静止应再触发一次提醒（证明计时清零）");
}

// T3: 告警中主人返回（人进入 safe_radius）→ CLEAR(主人返回)，告警清零
static void test_owner_return() {
    std::cout << "  T3 主人返回清除\n";
    Harness h; h.start("t3");
    for (int i = 0; i < 10; ++i) h.push(emptyFrame());
    const cv::Rect item(100, 200, 60, 60);
    for (int i = 0; i < 20; ++i) h.push(frameWithItem(item));
    check(h.det->alarmCount() >= 1, "先进入告警态");
    // 人靠近物品（物品质心 (130,230)，人框中心 (70,220) → 距离 ≈61px < 120）
    const TrackedObject p = person(7, cv::Rect(20, 170, 100, 100));
    for (int i = 0; i < 6; ++i) h.push(frameWithItem(item), {p});
    check(h.sawNote("主人返回"), "应产生 CLEAR(note=主人返回)");
    check(h.det->alarmCount() == 0, "主人返回后告警数应清零");
}

// T4: 告警中物品被拾取（连续丢失超限）→ CLEAR(被拾取/移出)，轨迹删除
static void test_pickup_clear() {
    std::cout << "  T4 被拾取清除\n";
    Harness h; h.start("t4");
    for (int i = 0; i < 10; ++i) h.push(emptyFrame());
    const cv::Rect item(100, 200, 60, 60);
    for (int i = 0; i < 20; ++i) h.push(frameWithItem(item));
    check(h.det->items().size() == 1, "先有 1 条轨迹");
    // 物品消失：继续喂纯背景帧
    for (int i = 0; i < 10; ++i) h.push(emptyFrame());
    check(h.sawNote("被拾取/移出"), "应产生 CLEAR(note=被拾取/移出)");
    check(h.det->items().empty(), "被拾取后轨迹应被删除");
}

// T5: 站着不动的人（静止块与人框高度重叠）→ 不建轨迹
static void test_standing_person_rejected() {
    std::cout << "  T5 站着的人不建轨\n";
    Harness h; h.start("t5");
    for (int i = 0; i < 10; ++i) h.push(emptyFrame());
    const cv::Rect body(100, 100, 60, 200);          // "人"的静止轮廓
    const TrackedObject p = person(3, cv::Rect(90, 90, 80, 220));  // person 框覆盖它
    for (int i = 0; i < 30; ++i) h.push(frameWithItem(body), {p});
    check(h.det->items().empty(), "站住不动的人不应产生滞留物轨迹");
    check(h.count(ItemEventType::REMIND) == 0 && h.count(ItemEventType::PUSH) == 0,
          "不应产生任何滞留告警");
}

// T6: 旁边一直有人看管 → 不计时、不告警
static void test_attended_no_alarm() {
    std::cout << "  T6 有人看管不告警\n";
    Harness h; h.start("t6");
    for (int i = 0; i < 10; ++i) h.push(emptyFrame());
    const cv::Rect item(100, 200, 60, 60);
    const TrackedObject p = person(11, cv::Rect(20, 170, 100, 100));  // 距物 ≈61px < 120
    for (int i = 0; i < 50; ++i) h.push(frameWithItem(item), {p});    // 5 秒，远超 push_sec=3s
    check(h.count(ItemEventType::REMIND) == 0, "有人看管时不应有一级提醒");
    check(h.count(ItemEventType::PUSH) == 0, "有人看管时不应有二级推送");
    check(h.det->alarmCount() == 0, "告警数应为 0");
}

// T7: 看管者离开 → 计时从"离开那一刻"起算（而不是从物品出现起算）
static void test_timer_starts_on_leave() {
    std::cout << "  T7 计时从主人离开起算\n";
    Harness h; h.start("t7");
    for (int i = 0; i < 10; ++i) h.push(emptyFrame());
    const cv::Rect item(100, 200, 60, 60);
    const TrackedObject p = person(11, cv::Rect(20, 170, 100, 100));
    // 前 40 帧（4s）一直有人看管：若计时从出现起算，早就该 PUSH 了
    for (int i = 0; i < 40; ++i) h.push(frameWithItem(item), {p});
    check(h.count(ItemEventType::PUSH) == 0, "看管期间不应推送");
    // 人离开后 1.2s（>remind_sec 1.0）应出现 REMIND，而不是立刻 PUSH
    for (int i = 0; i < 12; ++i) h.push(frameWithItem(item));
    check(h.count(ItemEventType::REMIND) == 1, "离开后 1.2s 应出现 1 次一级提醒");
    check(h.count(ItemEventType::PUSH) == 0, "离开仅 1.2s 不应直接二级推送");
}

// T8: 抽样帧间隔（sample_every>1）下计时仍按真实时间推进
static void test_sampling_keeps_timing() {
    std::cout << "  T8 抽样下计时正确\n";
    Harness h;
    h.cfg.sample_every = 5;          // 每 5 帧跑一次差分
    h.cfg.push_sec = 5.0;
    h.start("t8");
    for (int i = 0; i < 20; ++i) h.push(emptyFrame());   // warmup（抽样帧计数）
    const cv::Rect item(100, 200, 60, 60);
    // 总时长 8s（80 帧 × 100ms），其中抽样帧 16 个
    for (int i = 0; i < 80; ++i) h.push(frameWithItem(item));
    check(h.count(ItemEventType::PUSH) == 1, "抽样模式下仍应在 push_sec 触发二级推送");
    for (auto& e : h.all)
        if (e.type == ItemEventType::PUSH)
            check(e.static_sec >= 4.5, "抽样模式下 static_sec 应接近真实经过时间");
}

// ==================== main ====================
int main() {
    std::cout << "=== test_item_abandonment ===\n";
    test_l1_l2_flow();
    test_move_reset();
    test_owner_return();
    test_pickup_clear();
    test_standing_person_rejected();
    test_attended_no_alarm();
    test_timer_starts_on_leave();
    test_sampling_keeps_timing();

    std::cout << "\n---- 汇总 ----\n";
    std::cout << "通过: " << g_pass << "  失败: " << g_fail << "\n";
    return g_fail == 0 ? 0 : 1;
}
