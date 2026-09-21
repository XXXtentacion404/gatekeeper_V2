// ============================================================================
// ByteTrack 遮挡重连（二次关联）宿主侧功能测试
//
// 要求 2 的验证：简单的遮挡不得导致 ID 切换。
//   场景 A：目标右行被遮挡 4 帧，遮挡期间其实掉头往回走 —— 卡尔曼外推框与
//           真实框完全不重叠（IoU=0，主通道必然失配），此时必须靠"遮挡重连"
//           把检测挂回原 ID，而不是新建 ID。
//   场景 B（反向保护）：另一个外观明显不同的人出现在搜索窗内 → 外观门限必须
//           拒绝关联，给它新建 ID（不允许 ID 盗用/身份串号）。
// ============================================================================
#include "bytetrack/tracker.h"

#include <algorithm>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    g_checks++;
    if (!ok) {
        g_failures++;
        std::printf("  [FAIL] %s\n", what.c_str());
    } else {
        std::printf("  [ ok ] %s\n", what.c_str());
    }
}

const cv::Scalar kBlue(255, 0, 0);   // BGR
const cv::Scalar kRed(0, 0, 255);
const int kW = 640, kH = 480;

cv::Mat makeFrame(const std::vector<std::pair<cv::Rect, cv::Scalar>>& persons) {
    cv::Mat f(kH, kW, CV_8UC3, cv::Scalar(128, 128, 128));
    for (const auto& p : persons) {
        cv::Rect v = p.first & cv::Rect(0, 0, kW, kH);
        if (v.area() > 0) cv::rectangle(f, v, p.second, cv::FILLED);
    }
    return f;
}

Detection makeDet(const cv::Rect& box, float conf) {
    Detection d;
    d.box = box;
    d.confidence = conf;
    d.class_id = 0;
    d.className = "people";
    return d;
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect i = a & b;
    if (i.area() <= 0) return 0.0f;
    return (float)i.area() / (float)(a.area() + b.area() - i.area());
}

}  // namespace

int main() {
    std::printf("ByteTrack 遮挡重连测试\n");

    ByteTrackTracker tracker(0.18f, 0.35f, 0.25f, 180, 2);
    std::set<int> all_ids;
    const cv::Rect person_size(0, 0, 80, 160);
    const int y = 200;

    auto person_box = [&](int cx) {
        return cv::Rect(cx - person_size.width / 2, y,
                        person_size.width, person_size.height);
    };

    // ---- 场景 A 阶段 1：目标右行 12 帧，速度 +20px/帧 ----
    int last_cx = 0;
    std::vector<int> ids_phase1;
    for (int k = 0; k < 12; k++) {
        const int cx = 100 + 20 * k;
        last_cx = cx;
        cv::Rect box = person_box(cx);
        cv::Mat frame = makeFrame({{box, kBlue}});
        auto out = tracker.update({makeDet(box, 0.90f)}, frame, kW, kH);
        for (const auto& t : out) {
            all_ids.insert(t.track_id);
            if (t.observed) ids_phase1.push_back(t.track_id);
        }
    }
    check(!ids_phase1.empty(), "阶段1：目标已被稳定跟踪并输出");
    const int id_a = ids_phase1.back();
    check(all_ids.size() == 1, "阶段1：始终只有一个 ID（无抖动切换）");
    std::printf("  (跟踪 ID = %d, 最后中心 x = %d)\n", id_a, last_cx);

    // ---- 场景 A 阶段 2：遮挡 4 帧（检测器无输出，跟踪器仅预测）----
    for (int k = 0; k < 4; k++) {
        cv::Mat frame = makeFrame({});
        tracker.update({}, frame, kW, kH);
    }

    // ---- 场景 A 阶段 3：目标掉头往回走，位置与卡尔曼外推框完全不重叠 ----
    // 卡尔曼外推：last_cx + 20 * (遮挡帧数 + 1) = 320 + 100 = 420
    const int predicted_cx = last_cx + 20 * (4 + 1);
    const int reappear_cx = 280;   // 实际掉头往回走
    const cv::Rect predicted_box = person_box(predicted_cx);
    const cv::Rect reappear_box = person_box(reappear_cx);
    const float main_iou = iou(predicted_box, reappear_box);
    std::printf("  (重连前 IoU(外推框, 真实框) = %.3f → 主通道必然失配)\n", main_iou);
    check(main_iou < 0.05f, "构造出的遮挡场景确实无法被 IoU 主通道匹配");

    bool same_id_after_occlusion = false;
    bool new_id_created = false;
    for (int k = 0; k < 4; k++) {
        const int cx = reappear_cx + 20 * k;   // 重现后继续行走
        cv::Rect box = person_box(cx);
        cv::Mat frame = makeFrame({{box, kBlue}});
        auto out = tracker.update({makeDet(box, 0.90f)}, frame, kW, kH);
        for (const auto& t : out) {
            all_ids.insert(t.track_id);
            if (t.observed && t.track_id == id_a) same_id_after_occlusion = true;
            if (t.observed && t.track_id != id_a) new_id_created = true;
        }
    }
    check(same_id_after_occlusion, "遮挡后仍以原 ID 跟踪（未被切换）");
    check(!new_id_created, "遮挡后没有生成新 ID");

    // ---- 场景 B：外观不同的另一个人出现在搜索窗内 → 必须新建 ID ----
    for (int k = 0; k < 3; k++) {           // 目标再消失 3 帧
        cv::Mat frame = makeFrame({});
        tracker.update({}, frame, kW, kH);
    }
    bool stranger_new_id = false;
    bool stranger_stole_id = false;
    for (int k = 0; k < 4; k++) {
        cv::Rect box = person_box(reappear_cx + 20 * 3);   // 站在原目标上次位置附近
        cv::Mat frame = makeFrame({{box, kRed}});          // 红色：与蓝色目标外观不同
        auto out = tracker.update({makeDet(box, 0.90f)}, frame, kW, kH);
        for (const auto& t : out) {
            if (!t.observed) continue;
            all_ids.insert(t.track_id);
            if (t.track_id == id_a) stranger_stole_id = true;
            else stranger_new_id = true;
        }
    }
    check(stranger_new_id, "外观不同的人获得新 ID");
    check(!stranger_stole_id, "外观门限拒绝串号（未盗用原 ID）");

    std::printf("\n==== 结果: %d 项检查, %d 项失败 (累计 ID 数=%d) ====\n",
                g_checks, g_failures, (int)all_ids.size());
    return g_failures == 0 ? 0 : 1;
}
