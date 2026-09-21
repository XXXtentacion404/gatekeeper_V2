// ============================================================================
// 真实 cameras_config.txt 的端到端冒烟测试（宿主侧，不属于"正式套件"但建议常跑）
//
// 为什么单独有这一个：
//   正式套件 test_gate_counter.cpp 用的是**合成**几何（竖直/折角判定线），
//   而现场真正跑的是配置文件里那条**几十个顶点的边界弧**（zin,/zout, 折线）。
//   2026-09-16 的板端事故（"画面不画区域 + IN/OUT/RETRO 全程 0"）根因就是把配置
//   解析出来的 zone_line 漏搬进 CameraConfig —— 这一类"接线"错误合成测试抓不到，
//   所以这里直接拿真配置跑：解析 -> 建掩膜 -> 建 GateIdCounter -> 喂一条穿越轨迹，
//   断言【判定线/区域环非空】且【恰好各产生一次进入类/离开类事件】。
//
// 用法: test_real_config.exe [cameras_config.txt]
// 退出码 0 = 全过。上板前先跑它，比在板上试错快得多。
// ============================================================================
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "gate_id_counter.h"

static int g_total = 0;
static int g_fail = 0;

static void check(const char* name, bool ok, const std::string& detail = std::string()) {
    ++g_total;
    if (!ok) ++g_fail;
    std::printf("  [%s] %s%s\n", ok ? "ok" : "FAIL", name,
                (ok || detail.empty()) ? "" : ("  " + detail).c_str());
}

struct ParsedCam {
    std::string ip;
    std::vector<cv::Point> poly;
    std::vector<cv::Point2f> line;   // 判定线折线（zin,/zout,）
    int role = 0;                    // 0=入口 1=出口
    float depth = 100.0f;
    std::string kind = "无";
};

// 只解析本测试需要的部分：多边形 + 判定线 token。
// （不直接调 mask_utils.cpp 是因为它依赖 cv::imwrite，宿主 OpenCV 只编译了 core/imgproc/video）
static std::vector<ParsedCam> parseConfig(const std::string& path) {
    std::vector<ParsedCam> out;
    std::ifstream in(path);
    if (!in) return out;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::vector<std::string> toks;
        std::string t;
        while (ss >> t) toks.push_back(t);
        if (toks.size() < 6) continue;
        ParsedCam c;
        c.ip = toks[0];
        // 多边形 [x,y x,y ...]：可能跨多个空白分隔的 token
        std::string poly_str;
        bool in_poly = false, got_poly = false;
        for (size_t i = 5; i < toks.size(); ++i) {
            const std::string& tk = toks[i];
            if (!in_poly && !tk.empty() && tk[0] == '[') {
                in_poly = true;
                poly_str = tk.substr(1);
                if (!poly_str.empty() && poly_str.back() == ']') {
                    poly_str.pop_back(); in_poly = false; got_poly = true;
                }
                continue;
            }
            if (in_poly) {
                if (!tk.empty() && tk.back() == ']') {
                    poly_str += " " + tk.substr(0, tk.size() - 1);
                    in_poly = false; got_poly = true;
                } else {
                    poly_str += " " + tk;
                }
                continue;
            }
            // 判定线 token
            if (tk.rfind("zin,", 0) == 0 || tk.rfind("zout,", 0) == 0) {
                const bool is_entry = (tk.rfind("zin,", 0) == 0);
                std::string body = tk.substr(4);
                std::replace(body.begin(), body.end(), ',', ' ');
                std::istringstream ps(body);
                std::vector<float> nums; float v = 0.0f;
                while (ps >> v) nums.push_back(v);
                if (nums.size() >= 4 && nums.size() % 2 == 0) {
                    c.line.clear();
                    for (size_t k = 0; k + 1 < nums.size(); k += 2)
                        c.line.emplace_back(nums[k], nums[k + 1]);
                    c.role = is_entry ? 0 : 1;
                    c.kind = is_entry ? "zin(入口)" : "zout(出口)";
                }
            } else if (tk.rfind("zo,", 0) == 0) {
                try { c.depth = std::stof(tk.substr(3)); } catch (...) {}
            }
        }
        if (got_poly && !poly_str.empty()) {
            std::istringstream ps(poly_str);
            std::string pt;
            while (ps >> pt) {
                const size_t comma = pt.find(',');
                if (comma == std::string::npos) continue;
                try {
                    c.poly.emplace_back(std::stoi(pt.substr(0, comma)),
                                        std::stoi(pt.substr(comma + 1)));
                } catch (...) {}
            }
        }
        if (!c.poly.empty()) out.push_back(c);
    }
    return out;
}

// 折线弧长中点所在段（与 GateIdCounter::midSegment 同口径，这里独立实现以便交叉验证）
static int midSegOf(const std::vector<cv::Point2f>& pts) {
    if (pts.size() < 2) return -1;
    float total = 0.0f;
    for (size_t i = 0; i + 1 < pts.size(); ++i) total += (float)cv::norm(pts[i + 1] - pts[i]);
    const float half = total * 0.5f;
    float acc = 0.0f;
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        const float L = (float)cv::norm(pts[i + 1] - pts[i]);
        if (L < 1e-3f) continue;
        if (acc + L >= half) return (int)i;
        acc += L;
    }
    return (int)pts.size() - 2;
}

static TrackedObject mkFoot(const cv::Point2f& foot, int fid, int tid) {
    TrackedObject t;
    t.track_id = tid;
    const int w = 48, h = 120;
    t.box = cv::Rect((int)std::lround(foot.x - w * 0.5),
                     (int)std::lround(foot.y - h * 0.92f), w, h);
    t.confidence = 0.90f;
    t.class_id = 0;
    t.class_name = "people";
    t.observed = true;
    t.hit_streak = 5;          // 过 kMinHitStreak / kMinConfidence 质量门
    t.time_since_update = 0;
    t.result_frame_id = fid;
    return t;
}

int main(int argc, char** argv) {
    const std::string path = (argc > 1) ? argv[1] : "cameras_config.txt";
    std::printf("=== 真实配置端到端冒烟: %s ===\n", path.c_str());
    const std::vector<ParsedCam> cams = parseConfig(path);
    check("解析出相机(含多边形与判定线)", !cams.empty(),
          "空 -> 配置路径不对或没有 zin,/zout,");
    if (cams.empty()) {
        std::printf("\n==== 结果: %d 项检查, %d 项失败 ====\n", g_total, g_fail);
        return 1;
    }

    int fid = 0;
    for (const auto& c : cams) {
        std::printf("\n-- %s  判定线=%s %d 顶点  区域沿 x 平移 %+.0f ---\n",
                    c.ip.c_str(), c.kind.c_str(), (int)c.line.size(), c.depth);
        check((c.ip + " 判定线至少 2 点").c_str(), c.line.size() >= 2,
              "解析到 " + std::to_string(c.line.size()) + " 点");

        cv::Mat mask(1080, 1920, CV_8UC1, cv::Scalar(0));
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{c.poly}, cv::Scalar(255));
        const GateRole role = (c.role == 0) ? GATE_ROLE_ENTRY : GATE_ROLE_EXIT;
        GateIdCounter counter(role, c.line, c.depth, mask);

        const int want_enter = (role == GATE_ROLE_ENTRY) ? GATE_EVENT_IN : GATE_EVENT_RETRO;
        const int want_leave = (role == GATE_ROLE_ENTRY) ? GATE_EVENT_RETRO : GATE_EVENT_OUT;
        const char* want_enter_name = (role == GATE_ROLE_ENTRY) ? "IN" : "RETRO";
        const char* want_leave_name = (role == GATE_ROLE_ENTRY) ? "RETRO" : "OUT";

        // 1) 接线自检：判定线 / 区域环必须非空（板端"不画区域"就是这里空掉了）
        //    注意要写 >=2 / >=3 而不是 == c.line.size()：后者在"判定线为空"时 0==0 会假通过。
        check((c.ip + " GateIdCounter.judgingLine() ≥2 点").c_str(),
              counter.judgingLine().size() >= 2,
              "got=" + std::to_string(counter.judgingLine().size()));
        check((c.ip + " zoneRing() ≥3 顶点(=2N)").c_str(),
              counter.zoneRing().size() >= 3 &&
                  counter.zoneRing().size() == 2 * counter.judgingLine().size(),
              "got=" + std::to_string(counter.zoneRing().size()));

        // 2) 找一段"跨越它能明确区分多边形内外"的线段，作为穿越点
        const std::vector<cv::Point2f>& jl = counter.judgingLine();
        int use_seg = -1;
        cv::Point2f dir(0, 0), nrm(0, 0), mid(0, 0);
        float inside_sign = 0.0f;
        const int start = midSegOf(jl);
        for (int k = 0; k < (int)jl.size() - 1; ++k) {
            const int i = (start + k) % ((int)jl.size() - 1);
            const cv::Point2f d = jl[i + 1] - jl[i];
            const float L = std::sqrt(d.dot(d));
            if (L < 1.0f) continue;
            const cv::Point2f n(d.y / L, -d.x / L);
            const cv::Point2f m = (jl[i] + jl[i + 1]) * 0.5f;
            const cv::Point2f pp = m + n * 60.0f, pn = m - n * 60.0f;
            if (pp.x < 5 || pp.x > 1915 || pp.y < 5 || pp.y > 1075) continue;
            if (pn.x < 5 || pn.x > 1915 || pn.y < 5 || pn.y > 1075) continue;
            const double dp = cv::pointPolygonTest(c.poly, pp, true);
            const double dn = cv::pointPolygonTest(c.poly, pn, true);
            if (std::abs(dp) < 40.0 || std::abs(dn) < 40.0) continue;   // 两侧要都在多边形里/外明显处
            if ((dp > 0) == (dn > 0)) continue;                          // 必须一内一外
            use_seg = i; dir = d; nrm = n; mid = m;
            inside_sign = (dp > 0) ? 1.0f : -1.0f;
            break;
        }
        if (use_seg < 0) {
            check((c.ip + " 找到可穿越的边界线段").c_str(), false, "该相机判定线两端都贴边，跳过");
            continue;
        }
        check((c.ip + " 找到可穿越的边界线段").c_str(), true);

        // 沿法向做"逐帧小步"移动：单帧位移 ≤20px，避开遮挡脚底冻结(阈值 40~60px)
        // 与 M2 假框守卫(>300px)。k>0 = 多边形内侧。
        auto ptAt = [&](float k) { return mid + nrm * (inside_sign * k); };

        // 3) 穿越一：线外 -> 线内，应恰好 1 次"进入类"
        const float enter_seq[] = {-60, -45, -30, -10, 10, 30, 45, 60};
        int n_enter = 0, n_leave = 0, other = 0;
        for (float k : enter_seq) {
            const std::vector<TrackedObject> tr{mkFoot(ptAt(k), ++fid, 7)};
            for (const auto& e : counter.update(tr)) {
                if (e.type == want_enter) ++n_enter;
                else if (e.type == want_leave) ++n_leave;
                else ++other;
            }
        }
        check((c.ip + std::string(" 线外->线内 恰好 1 次 ") + want_enter_name).c_str(),
              n_enter == 1 && n_leave == 0,
              "进入类=" + std::to_string(n_enter) + " 离开类=" + std::to_string(n_leave));

        // 3b) marks() 里新增的调试字段（现场排查"某台相机全程 0 事件"要用它）：
        //     刚跨进线内侧，应当 side=1 且 signed>0，且脚底投影在折线跨度内。
        for (const auto& mk : counter.marks()) {
            if (mk.track_id != 7 || mk.lost_frames > 0) continue;
            check((c.ip + " marks() 暴露 line_side/line_signed（已进到线内侧）").c_str(),
                  mk.line_side == 1 && mk.line_signed > 0.0f && mk.line_active,
                  "side=" + std::to_string(mk.line_side) +
                      " signed=" + std::to_string((int)mk.line_signed) +
                      " active=" + std::to_string((int)mk.line_active));
        }

        // 4) 穿越二：线内 -> 线外，应恰好 1 次"离开类"
        const float leave_seq[] = {60, 45, 30, 10, -10, -30, -45, -60};
        n_enter = n_leave = other = 0;
        for (float k : leave_seq) {
            const std::vector<TrackedObject> tr{mkFoot(ptAt(k), ++fid, 7)};
            for (const auto& e : counter.update(tr)) {
                if (e.type == want_enter) ++n_enter;
                else if (e.type == want_leave) ++n_leave;
                else ++other;
            }
        }
        check((c.ip + std::string(" 线内->线外 恰好 1 次 ") + want_leave_name).c_str(),
              n_leave == 1 && n_enter == 0,
              "进入类=" + std::to_string(n_enter) + " 离开类=" + std::to_string(n_leave));
        check((c.ip + " 没有其它类型事件").c_str(), other == 0,
              "other=" + std::to_string(other));
    }

    std::printf("\n==== 结果: %d 项检查, %d 项失败 ====\n", g_total, g_fail);
    return g_fail ? 1 : 0;
}
