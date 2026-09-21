// ============================================================================
// 区域几何的【跨语言一致性自检】—— 离线工具，不属于正式测试套件。
//
// 背景：判定线 = 区域边缘折线（zin,/zout,，点数不限），区域 = 折线 ∪ 其沿 x 平移 |depth| px
//       的副本。同一份 cameras_config.txt 现在被两处独立实现消费：
//         1) C++  src/gate_id_counter.cpp   ::zonePolygon()/regionOf()
//         2) Py   闸门计数-网络摄像头配置文件绘制.py :: _zone_ring()
//       两侧一旦算的方向不一致，Python 预览的区域和板端判定的区域会分居判定线两侧，
//       而画面看起来"没问题"—— 必须机器对拍，不能靠肉眼看。
//
// 用法：chk_zone.exe cameras_config.txt > cpp.txt
//       再用 _chk_zone_py.py 生成 py.txt，diff 两份输出（应逐行相同）。
// 输出：每行 "<cam> <x> <y> <region>"，region 0=区域外 1=区域 2=检测区。
// ============================================================================
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "gate_id_counter.h"

int main(int argc, char** argv) {
    const char* cfg_path = argc > 1 ? argv[1] : "cameras_config.txt";
    std::ifstream in(cfg_path);
    if (!in) {
        std::printf("# cannot open %s\n", cfg_path);
        return 1;
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::vector<std::string> tk;
        std::string t;
        while (ss >> t) tk.push_back(t);
        if (tk.size() < 4) continue;

        int W = 1920, H = 1080;
        size_t start = 4;
        if (tk.size() > 4 && tk[4].find('*') != std::string::npos) {
            std::sscanf(tk[4].c_str(), "%d*%d", &W, &H);
            start = 5;
        }

        std::vector<std::vector<cv::Point>> polys;
        Tripwire t1;
        std::vector<cv::Point2f> zone_line;
        float depth = 0.0f;
        bool depth_explicit = false;
        int role_from_cfg = -1;
        std::string poly_acc;

        auto flushPoly = [&]() {
            if (poly_acc.empty()) return;
            std::vector<cv::Point> poly;
            std::istringstream ps(poly_acc);
            std::string p;
            while (ps >> p) {
                size_t c = p.find(',');
                if (c == std::string::npos) continue;
                poly.push_back(cv::Point(std::stoi(p.substr(0, c)),
                                         std::stoi(p.substr(c + 1))));
            }
            if (poly.size() >= 3) polys.push_back(poly);
            poly_acc.clear();
        };

        for (size_t i = start; i < tk.size(); ++i) {
            if (tk[i].rfind("t1,", 0) == 0) {
                std::string body = tk[i].substr(3);
                std::replace(body.begin(), body.end(), ',', ' ');
                std::istringstream ps(body);
                float x1, y1, x2, y2, d;
                ps >> x1 >> y1 >> x2 >> y2;
                if (ps >> d) { depth = d; depth_explicit = true; }
                t1 = Tripwire(x1, y1, x2, y2);
            } else if (tk[i].rfind("zin,", 0) == 0 || tk[i].rfind("zout,", 0) == 0) {
                const bool is_entry = (tk[i].rfind("zin,", 0) == 0);
                std::string body = tk[i].substr(4);
                std::replace(body.begin(), body.end(), ',', ' ');
                std::istringstream ps(body);
                std::vector<float> nums; float v;
                while (ps >> v) nums.push_back(v);
                if (nums.size() >= 4 && nums.size() % 2 == 0) {
                    zone_line.clear();
                    for (size_t k = 0; k + 1 < nums.size(); k += 2)
                        zone_line.emplace_back(nums[k], nums[k + 1]);
                    role_from_cfg = is_entry ? 0 : 1;
                }
            } else if (tk[i].rfind("zo,", 0) == 0) {
                depth = (float)std::atof(tk[i].c_str() + 3);
                depth_explicit = true;
            } else if (tk[i][0] == '[') {
                size_t e = tk[i].find(']');
                poly_acc = (e == std::string::npos) ? tk[i].substr(1)
                                                   : tk[i].substr(1, e - 1);
                if (e != std::string::npos) flushPoly();
            } else if (!poly_acc.empty()) {
                size_t e = tk[i].find(']');
                poly_acc += " " + ((e == std::string::npos) ? tk[i] : tk[i].substr(0, e));
                if (e != std::string::npos) flushPoly();
            }
        }
        if (!zone_line.empty() && !depth_explicit) depth = 100.0f;

        cv::Mat mask = cv::Mat::zeros(H, W, CV_8UC1);
        if (!polys.empty()) cv::fillPoly(mask, polys, cv::Scalar(255));

        // 角色：优先用配置里显式写的（zin=入口 / zout=出口），否则按 IP 末段
        // （与 main_tracking.cpp loadCameras 一致；regionOf 与角色无关）
        const size_t dot = tk[0].rfind('.');
        const std::string last3 = (dot == std::string::npos) ? "" : tk[0].substr(dot + 1);
        const GateRole role = (role_from_cfg >= 0)
            ? (role_from_cfg == 0 ? GATE_ROLE_ENTRY : GATE_ROLE_EXIT)
            : (last3 == "201" ? GATE_ROLE_ENTRY : GATE_ROLE_EXIT);

        GateIdCounter gc = zone_line.size() >= 2
            ? GateIdCounter(role, zone_line, depth, mask)
            : GateIdCounter(role, t1, depth, mask);

        std::printf("# %s W=%d H=%d depth=%.0f polys=%zu\n",
                    tk[0].c_str(), W, H, depth, polys.size());
        for (int y = 0; y < H; y += 60) {
            for (int x = 0; x < W; x += 60) {
                std::printf("%s %d %d %d\n", last3.c_str(), x, y,
                            gc.regionOf(cv::Point2f((float)x, (float)y)));
            }
        }
    }
    return 0;
}
