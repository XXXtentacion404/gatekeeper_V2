#include "mask_utils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

// ---- 解析配置行 ---- 
std::vector<CameraConfigEntry> parseCameraConfig(const std::string& configFile) {
    std::vector<CameraConfigEntry> configs;
    std::ifstream file(configFile);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        CameraConfigEntry config;
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (iss >> token) tokens.push_back(token);
        if (tokens.size() < 4) continue;
        
        config.ip = tokens[0];
        config.username = tokens[1];
        config.password = tokens[2];
        config.channel = std::stoi(tokens[3]);
        config.width = 1920;
        config.height = 1080;
        
        if (tokens.size() > 4 && tokens[4].find('*') != std::string::npos) {
            size_t pos = tokens[4].find('*');
            try {
                config.width = std::stoi(tokens[4].substr(0, pos));
                config.height = std::stoi(tokens[4].substr(pos+1));
            } catch (...) {}
        }
        
        size_t poly_start = (tokens.size() > 4 && tokens[4].find('*') != std::string::npos) ? 5 : 4;
        for (size_t i = poly_start; i < tokens.size(); i++) {
            // 绊线: t1,x1,y1,x2,y2[,depth] 或 t2,x1,y1,x2,y2[,depth]
            if (tokens[i].rfind("t1,", 0) == 0) {
                std::string body = tokens[i].substr(3);
                std::replace(body.begin(), body.end(), ',', ' ');
                std::istringstream ps(body);
                float x1, y1, x2, y2, depth = 0;
                ps >> x1 >> y1 >> x2 >> y2;
                if (ps >> depth) config.entry_zone_depth = depth;
                config.entry_line = Tripwire(x1, y1, x2, y2);
            }
            else if (tokens[i].rfind("t2,", 0) == 0) {
                std::string body = tokens[i].substr(3);
                std::replace(body.begin(), body.end(), ',', ' ');
                std::istringstream ps(body);
                float x1, y1, x2, y2, depth = 0;
                ps >> x1 >> y1 >> x2 >> y2;
                if (ps >> depth) config.exit_zone_depth = depth;
                config.exit_line = Tripwire(x1, y1, x2, y2);
            }
            // 检测区域多边形 [x1,y1 x2,y2 ...]
            else if (tokens[i][0] == '[') {
                std::string poly_str;
                size_t end_pos = tokens[i].find(']');
                if (end_pos != std::string::npos) {
                    poly_str = tokens[i].substr(1, end_pos-1);
                } else {
                    poly_str = tokens[i].substr(1);
                    for (i++; i < tokens.size(); i++) {
                        end_pos = tokens[i].find(']');
                        if (end_pos != std::string::npos) {
                            poly_str += " " + tokens[i].substr(0, end_pos);
                            break;
                        }
                        poly_str += " " + tokens[i];
                    }
                }
                
                std::vector<cv::Point> polygon;
                std::istringstream point_ss(poly_str);
                std::string point;
                while (point_ss >> point) {
                    size_t comma_pos = point.find(',');
                    if (comma_pos != std::string::npos) {
                        try {
                            int x = std::stoi(point.substr(0, comma_pos));
                            int y = std::stoi(point.substr(comma_pos+1));
                            x = std::max(0, std::min(x, config.width-1));
                            y = std::max(0, std::min(y, config.height-1));
                            polygon.emplace_back(x, y);
                        } catch (...) {}
                    }
                }
                if (polygon.size() >= 3) {
                    if (polygon.front() != polygon.back()) polygon.push_back(polygon.front());
                    config.detection_zones.push_back(polygon);
                }
            }
        }
        
        // 保存掩膜预览
        cv::Mat mask = createDetectionMask(config.width, config.height, config.detection_zones);
        std::string safe_ip = config.ip;
        std::replace(safe_ip.begin(), safe_ip.end(), ':', '_');
        std::string filename = "mask_" + safe_ip + "_ch" + std::to_string(config.channel) + ".png";
        if (!mask.empty()) {
            cv::imwrite(filename, mask);
            std::cout << "Saved mask: " << filename << std::endl;
        }
        configs.push_back(config);
    }
    return configs;
}

// ---- 创建二值掩膜 ---- 
cv::Mat createDetectionMask(int width, int height, const std::vector<std::vector<cv::Point>>& zones) {
    cv::Mat mask = cv::Mat::zeros(height, width, CV_8UC1);
    if (zones.empty()) return mask;
    for (const auto& zone : zones) {
        if (zone.size() < 3) continue;
        cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{zone}, cv::Scalar(255), 8);
    }
    return mask;
}

// ---- 判定 bbox 是否在区域内（脚底判定 + 面积比双保险）----
// 核心：检测区画在地板上，人上半身可能超出区域但脚不会
// 1) 优先检查底部中心点（脚一定在地板上）
// 2) 不明确时检查 bbox 底部 35% 的面积比
bool isInsideZone(const cv::Rect& box, const cv::Mat& mask, float threshold) {
    if (mask.empty() || box.area() <= 0) return false;
    
    // 底边中点（脚底位置）
    int cx = box.x + box.width / 2;
    int cy = box.y + (int)(box.height * 0.92f);
    cx = std::max(0, std::min(cx, mask.cols - 1));
    cy = std::max(0, std::min(cy, mask.rows - 1));
    
    // 脚底单点快速判定
    if (mask.at<uchar>(cy, cx) != 0) return true;
    
    // 脚底不中 → 检查 bbox 底部 35% 面积比
    int foot_h = std::max(1, (int)(box.height * 0.35f));
    int foot_y = box.y + box.height - foot_h;
    cv::Rect foot_box(box.x, foot_y, box.width, foot_h);
    
    cv::Rect safe = foot_box & cv::Rect(0, 0, mask.cols, mask.rows);
    if (safe.area() <= 0) return false;
    cv::Mat roi = mask(safe);
    float ratio = (float)cv::countNonZero(roi) / (float)safe.area();
    return ratio >= threshold;
}

// ---- 2D 线段相交检测（严格跨立判据）----
bool segmentsIntersect(const cv::Point2f& A, const cv::Point2f& B,
                       const cv::Point2f& C, const cv::Point2f& D) {
    auto cross = [](const cv::Point2f& a, const cv::Point2f& b) {
        return a.x * b.y - a.y * b.x;
    };
    cv::Point2f AB = B - A;
    float d1 = cross(AB, C - A);
    float d2 = cross(AB, D - A);
    cv::Point2f CD = D - C;
    float d3 = cross(CD, A - C);
    float d4 = cross(CD, B - C);
    // 严格跨立: 两端点在异侧 (乘积<0)
    return (d1 * d2 < 0.0f) && (d3 * d4 < 0.0f);
}

// ---- 绘制绊线到图像（入口绿 / 出口红）----
void drawTripwires(cv::Mat& img, const Tripwire& entry, const Tripwire& exit) {
    if (entry.isValid()) {
        cv::line(img, cv::Point((int)entry.p1.x, (int)entry.p1.y),
                 cv::Point((int)entry.p2.x, (int)entry.p2.y),
                 cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }
    if (exit.isValid()) {
        cv::line(img, cv::Point((int)exit.p1.x, (int)exit.p1.y),
                 cv::Point((int)exit.p2.x, (int)exit.p2.y),
                 cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }
}
