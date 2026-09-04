#ifndef MASK_UTILS_H
#define MASK_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "types.h"

// 解析配置文件，提取检测区域多边形 + 绊线 + 方向配置
std::vector<CameraConfigEntry> parseCameraConfig(const std::string& configFile);

// 从多边形列表创建二值掩膜 (0=不可信, 255=可信)
cv::Mat createDetectionMask(int width, int height,
                            const std::vector<std::vector<cv::Point>>& zones);

// 判定 bbox 是否在检测区域内（脚底判定 + 面积比双保险）
bool isInsideZone(const cv::Rect& box, const cv::Mat& mask, float threshold = 0.25f);

// 2D 线段相交检测（严格跨立）
bool segmentsIntersect(const cv::Point2f& A, const cv::Point2f& B,
                       const cv::Point2f& C, const cv::Point2f& D);

// 绘制绊线到图像（入口绿 / 出口红）
void drawTripwires(cv::Mat& img, const Tripwire& entry, const Tripwire& exit);

#endif
