#ifndef CV_DRAW_H
#define CV_DRAW_H

#include <opencv2/opencv.hpp>
#include "types.h"

void DrawDetections(cv::Mat& img, const std::vector<Detection>& detections);

#endif // CV_DRAW_H
