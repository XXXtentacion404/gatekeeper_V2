#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <opencv2/opencv.hpp>
#include "types.h"

struct LetterBoxInfo {
    bool hor = false;      // 补边方向: true=左右补(hor pad), false=上下补
    int pad = 0;           // 补边像素(画布坐标系)
    float scale = 1.0f;    // 原图像素 / 画布内容像素, 坐标还原用
    bool rga = false;      // 画布是否由 RGA 生成(已是 RGB, 可直接 memcpy)
};

LetterBoxInfo letterbox(const cv::Mat& img, cv::Mat& img_letterbox, float wh_ratio);
LetterBoxInfo letterbox_rga(const cv::Mat& img, cv::Mat& img_letterbox, float wh_ratio);
void cvimg2tensor(const cv::Mat& img, uint32_t width, uint32_t height, tensor_data_s& tensor);
void cvimg2tensor_rga(const cv::Mat& img, uint32_t width, uint32_t height, tensor_data_s& tensor);

#endif // PREPROCESS_H
