#include "preprocess.h"
#include "types.h"

#include <cmath>
#include <cstdio>

namespace {
// letterbox 画布长边上限。与模型输入(640)一致时, 后续 resize 成为零拷贝直通;
// 即使模型输入不同, cvimg2tensor 仍会把画布缩放到模型输入, 坐标还原不受影响。
constexpr int kLetterboxMaxSide = 640;

// 计算画布几何: 画布宽高比 = wh_ratio(与模型输入一致), 内容等比缩放居中放置。
bool computeLetterboxGeom(const cv::Mat& img, float wh_ratio,
                          int& cw, int& ch, int& rw, int& rh,
                          int& pad_h, int& pad_v, float& scale) {
    if (img.channels() != 3 || img.empty() || wh_ratio < 1e-3f) return false;
    if (wh_ratio >= 1.0f) {
        cw = kLetterboxMaxSide;
        ch = (int)std::lround(kLetterboxMaxSide / wh_ratio);
    } else {
        ch = kLetterboxMaxSide;
        cw = (int)std::lround(kLetterboxMaxSide * wh_ratio);
    }
    const double s = std::min((double)cw / img.cols, (double)ch / img.rows);
    rw = std::max(1, (int)std::lround(img.cols * s));
    rh = std::max(1, (int)std::lround(img.rows * s));
    pad_h = (cw - rw) / 2;
    pad_v = (ch - rh) / 2;
    scale = (float)(1.0 / s);
    return true;
}
}  // namespace

// 先缩后补: 直接把原图缩到 640 量级的画布内(内容 640x360), 再补边到画布。
// 旧实现先把 1080p 补边成 1920x1920 再缩放, 像素处理量浪费约 3.4 倍。
// 坐标还原: x_orig = (x_canvas - pad_h_or_v) * scale (仅在补边轴上减 pad)。
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio) {
    LetterBoxInfo info;
    int cw, ch, rw, rh, pad_h, pad_v;
    float scale;
    if (!computeLetterboxGeom(img, wh_ratio, cw, ch, rw, rh, pad_h, pad_v, scale))
        return info;
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(rw, rh), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, img_letterbox, pad_v, ch - rh - pad_v,
                       pad_h, cw - rw - pad_h,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    info.hor = pad_h > 0;
    info.pad = info.hor ? pad_h : pad_v;
    info.scale = scale;
    info.rga = false;
    return info;
}

void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor) {
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    if (img_rgb.cols == (int)width && img_rgb.rows == (int)height) {
        // 画布已是模型输入尺寸, 跳过 resize 直通拷贝
        memcpy(tensor.data, img_rgb.data, tensor.attr.size);
        return;
    }
    cv::Mat img_resized;
    cv::resize(img_rgb, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    memcpy(tensor.data, img_resized.data, tensor.attr.size);
}

// ---------------------------------------------------------------------------
// RGA 硬件加速: 用 RK3588 的 2D 图形加速器一次完成 缩放+BGR->RGB+放置到画布,
// CPU 只做一次 640x640 画布的 memcpy(或者 Preprocess 直接把画布拷入 tensor)。
// 头文件缺失或 RGA 调用失败时自动回退 OpenCV 路径, 不影响功能。
// ---------------------------------------------------------------------------
#if defined(HAS_RGA)
#include <im2d.hpp>
#include <rga.h>
#endif

LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio) {
    LetterBoxInfo info;
#ifdef HAS_RGA
    int cw, ch, rw, rh, pad_h, pad_v;
    float scale;
    if (!computeLetterboxGeom(img, wh_ratio, cw, ch, rw, rh, pad_h, pad_v, scale))
        return info;
    img_letterbox.create(ch, cw, CV_8UC3);
    img_letterbox.setTo(0);   // 补边区固定为黑
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img.data, img.cols, img.rows,
                                              RK_FORMAT_BGR_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr(img_letterbox.data, cw, ch,
                                              RK_FORMAT_RGB_888);
    im_rect srect = {0, 0, img.cols, img.rows};
    im_rect drect = {pad_h, pad_v, rw, rh};
    rga_buffer_t pat = {};
    im_rect prect = {};
    IM_STATUS st = improcess(src, dst, pat, srect, drect, prect, IM_SYNC);
    // 成功 = IM_STATUS_SUCCESS(1) / IM_STATUS_NOERROR(2)，新旧头文件均 >=1；
    // 老板子头文件没有 IM_OK 别名，用数值判断兼容两版
    if (st < IM_STATUS_SUCCESS) {
        fprintf(stderr, "[PREPROCESS] RGA improcess failed (st=%d), fallback to OpenCV\n", (int)st);
        return LetterBoxInfo{};   // RGA 失败 → 调用方回退 OpenCV
    }
    info.hor = pad_h > 0;
    info.pad = info.hor ? pad_h : pad_v;
    info.scale = scale;
    info.rga = true;   // 画布已是 RGB, 调用方可直接 memcpy 进 tensor
    return info;
#else
    (void)img_letterbox; (void)wh_ratio;
    return info;   // 未启用 RGA, 调用方回退 OpenCV
#endif
}

void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor) {
    cvimg2tensor(img, width, height, tensor);
}
