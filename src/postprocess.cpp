#include "postprocess.h"
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include "types.h"

namespace yolo {
    typedef struct {
        float xmin, ymin, xmax, ymax;
        float score;
        int classId;
    } DetectRect;
    
    static int input_w = 640;
    static int input_h = 640;
    static float objectThreshold = 0.35f;
    static float nmsThreshold = 0.5f;
    static int headNum = 3;
    static int class_num = 2;
    static int strides[3] = {8, 16, 32};
    static int mapSize[3][2] = {{80, 80}, {40, 40}, {20, 20}};

    static inline float fast_exp(float x) {
        union { uint32_t i; float f; } v;
        v.i = (uint32_t)(12102203.1616540672f * x + 1064807160.56887296f);
        return v.f;
    }

    static float sigmoid(float x) { return 1.0f / (1.0f + fast_exp(-x)); }

    static inline float IOU(float x1, float y1, float x2, float y2,
                           float x3, float y3, float x4, float y4) {
        float ix = std::max(x1, x3), iy = std::max(y1, y3);
        float ax = std::min(x2, x4), ay = std::min(y2, y4);
        float iw = std::max(0.0f, ax - ix), ih = std::max(0.0f, ay - iy);
        float inter = iw * ih;
        float area1 = (x2 - x1) * (y2 - y1);
        float area2 = (x4 - x3) * (y4 - y3);
        return inter / (area1 + area2 - inter + 1e-6f);
    }

    static float DeQnt2F32(int8_t qnt, int zp, float scale) {
        return ((float)qnt - (float)zp) * scale;
    }

    static std::vector<float> GenerateMeshgrid() {
        std::vector<float> meshgrid;
        for (int idx = 0; idx < headNum; idx++) {
            for (int i = 0; i < mapSize[idx][0]; i++)
                for (int j = 0; j < mapSize[idx][1]; j++) {
                    meshgrid.push_back((float)j + 0.5f);
                    meshgrid.push_back((float)i + 0.5f);
                }
        }
        return meshgrid;
    }

    int GetConvDetectionResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp,
                                   std::vector<float> &qnt_scale,
                                   std::vector<float> &DetectiontRects) {
        static auto meshgrid = GenerateMeshgrid();
        int gridIndex = -2;
        float xmin, ymin, xmax, ymax, cls_val, cls_max;
        int cls_index;
        std::vector<DetectRect> detectRects;

        for (int idx = 0; idx < headNum; idx++) {
            int8_t *reg = (int8_t *)pBlob[idx * 2 + 0];
            int8_t *cls = (int8_t *)pBlob[idx * 2 + 1];
            int qz_reg = qnt_zp[idx * 2 + 0], qz_cls = qnt_zp[idx * 2 + 1];
            float qs_reg = qnt_scale[idx * 2 + 0], qs_cls = qnt_scale[idx * 2 + 1];
            int stride = strides[idx];

            for (int h = 0; h < mapSize[idx][0]; h++) {
                for (int w = 0; w < mapSize[idx][1]; w++) {
                    gridIndex += 2;
                    cls_max = 0; cls_index = 0;
                    for (int cl = 0; cl < class_num; cl++) {
                        cls_val = sigmoid(DeQnt2F32(
                            cls[cl * mapSize[idx][0] * mapSize[idx][1] + h * mapSize[idx][1] + w],
                            qz_cls, qs_cls));
                        if (cl == 0 || cls_val > cls_max) { cls_max = cls_val; cls_index = cl; }
                    }
                    if (cls_max > objectThreshold) {
                        int base = h * mapSize[idx][1] + w;
                        xmin = (meshgrid[gridIndex] - DeQnt2F32(reg[0 * mapSize[idx][0] * mapSize[idx][1] + base], qz_reg, qs_reg)) * stride;
                        ymin = (meshgrid[gridIndex+1] - DeQnt2F32(reg[1 * mapSize[idx][0] * mapSize[idx][1] + base], qz_reg, qs_reg)) * stride;
                        xmax = (meshgrid[gridIndex] + DeQnt2F32(reg[2 * mapSize[idx][0] * mapSize[idx][1] + base], qz_reg, qs_reg)) * stride;
                        ymax = (meshgrid[gridIndex+1] + DeQnt2F32(reg[3 * mapSize[idx][0] * mapSize[idx][1] + base], qz_reg, qs_reg)) * stride;
                        xmin = std::max(0.0f, xmin); ymin = std::max(0.0f, ymin);
                        xmax = std::min((float)input_w, xmax); ymax = std::min((float)input_h, ymax);
                        if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h) {
                            DetectRect tmp;
                            tmp.xmin = xmin / input_w; tmp.ymin = ymin / input_h;
                            tmp.xmax = xmax / input_w; tmp.ymax = ymax / input_h;
                            tmp.classId = cls_index; tmp.score = cls_max;
                            detectRects.push_back(tmp);
                        }
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](const DetectRect& a, const DetectRect& b) { return a.score > b.score; });

        for (int i = 0; i < (int)detectRects.size(); ++i) {
            if (detectRects[i].classId != -1) {
                DetectiontRects.push_back((float)detectRects[i].classId);
                DetectiontRects.push_back(detectRects[i].score);
                DetectiontRects.push_back(detectRects[i].xmin);
                DetectiontRects.push_back(detectRects[i].ymin);
                DetectiontRects.push_back(detectRects[i].xmax);
                DetectiontRects.push_back(detectRects[i].ymax);
                for (int j = i + 1; j < (int)detectRects.size(); ++j) {
                    if (IOU(detectRects[i].xmin, detectRects[i].ymin, detectRects[i].xmax, detectRects[i].ymax,
                            detectRects[j].xmin, detectRects[j].ymin, detectRects[j].xmax, detectRects[j].ymax) > nmsThreshold)
                        detectRects[j].classId = -1;
                }
            }
        }
        return 0;
    }

    int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects) {
        static auto meshgrid = GenerateMeshgrid();
        int gridIndex = -2;
        float xmin, ymin, xmax, ymax, cls_val, cls_max;
        int cls_index;
        std::vector<DetectRect> detectRects;

        for (int idx = 0; idx < headNum; idx++) {
            float *reg = (float *)pBlob[idx * 2 + 0];
            float *cls = (float *)pBlob[idx * 2 + 1];
            int stride = strides[idx];

            for (int h = 0; h < mapSize[idx][0]; h++) {
                for (int w = 0; w < mapSize[idx][1]; w++) {
                    gridIndex += 2;
                    cls_max = 0; cls_index = 0;
                    for (int cl = 0; cl < class_num; cl++) {
                        cls_val = sigmoid(cls[cl * mapSize[idx][0] * mapSize[idx][1] + h * mapSize[idx][1] + w]);
                        if (cl == 0 || cls_val > cls_max) { cls_max = cls_val; cls_index = cl; }
                    }
                    if (cls_max > objectThreshold) {
                        int base = h * mapSize[idx][1] + w;
                        xmin = (meshgrid[gridIndex] - reg[0 * mapSize[idx][0] * mapSize[idx][1] + base]) * stride;
                        ymin = (meshgrid[gridIndex+1] - reg[1 * mapSize[idx][0] * mapSize[idx][1] + base]) * stride;
                        xmax = (meshgrid[gridIndex] + reg[2 * mapSize[idx][0] * mapSize[idx][1] + base]) * stride;
                        ymax = (meshgrid[gridIndex+1] + reg[3 * mapSize[idx][0] * mapSize[idx][1] + base]) * stride;
                        xmin = std::max(0.0f, xmin); ymin = std::max(0.0f, ymin);
                        xmax = std::min((float)input_w, xmax); ymax = std::min((float)input_h, ymax);
                        if (xmin >= 0 && ymin >= 0 && xmax <= input_w && ymax <= input_h) {
                            DetectRect tmp;
                            tmp.xmin = xmin / input_w; tmp.ymin = ymin / input_h;
                            tmp.xmax = xmax / input_w; tmp.ymax = ymax / input_h;
                            tmp.classId = cls_index; tmp.score = cls_max;
                            detectRects.push_back(tmp);
                        }
                    }
                }
            }
        }

        std::sort(detectRects.begin(), detectRects.end(),
                  [](const DetectRect& a, const DetectRect& b) { return a.score > b.score; });

        for (int i = 0; i < (int)detectRects.size(); ++i) {
            if (detectRects[i].classId != -1) {
                DetectiontRects.push_back((float)detectRects[i].classId);
                DetectiontRects.push_back(detectRects[i].score);
                DetectiontRects.push_back(detectRects[i].xmin);
                DetectiontRects.push_back(detectRects[i].ymin);
                DetectiontRects.push_back(detectRects[i].xmax);
                DetectiontRects.push_back(detectRects[i].ymax);
                for (int j = i + 1; j < (int)detectRects.size(); ++j) {
                    if (IOU(detectRects[i].xmin, detectRects[i].ymin, detectRects[i].xmax, detectRects[i].ymax,
                            detectRects[j].xmin, detectRects[j].ymin, detectRects[j].xmax, detectRects[j].ymax) > nmsThreshold)
                        detectRects[j].classId = -1;
                }
            }
        }
        return 0;
    }
}
