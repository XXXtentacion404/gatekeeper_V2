#ifndef GATE_TRACKING_TYPES_H
#define GATE_TRACKING_TYPES_H

#include <stdint.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// ==================== 错误码 ====================
typedef enum {
    NN_SUCCESS = 0,
    NN_LOAD_MODEL_FAIL = -1,
    NN_RKNN_INIT_FAIL = -2,
    NN_RKNN_QUERY_FAIL = -3,
    NN_RKNN_INPUT_SET_FAIL = -4,
    NN_RKNN_RUNTIME_ERROR = -5,
    NN_IO_NUM_NOT_MATCH = -6,
    NN_RKNN_OUTPUT_GET_FAIL = -7,
    NN_RKNN_INPUT_ATTR_ERROR = -8,
    NN_RKNN_OUTPUT_ATTR_ERROR = -9,
    NN_RKNN_MODEL_NOT_LOAD = -10,
    NN_STOPED = -11,
    NN_TIMEOUT = -12,
} nn_error_e;

// ==================== 张量 ====================
typedef enum {
    NN_TENSORT_LAYOUT_UNKNOWN = 0,
    NN_TENSOR_NCHW = 1,
    NN_TENSOR_NHWC = 2,
    NN_TENSOR_OTHER = 3,
} tensor_layout_e;

typedef enum {
    NN_TENSOR_INT8 = 1,
    NN_TENSOR_UINT8 = 2,
    NN_TENSOR_FLOAT = 3,
    NN_TENSOR_FLOAT16 = 4,
} tensor_datatype_e;

static const int g_max_num_dims = 4;

typedef struct {
    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[4];
    uint32_t n_elems;
    uint32_t size;
    tensor_datatype_e type;
    tensor_layout_e layout;
    int32_t zp;
    float scale;
} tensor_attr_s;

typedef struct {
    tensor_attr_s attr;
    void *data;
} tensor_data_s;

static size_t nn_tensor_type_to_size(tensor_datatype_e type) {
    switch (type) {
        case NN_TENSOR_INT8: return sizeof(int8_t);
        case NN_TENSOR_UINT8: return sizeof(uint8_t);
        case NN_TENSOR_FLOAT: return sizeof(float);
        case NN_TENSOR_FLOAT16: return sizeof(uint16_t);
        default: return 0;
    }
}

static void nn_tensor_attr_to_cvimg_input_data(const tensor_attr_s &attr, tensor_data_s &data) {
    data.attr.n_dims = attr.n_dims;
    data.attr.index = 0;
    data.attr.type = NN_TENSOR_UINT8;
    data.attr.layout = NN_TENSOR_NHWC;
    if (attr.layout == NN_TENSOR_NCHW) {
        data.attr.dims[0] = attr.dims[0];
        data.attr.dims[1] = attr.dims[2];
        data.attr.dims[2] = attr.dims[3];
        data.attr.dims[3] = attr.dims[1];
    } else {
        for (int i = 0; i < 4; i++) data.attr.dims[i] = attr.dims[i];
    }
    data.attr.n_elems = data.attr.dims[0] * data.attr.dims[1] *
                        data.attr.dims[2] * data.attr.dims[3];
    data.attr.size = data.attr.n_elems * sizeof(uint8_t);
}

// ==================== 检测框 ====================
struct Detection {
    int class_id{0};
    std::string className{};
    float confidence{0.0f};
    cv::Scalar color{};
    cv::Rect box{};
};

typedef struct {
    float x, y, w, h, score;
    int class_id;
} nn_object_s;

// ==================== 绊线（虚拟跨线）====================
struct Tripwire {
    cv::Point2f p1;
    cv::Point2f p2;
    Tripwire() : p1(0,0), p2(0,0) {}
    Tripwire(float x1, float y1, float x2, float y2) : p1(x1,y1), p2(x2,y2) {}
    bool isValid() const { return (p1.x != p2.x || p1.y != p2.y); }
};

// ==================== 掩膜配置 ====================
struct CameraConfigEntry {
    std::string ip;
    std::string username;
    std::string password;
    int channel;
    int width;
    int height;
    std::vector<std::vector<cv::Point>> detection_zones;
    // ---- 判定线（区域边缘，点数不限）----
    // 旧格式 t1：2 点直线，作为区域的基准边，整体沿 x 平移 |entry_zone_depth| px。
    Tripwire entry_line;
    float entry_zone_depth = 0; // 区域沿 x 平移量(px，符号决定平移方向)，<1 = 无区域
    // 新格式 zin,/zout,：判定线 = 区域边缘折线（点数不限，边缘不一定是直线）。
    // 非空时优先于 entry_line；"区域边缘本身就是判定线"，区域 = 折线 ∪ 其 x 平移副本。
    std::vector<cv::Point2f> zone_line;
    // 本相机角色：-1 = 未指定（按 IP 末段判定：200=出口，201=入口），0 = 入口，1 = 出口。
    // 由 zin,/zout, 这两种 token 决定 —— 由用户选择哪台相机画入口、哪台画出口。
    int zone_role = -1;
};

// 日志宏
#define NN_LOG_DEBUG(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define NN_LOG_INFO(fmt, ...)  printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define NN_LOG_WARN(fmt, ...)  printf("[WARN] " fmt "\n", ##__VA_ARGS__)
#define NN_LOG_ERROR(fmt, ...) fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__)

#endif // GATE_TRACKING_TYPES_H
