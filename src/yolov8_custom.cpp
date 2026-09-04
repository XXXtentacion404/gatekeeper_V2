#include "yolov8_custom.h"
#include "postprocess.h"
#include <atomic>

static std::vector<std::string> g_classes = {"people", "umbrella"};

Yolov8Custom::Yolov8Custom() {
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    want_float_ = false;
    ready_ = false;
}

Yolov8Custom::~Yolov8Custom() {
    std::lock_guard<std::mutex> lock(model_mutex_);
    if (input_tensor_.data != nullptr) {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    for (auto &tensor : output_tensors_) {
        if (tensor.data != nullptr) {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
}

nn_error_e Yolov8Custom::LoadModel(const char *model_path, int core_index) {
    std::lock_guard<std::mutex> lock(model_mutex_);

    engine_->setCoreIndex(core_index);   // -1 = 自动轮询
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS) return NN_LOAD_MODEL_FAIL;

    auto input_shapes = engine_->GetInputShapes();
    if (input_shapes.size() != 1) return NN_RKNN_INPUT_ATTR_ERROR;

    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);
    if (!input_tensor_.data) return NN_RKNN_INPUT_SET_FAIL;
    
    memset(input_tensor_.data, 0, input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();
    if (output_shapes.size() != 6) return NN_RKNN_OUTPUT_ATTR_ERROR;

    want_float_ = (output_shapes[0].type == NN_TENSOR_FLOAT16);
    
    output_tensors_.clear();
    out_zps_.clear();
    out_scales_.clear();
    
    for (int i = 0; i < (int)output_shapes.size(); i++) {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        tensor.attr.type = want_float_ ? NN_TENSOR_FLOAT : output_shapes[i].type;
        tensor.attr.index = 0;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        if (!tensor.data) return NN_RKNN_OUTPUT_GET_FAIL;
        memset(tensor.data, 0, tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }
    
    ready_ = true;
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox) {
    if (!ready_) return NN_RKNN_MODEL_NOT_LOAD;
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];

    if (process_type == "rga") {
        // RGA 硬件路径: 缩放+BGR->RGB+补边一次完成, 画布已是 RGB, 直接拷入
        // tensor, CPU 侧零图像处理。失败时回退 OpenCV 路径。
        LetterBoxInfo info = letterbox_rga(img, image_letterbox, wh_ratio);
        if (info.rga && !image_letterbox.empty()) {
            // 每进程只打一次, 确认 RGA 硬件路径真实生效(而非静默回退)
            static std::atomic<bool> rga_ok_logged{false};
            if (!rga_ok_logged.exchange(true))
                fprintf(stderr, "[PREPROCESS] RGA hardware path active\n");
            letterbox_info_ = info;
            memcpy(input_tensor_.data, image_letterbox.data, input_tensor_.attr.size);
            return NN_SUCCESS;
        }
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2],
                     input_tensor_.attr.dims[1], input_tensor_);
        return NN_SUCCESS;
    }
    if (process_type == "opencv") {
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }
    return NN_SUCCESS;
}

nn_error_e Yolov8Custom::Inference() {
    if (!ready_) return NN_RKNN_MODEL_NOT_LOAD;
    std::lock_guard<std::mutex> lock(model_mutex_);
    std::vector<tensor_data_s> inputs = {input_tensor_};
    return engine_->Run(inputs, output_tensors_, want_float_);
}

nn_error_e Yolov8Custom::Postprocess(const cv::Mat &img, std::vector<Detection> &objects) {
    if (!ready_) return NN_RKNN_MODEL_NOT_LOAD;
    std::lock_guard<std::mutex> lock(model_mutex_);
    
    void *output_data[6];
    for (int i = 0; i < 6; i++) output_data[i] = output_tensors_[i].data;
    
    std::vector<float> DetectiontRects;
    if (want_float_) {
        yolo::GetConvDetectionResult((float **)output_data, DetectiontRects);
    } else {
        yolo::GetConvDetectionResultInt8((int8_t **)output_data, out_zps_, out_scales_, DetectiontRects);
    }
    
    int img_width = img.cols;
    int img_height = img.rows;
    objects.clear();
    
    for (size_t i = 0; i < DetectiontRects.size(); i += 6) {
        if (i + 5 >= DetectiontRects.size()) break;
        int classId = static_cast<int>(DetectiontRects[i]);
        float conf = DetectiontRects[i + 1];
        if (conf < 0.25f) continue;
        
        int xmin = static_cast<int>(DetectiontRects[i + 2] * img_width + 0.5f);
        int ymin = static_cast<int>(DetectiontRects[i + 3] * img_height + 0.5f);
        int xmax = static_cast<int>(DetectiontRects[i + 4] * img_width + 0.5f);
        int ymax = static_cast<int>(DetectiontRects[i + 5] * img_height + 0.5f);
        
        xmin = std::max(0, std::min(xmin, img_width - 1));
        ymin = std::max(0, std::min(ymin, img_height - 1));
        xmax = std::max(0, std::min(xmax, img_width - 1));
        ymax = std::max(0, std::min(ymax, img_height - 1));
        
        if (xmax <= xmin || ymax <= ymin) continue;
        
        Detection result;
        result.class_id = classId;
        result.confidence = conf;
        result.color = cv::Scalar(0, 255, 0);
        result.className = classId < (int)g_classes.size() ? g_classes[classId] : "unknown";
        result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
        objects.push_back(result);
    }
    return NN_SUCCESS;
}

void Yolov8Custom::LetterboxDecode(std::vector<Detection> &objects, bool hor, int pad, float scale) {
    // 画布坐标 -> 原图坐标: 补边轴先减 pad, 两轴均乘 scale。
    // (画布内容只有 640x360, 1080p 下 scale=3.0)
    for (auto &obj : objects) {
        float x = (float)obj.box.x;
        float y = (float)obj.box.y;
        if (hor) x -= (float)pad;
        else     y -= (float)pad;
        obj.box.x = cvRound(x * scale);
        obj.box.y = cvRound(y * scale);
        obj.box.width  = std::max(1, cvRound((float)obj.box.width * scale));
        obj.box.height = std::max(1, cvRound((float)obj.box.height * scale));
    }
}

nn_error_e Yolov8Custom::Run(const cv::Mat &img, std::vector<Detection> &objects) {
    if (!ready_) return NN_RKNN_MODEL_NOT_LOAD;
    cv::Mat image_letterbox;
    auto ret = Preprocess(img, "rga", image_letterbox);
    if (ret != NN_SUCCESS) return ret;
    ret = Inference();
    if (ret != NN_SUCCESS) return ret;
    ret = Postprocess(image_letterbox, objects);
    if (ret != NN_SUCCESS) return ret;
    LetterboxDecode(objects, letterbox_info_.hor, letterbox_info_.pad,
                    letterbox_info_.scale);
    return NN_SUCCESS;
}
