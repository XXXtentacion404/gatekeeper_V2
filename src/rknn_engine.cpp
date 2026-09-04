#include "engine.h"
#include "rknn_engine.h"
#include "engine_helper.h"

#include <string.h>
#include <rknn_api.h>
#include <cstdlib>
#include <atomic>

static const int g_max_io_num = 10;
static std::atomic<unsigned int> g_next_npu_core{0};

nn_error_e RKEngine::LoadModelFile(const char *model_file) {
    int model_len = 0;
    auto model = load_model(model_file, &model_len);
    if (!model) { NN_LOG_ERROR("load model %s fail!", model_file); return NN_LOAD_MODEL_FAIL; }
    
    // 设置环境变量启用所有NPU核心（RK3588 3核）
    int ret = rknn_init(&rknn_ctx_, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) { NN_LOG_ERROR("rknn_init fail! ret=%d", ret); return NN_RKNN_INIT_FAIL; }
    static const rknn_core_mask core_masks[] = {
        RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2
    };
    // 优先用 setCoreIndex() 指定的核（按相机绑核，避免 4 实例撞 3 核）；
    // 未指定(-1)时回退到全局轮询
    unsigned int core_index;
    if (core_index_ >= 0 && core_index_ < (int)(sizeof(core_masks) / sizeof(core_masks[0]))) {
        core_index = (unsigned int)core_index_;
    } else {
        core_index = g_next_npu_core.fetch_add(1) % (sizeof(core_masks) / sizeof(core_masks[0]));
    }
    ret = rknn_set_core_mask(rknn_ctx_, core_masks[core_index]);
    if (ret != RKNN_SUCC) {
        NN_LOG_WARN("rknn_set_core_mask core=%u fail! ret=%d, use driver default",
                    core_index, ret);
    } else {
        NN_LOG_INFO("rknn_init success! bound to NPU Core%u%s",
                    core_index, core_index_ >= 0 ? " (pinned)" : "");
    }
    ctx_created_ = true;
    
    rknn_sdk_version version;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    NN_LOG_INFO("RKNN API v: %s, Driver v: %s", version.api_version, version.drv_version);
    
    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return NN_RKNN_QUERY_FAIL;
    input_num_ = io_num.n_input;
    output_num_ = io_num.n_output;
    NN_LOG_INFO("input num: %d, output num: %d", input_num_, output_num_);
    
    NN_LOG_INFO("input tensors:");
    for (int i = 0; i < (int)input_num_; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        print_tensor_attr(&attr);
        in_shapes_.push_back(rknn_tensor_attr_convert(attr));
    }
    
    NN_LOG_INFO("output tensors:");
    for (int i = 0; i < (int)output_num_; i++) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        print_tensor_attr(&attr);
        out_shapes_.push_back(rknn_tensor_attr_convert(attr));
    }
    return NN_SUCCESS;
}

const std::vector<tensor_attr_s>& RKEngine::GetInputShapes() { return in_shapes_; }
const std::vector<tensor_attr_s>& RKEngine::GetOutputShapes() { return out_shapes_; }

nn_error_e RKEngine::Run(std::vector<tensor_data_s> &inputs,
                        std::vector<tensor_data_s> &outputs, bool want_float) {
    if ((int)inputs.size() != (int)input_num_) return NN_IO_NUM_NOT_MATCH;
    if ((int)outputs.size() != (int)output_num_) return NN_IO_NUM_NOT_MATCH;
    
    rknn_input rknn_inputs[g_max_io_num];
    for (int i = 0; i < (int)inputs.size(); i++)
        rknn_inputs[i] = tensor_data_to_rknn_input(inputs[i]);
    
    int ret = rknn_inputs_set(rknn_ctx_, (uint32_t)inputs.size(), rknn_inputs);
    if (ret < 0) { NN_LOG_ERROR("rknn_inputs_set fail! %d", ret); return NN_RKNN_INPUT_SET_FAIL; }
    
    ret = rknn_run(rknn_ctx_, nullptr);
    if (ret < 0) { NN_LOG_ERROR("rknn_run fail! %d", ret); return NN_RKNN_RUNTIME_ERROR; }
    
    rknn_output rknn_outputs[g_max_io_num];
    memset(rknn_outputs, 0, sizeof(rknn_outputs));
    for (int i = 0; i < (int)output_num_; i++)
        rknn_outputs[i].want_float = want_float ? 1 : 0;
    
    ret = rknn_outputs_get(rknn_ctx_, output_num_, rknn_outputs, NULL);
    if (ret < 0) { NN_LOG_ERROR("rknn_outputs_get fail! %d", ret); return NN_RKNN_OUTPUT_GET_FAIL; }
    
    for (int i = 0; i < (int)output_num_; i++) {
        rknn_output_to_tensor_data(rknn_outputs[i], outputs[i]);
        free(rknn_outputs[i].buf);
    }
    return NN_SUCCESS;
}

RKEngine::~RKEngine() {
    if (ctx_created_) { rknn_destroy(rknn_ctx_); NN_LOG_INFO("rknn context destroyed!"); }
}

std::shared_ptr<NNEngine> CreateRKNNEngine() {
    return std::make_shared<RKEngine>();
}
