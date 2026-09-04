#ifndef RKNN_ENGINE_H
#define RKNN_ENGINE_H

#include "engine.h"
#include <vector>
#include <rknn_api.h>

class RKEngine : public NNEngine {
public:
    RKEngine() : rknn_ctx_(0), ctx_created_(false), input_num_(0), output_num_(0) {}
    ~RKEngine() override;

    nn_error_e LoadModelFile(const char *model_file) override;
    void setCoreIndex(int core_index) override { core_index_ = core_index; }
    const std::vector<tensor_attr_s> &GetInputShapes() override;
    const std::vector<tensor_attr_s> &GetOutputShapes() override;
    nn_error_e Run(std::vector<tensor_data_s> &inputs,
                  std::vector<tensor_data_s> &outputs, bool want_float) override;

private:
    rknn_context rknn_ctx_;
    bool ctx_created_;
    uint32_t input_num_;
    uint32_t output_num_;
    int core_index_ = -1;   // -1 = 自动轮询；0/1/2 = 固定 NPU 核
    std::vector<tensor_attr_s> in_shapes_;
    std::vector<tensor_attr_s> out_shapes_;
};

#endif // RKNN_ENGINE_H
