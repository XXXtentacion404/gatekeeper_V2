#ifndef ENGINE_H
#define ENGINE_H

#include "types.h"
#include <vector>
#include <memory>

class NNEngine {
public:
    virtual ~NNEngine() {}
    virtual nn_error_e LoadModelFile(const char *model_file) = 0;
    // 绑定推理核（在 LoadModelFile 之前调用生效）：
    // RK3588 三核 NPU，0/1/2 = NPU0/1/2，-1 = 内部自动轮询（默认）
    virtual void setCoreIndex(int core_index) {}
    virtual const std::vector<tensor_attr_s> &GetInputShapes() = 0;
    virtual const std::vector<tensor_attr_s> &GetOutputShapes() = 0;
    virtual nn_error_e Run(std::vector<tensor_data_s> &inputs, 
                          std::vector<tensor_data_s> &outputs, bool want_float) = 0;
};

std::shared_ptr<NNEngine> CreateRKNNEngine();

#endif // ENGINE_H
