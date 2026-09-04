#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <stdint.h>
#include "types.h"

namespace yolo {
    int GetConvDetectionResult(float **pBlob, std::vector<float> &DetectiontRects);
    int GetConvDetectionResultInt8(int8_t **pBlob, std::vector<int> &qnt_zp,
                                   std::vector<float> &qnt_scale,
                                   std::vector<float> &DetectiontRects);
}

#endif // POSTPROCESS_H
