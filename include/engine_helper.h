#ifndef ENGINE_HELPER_H
#define ENGINE_HELPER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <rknn_api.h>
#include "types.h"

static unsigned char *load_model(const char *filename, int *model_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { NN_LOG_ERROR("fopen %s fail!", filename); return nullptr; }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != (int)fread(model, 1, model_len, fp)) {
        NN_LOG_ERROR("fread %s fail!", filename);
        free(model); fclose(fp); return nullptr;
    }
    *model_size = model_len;
    fclose(fp);
    return model;
}

static void print_tensor_attr(rknn_tensor_attr *attr) {
    NN_LOG_INFO("  index=%d, name=%s, n_dims=%d, dims=[%d,%d,%d,%d], "
                "n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, zp=%d, scale=%f",
                attr->index, attr->name, attr->n_dims,
                attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
                attr->n_elems, attr->size,
                get_format_string(attr->fmt), get_type_string(attr->type),
                get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static tensor_layout_e rknn_layout_convert(tensor_layout_e fmt) {
    return fmt;
}

static rknn_tensor_format rknn_layout_convert_rk(tensor_layout_e fmt) {
    switch (fmt) {
        case NN_TENSOR_NCHW: return RKNN_TENSOR_NCHW;
        case NN_TENSOR_NHWC: return RKNN_TENSOR_NHWC;
        default: NN_LOG_ERROR("unsupported layout"); exit(1);
    }
}

static rknn_tensor_type rknn_type_convert_rk(tensor_datatype_e type) {
    switch (type) {
        case NN_TENSOR_UINT8: return RKNN_TENSOR_UINT8;
        case NN_TENSOR_FLOAT: return RKNN_TENSOR_FLOAT32;
        default: NN_LOG_ERROR("unsupported type"); exit(1);
    }
}

static tensor_datatype_e rknn_type_convert(rknn_tensor_type type) {
    switch (type) {
        case RKNN_TENSOR_UINT8: return NN_TENSOR_UINT8;
        case RKNN_TENSOR_FLOAT32: return NN_TENSOR_FLOAT;
        case RKNN_TENSOR_INT8: return NN_TENSOR_INT8;
        case RKNN_TENSOR_FLOAT16: return NN_TENSOR_FLOAT16;
        default: NN_LOG_ERROR("unsupported rknn type"); exit(1);
    }
}

static tensor_layout_e rknn_layout_convert(rknn_tensor_format fmt) {
    switch (fmt) {
        case RKNN_TENSOR_NCHW: return NN_TENSOR_NCHW;
        case RKNN_TENSOR_NHWC: return NN_TENSOR_NHWC;
        default: return NN_TENSOR_OTHER;
    }
}

static tensor_attr_s rknn_tensor_attr_convert(const rknn_tensor_attr &attr) {
    tensor_attr_s shape;
    shape.n_dims = attr.n_dims;
    shape.index = attr.index;
    for (int i = 0; i < attr.n_dims; ++i) shape.dims[i] = attr.dims[i];
    shape.size = attr.size;
    shape.n_elems = attr.n_elems;
    shape.layout = rknn_layout_convert(attr.fmt);
    shape.type = rknn_type_convert(attr.type);
    shape.zp = attr.zp;
    shape.scale = attr.scale;
    return shape;
}

static rknn_input tensor_data_to_rknn_input(const tensor_data_s &data) {
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = data.attr.index;
    input.type = rknn_type_convert_rk(data.attr.type);
    input.size = data.attr.size;
    input.fmt = rknn_layout_convert_rk(data.attr.layout);
    input.buf = data.data;
    return input;
}

static void rknn_output_to_tensor_data(const rknn_output &output, tensor_data_s &data) {
    data.attr.index = output.index;
    data.attr.size = output.size;
    memcpy(data.data, output.buf, output.size);
}

#endif // ENGINE_HELPER_H
