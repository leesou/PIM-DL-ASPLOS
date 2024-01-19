#pragma once
#include "defines.h"
#include "ggml.h"
#include <vector>
extern "C"{
    #include <dpu.h>
}
#include "transformer_layer.h"


struct Transformer
{
    ggml_context* ctx;

    uint32_t layer_num;

    std::vector<TransformerLayer*> layer_params;
};


ggml_tensor* transformer_inference(dpu_set_t* dpu_set, ggml_context* ctx, Transformer* model_weight, ggml_tensor* input, TransformerParams& transformer_params);
