#include "transformer_model.h"
#include "utils.h"
#include <memory.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string>
#include <cassert>


#ifndef HOLD_INTERMEDIATE

ggml_tensor* transformer_inference(dpu_set_t* dpu_set, ggml_context* ctx, Transformer* model_weight, ggml_tensor* input, TransformerParams& transformer_params)
{
    size_t    data_size = 1024ll*1024ll*1024ll*4ll;
    struct ggml_init_params data_ctx_params = {
            /*.mem_size   =*/ data_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx0 = ggml_init(data_ctx_params);

    size_t    layer_output_size = 1024ll*1024ll*1024ll*4ll;
    struct ggml_init_params layer_output_ctx_params = {
            /*.mem_size   =*/ layer_output_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
    };
    struct ggml_context * layer_output_ctx = ggml_init(layer_output_ctx_params);

    ggml_tensor* tmp_layer_input = input;
    for(int i=0; i<model_weight->layer_num; ++i)
    {
        ggml_tensor* tmp_layer_output = pim_lut_transformer_layer(dpu_set, ctx0, model_weight->layer_params[i], tmp_layer_input, transformer_params);
        if(i < model_weight->layer_num-1)
        {    
            ggml_free(layer_output_ctx);
            layer_output_ctx = ggml_init(layer_output_ctx_params);
            tmp_layer_input = ggml_new_tensor_2d(layer_output_ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim, transformer_params.attention_params.n);
        }
        else
            tmp_layer_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim, transformer_params.attention_params.n);
        memcpy(((float*)tmp_layer_input->data), ((float*)tmp_layer_output->data), sizeof(float) * transformer_params.attention_params.token_dim * transformer_params.attention_params.n);

        ggml_free(ctx0);
        ctx0 = ggml_init(data_ctx_params);
    }
    return tmp_layer_input;
}

#else

ggml_tensor* transformer_inference(dpu_set_t* dpu_set, ggml_context* ctx, Transformer* model_weight, ggml_tensor* input, TransformerParams& transformer_params)
{
    ggml_tensor* tmp_layer_input = input;
    ggml_tensor* tmp_layer_output = nullptr;
    for(int i=0; i<model_weight->layer_num; ++i)
    {
        tmp_layer_output = pim_lut_transformer_layer(dpu_set, ctx, model_weight->layer_params[i], tmp_layer_input, transformer_params);
        tmp_layer_input = tmp_layer_output;
    }
    return tmp_layer_output;
}

#endif