#pragma once
#include "defines.h"
#include "ggml.h"
#include <vector>
extern "C"{
    #include <dpu.h>
}


struct TransformerLayer
{
    ggml_context* ctx;

    // attention norm
    struct ggml_tensor* attention_norm;
    struct ggml_tensor* attention_norm_bias;

#ifndef SEPARATE
    // qkv lut
    struct ggml_tensor* qkv_centroid; 
    struct ggml_tensor* qkv_lut_table;
    struct ggml_tensor* qkv_bias;
#else
    // qkv lut
    struct ggml_tensor* q_centroid; 
    struct ggml_tensor* q_lut_table;
    struct ggml_tensor* q_bias;

    struct ggml_tensor* k_centroid; 
    struct ggml_tensor* k_lut_table;
    struct ggml_tensor* k_bias;

    struct ggml_tensor* v_centroid; 
    struct ggml_tensor* v_lut_table;
    struct ggml_tensor* v_bias;
#endif

    // o lut
    struct ggml_tensor* o_centroid;
    struct ggml_tensor* o_lut_table;
    struct ggml_tensor* o_bias;

    // ffn norm
    struct ggml_tensor* ffn_norm;
    struct ggml_tensor* ffn_norm_bias;

    // ffn1 lut
    struct ggml_tensor* ffn1_centroid;
    struct ggml_tensor* ffn1_lut_table;
    struct ggml_tensor* ffn1_bias;

    // ffn2 lut
    struct ggml_tensor* ffn2_centroid;
    struct ggml_tensor* ffn2_lut_table;
    struct ggml_tensor* ffn2_bias;
};


ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* norm_weight, ggml_tensor* input, AttentionParams& attention_params);


ggml_tensor* norm(ggml_context* ctx, ggml_tensor* norm_weight, ggml_tensor* norm_bias, ggml_tensor* input, AttentionParams& attention_params);


ggml_tensor* pim_lut_attention(ggml_context* ctx, ggml_tensor* lut_qkv, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V,  AttentionParams& attention_params, uint32_t feature_mtile_size);


ggml_tensor* silu(ggml_context* ctx, ggml_tensor* input, uint32_t num_threads);


ggml_tensor* gelu(ggml_context* ctx, ggml_tensor* input, uint32_t num_threads);


ggml_tensor* pim_lut_transformer_layer(dpu_set_t* dpu_set, ggml_context* ctx, TransformerLayer* layer_weight, ggml_tensor* input, TransformerParams& transformer_params);

