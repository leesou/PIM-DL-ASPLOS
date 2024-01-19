#include "transformer_layer.h"
#include "amm_host.h"
#include "dpu_common.h"
#include "utils.h"
#include <memory.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string>
#include <cassert>

#define USE_FLASH_ATTN

// modified from GGML
ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* norm_weight, ggml_tensor* input, AttentionParams& attention_params)
{
    ggml_tensor* normed_input_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, input->ne[0], input->ne[1], input->ne[2], input->ne[3]);

    const float eps = 1e-6f; // TODO: change to a parameter

    #pragma omp parallel for num_threads(attention_params.num_threads)
    for(uint32_t tmp_token=0; tmp_token<attention_params.n; ++tmp_token)
    {
        const float* tmp_token_ptr = ((float*)input->data) + tmp_token*attention_params.token_dim;

        double sum = 0.0;
        for(uint32_t tmp_dim=0; tmp_dim<attention_params.token_dim; ++tmp_dim)
        {
            sum += (double)(tmp_token_ptr[tmp_dim] * tmp_token_ptr[tmp_dim]);
        }

        const float mean = sum / attention_params.token_dim;

        float* tmp_normed_token_ptr = ((float*)normed_input_tensor->data) + tmp_token*attention_params.token_dim;
        memcpy(tmp_normed_token_ptr, tmp_token_ptr, sizeof(float)*attention_params.token_dim);

        const float scale = 1.0f / sqrtf(mean + eps);
        for(uint32_t tmp_dim=0; tmp_dim<attention_params.token_dim; ++tmp_dim)
        {
            tmp_normed_token_ptr[tmp_dim] *= scale;
            tmp_normed_token_ptr[tmp_dim] *= ((float*)norm_weight->data)[tmp_dim];
        }
    }

    return normed_input_tensor;
}


// modified from GGML
ggml_tensor* norm(ggml_context* ctx, ggml_tensor* norm_weight, ggml_tensor* norm_bias, ggml_tensor* input, AttentionParams& attention_params)
{
    ggml_tensor* normed_input_tensor = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, input->ne[0], input->ne[1], input->ne[2], input->ne[3]);

    const float eps = 1e-6f; // TODO: change to a parameter

    #pragma omp parallel for num_threads(attention_params.num_threads)
    for(uint32_t tmp_token=0; tmp_token<attention_params.n; ++tmp_token)
    {
        const float* tmp_token_ptr = ((float*)input->data) + tmp_token*attention_params.token_dim;

        double sum = 0.0;
        for(uint32_t tmp_dim=0; tmp_dim<attention_params.token_dim; ++tmp_dim)
        {
            sum += (double)(tmp_token_ptr[tmp_dim]);
        }

        float mean = sum / attention_params.token_dim;

        float* tmp_normed_token_ptr = ((float*)normed_input_tensor->data) + tmp_token*attention_params.token_dim;

        double sum2 = 0.0;
        for(uint32_t tmp_dim=0; tmp_dim<attention_params.token_dim; ++tmp_dim)
        {
            float v = tmp_token_ptr[tmp_dim] - mean;
            tmp_normed_token_ptr[tmp_dim] = v;
            sum2 += (double)(v*v);
        }

        float variance = sum2 / attention_params.token_dim;
        const float scale = 1.0f / sqrtf(variance + eps);
        for(uint32_t tmp_dim=0; tmp_dim<attention_params.token_dim; ++tmp_dim)
        {
            tmp_normed_token_ptr[tmp_dim] *= scale;
            tmp_normed_token_ptr[tmp_dim] *= ((float*)norm_weight->data)[tmp_dim];
            tmp_normed_token_ptr[tmp_dim] += ((float*)norm_bias->data)[tmp_dim];
        }
    }

    return normed_input_tensor;
}


ggml_tensor* pim_lut_attention(ggml_context* ctx, ggml_tensor* lut_qkv, ggml_tensor* Q, ggml_tensor* K, ggml_tensor* V,  AttentionParams& attention_params, uint32_t feature_mtile_size)
{
#ifndef SEPARATE
    // step 1: reorder data
    #pragma omp parallel for num_threads(attention_params.num_threads)
    for(uint32_t tmp_dpu=0; tmp_dpu<attention_params.dpu_num; ++tmp_dpu)
    {
        uint32_t tmp_dpu_tensor_offset = tmp_dpu * attention_params.n_tile_size * attention_params.token_tile_size * 3;
        uint32_t tmp_n_tile_id = tmp_dpu / attention_params.lut_parallelism;
        uint32_t tmp_feature_tile_id = tmp_dpu % attention_params.lut_parallelism;

        // copy Q
        float* tmp_dpu_q_tile = ((float*)lut_qkv->data) + tmp_dpu_tensor_offset;
        uint32_t tmp_head_dim_offset = tmp_feature_tile_id * attention_params.token_tile_size % attention_params.head_dim;
        uint32_t tmp_head_offset = tmp_feature_tile_id * attention_params.token_tile_size / attention_params.head_dim;
        for(uint32_t tmp_dim=0; tmp_dim<attention_params.token_tile_size; tmp_dim+=feature_mtile_size)
        {
            uint32_t tmp_seq_offset = tmp_n_tile_id * attention_params.n_tile_size % attention_params.seq_len;
            uint32_t tmp_batch_offset = tmp_n_tile_id * attention_params.n_tile_size / attention_params.seq_len;
            for(uint32_t tmp_row=0; tmp_row<attention_params.n_tile_size; ++tmp_row)
            {
                uint32_t q_dpu_offset = tmp_dim * attention_params.n_tile_size
                                      + tmp_row * feature_mtile_size;
                uint32_t q_offset = tmp_head_dim_offset 
                                  + tmp_seq_offset * attention_params.head_dim
                                  + tmp_head_offset * attention_params.head_dim * attention_params.seq_len
                                  + tmp_batch_offset * attention_params.head_dim * attention_params.seq_len * attention_params.head_num;
                memcpy(((float*)Q->data)+q_offset, tmp_dpu_q_tile+q_dpu_offset, sizeof(float)*feature_mtile_size);

                tmp_seq_offset++;
                if(tmp_seq_offset >= attention_params.seq_len)
                {
                    tmp_seq_offset = 0;
                    tmp_batch_offset++;
                }
            }

            tmp_head_dim_offset += feature_mtile_size;
            if(tmp_head_dim_offset >= attention_params.head_dim)
            {
                tmp_head_dim_offset = 0;
                tmp_head_offset++;
            }
        }

        // copy K
        float* tmp_dpu_k_tile = tmp_dpu_q_tile + attention_params.n_tile_size * attention_params.token_tile_size;
        tmp_head_dim_offset = tmp_feature_tile_id * attention_params.token_tile_size % attention_params.head_dim;
        tmp_head_offset = tmp_feature_tile_id * attention_params.token_tile_size / attention_params.head_dim;
        for(uint32_t tmp_dim=0; tmp_dim<attention_params.token_tile_size; tmp_dim+=feature_mtile_size)
        {
            uint32_t tmp_seq_offset = tmp_n_tile_id * attention_params.n_tile_size % attention_params.seq_len;
            uint32_t tmp_batch_offset = tmp_n_tile_id * attention_params.n_tile_size / attention_params.seq_len;
            for(uint32_t tmp_row=0; tmp_row<attention_params.n_tile_size; ++tmp_row)
            {
                uint32_t k_dpu_offset = tmp_dim * attention_params.n_tile_size
                                      + tmp_row * feature_mtile_size;
                uint32_t k_offset = tmp_head_dim_offset 
                                  + tmp_seq_offset * attention_params.head_dim
                                  + tmp_head_offset * attention_params.head_dim * attention_params.seq_len
                                  + tmp_batch_offset * attention_params.head_dim * attention_params.seq_len * attention_params.head_num;
                memcpy(((float*)K->data)+k_offset, tmp_dpu_k_tile+k_dpu_offset, sizeof(float)*feature_mtile_size);

                tmp_seq_offset++;
                if(tmp_seq_offset >= attention_params.seq_len)
                {
                    tmp_seq_offset = 0;
                    tmp_batch_offset++;
                }
            }

            tmp_head_dim_offset += feature_mtile_size;
            if(tmp_head_dim_offset >= attention_params.head_dim)
            {
                tmp_head_dim_offset = 0;
                tmp_head_offset++;
            }
        }

        // copy V
        float* tmp_dpu_v_tile = tmp_dpu_k_tile + attention_params.n_tile_size * attention_params.token_tile_size;
        tmp_head_dim_offset = tmp_feature_tile_id * attention_params.token_tile_size % attention_params.head_dim;
        tmp_head_offset = tmp_feature_tile_id * attention_params.token_tile_size / attention_params.head_dim;
        for(uint32_t tmp_dim=0; tmp_dim<attention_params.token_tile_size; tmp_dim+=feature_mtile_size)
        {
            uint32_t tmp_seq_offset = tmp_n_tile_id * attention_params.n_tile_size % attention_params.seq_len;
            uint32_t tmp_batch_offset = tmp_n_tile_id * attention_params.n_tile_size / attention_params.seq_len;
            for(uint32_t tmp_row=0; tmp_row<attention_params.n_tile_size; ++tmp_row)
            {
                uint32_t v_dpu_offset = tmp_dim * attention_params.n_tile_size
                                      + tmp_row * feature_mtile_size;
                uint32_t v_offset = tmp_head_dim_offset 
                                  + tmp_seq_offset * attention_params.head_dim
                                  + tmp_head_offset * attention_params.head_dim * attention_params.seq_len
                                  + tmp_batch_offset * attention_params.head_dim * attention_params.seq_len * attention_params.head_num;
                
                for(uint32_t tmp_offset=0; tmp_offset<attention_params.token_tile_size; ++tmp_offset)
                {
                    ((float*)V->data)[v_offset + tmp_offset*attention_params.seq_len] = tmp_dpu_v_tile[v_dpu_offset + tmp_dim];
                }

                tmp_seq_offset++;
                if(tmp_seq_offset >= attention_params.seq_len)
                {
                    tmp_seq_offset = 0;
                }
            }

            tmp_head_dim_offset += feature_mtile_size;
            if(tmp_head_dim_offset >= attention_params.head_dim)
            {
                tmp_head_dim_offset = 0;
                tmp_head_offset++;
            }
        }
    }
#endif

    // step 2: attention
    ggml_cgraph gf = {};
    gf.n_threads = attention_params.num_threads;

#ifndef USE_FLASH_ATTN
    struct ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);
    struct ggml_tensor* KQ_scaled = ggml_scale(ctx, KQ, ggml_new_f32(ctx, 1.0f/sqrtf(float(attention_params.token_dim)/attention_params.head_num)));
    struct ggml_tensor* KQ_soft_max = ggml_soft_max(ctx, KQ_scaled);
    struct ggml_tensor* KQV = ggml_mul_mat(ctx, V, KQ_soft_max);
    struct ggml_tensor* KQV_merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);
    struct ggml_tensor* result = ggml_reshape_2d(ctx, ggml_cont(ctx, KQV_merged), attention_params.token_dim, attention_params.n);
#else
    struct ggml_tensor* KQV = ggml_flash_attn(ctx, Q, K, V, false);
    struct ggml_tensor* KQV_merged = ggml_permute(ctx, KQV, 0, 2, 1, 3);
    struct ggml_tensor* result = ggml_reshape_2d(ctx, ggml_cont(ctx, KQV_merged), attention_params.token_dim, attention_params.n);
#endif

    ggml_build_forward_expand(&gf, result);
    ggml_graph_compute(ctx, &gf);

    return result;
}


ggml_tensor* silu(ggml_context* ctx, ggml_tensor* input, uint32_t num_threads)
{
    ggml_cgraph gf = {};
    gf.n_threads = num_threads;

    ggml_tensor* result = ggml_silu_inplace(ctx, input);

    ggml_build_forward_expand(&gf, result);
    ggml_graph_compute(ctx, &gf);

    return result;
}


ggml_tensor* gelu(ggml_context* ctx, ggml_tensor* input, uint32_t num_threads)
{
    ggml_cgraph gf = {};
    gf.n_threads = num_threads;

    ggml_tensor* result = ggml_gelu_inplace(ctx, input);

    ggml_build_forward_expand(&gf, result);
    ggml_graph_compute(ctx, &gf);

    return result;
}


ggml_tensor* pim_lut_transformer_layer(dpu_set_t* dpu_set, ggml_context* ctx, TransformerLayer* layer_weight, ggml_tensor* input, TransformerParams& transformer_params)
{
#if defined(TRANSFORMER_BREAKDOWN) || defined(TRANSFORMER_BREAKDOWN_AMM)
    double time1, time2;
#endif

    ////////////////////////////////////////////////////////////////////////////////////

    AttentionParams& attn_params = transformer_params.attention_params;
    AMMParams& qkv_params = transformer_params.amm_param_list[0];
    AMMParams& o_params = transformer_params.amm_param_list[1];
    AMMParams& ffn1_params = transformer_params.amm_param_list[2];
    AMMParams& ffn2_params = transformer_params.amm_param_list[3];
#ifdef SEPARATE
    AMMParams& q_params = transformer_params.amm_param_list[4];
    AMMParams& k_params = transformer_params.amm_param_list[5];
    AMMParams& v_params = transformer_params.amm_param_list[6];
#endif
    std::string qkv_pim_bin = transformer_params.pim_binary_list[0];
    std::string o_pim_bin = transformer_params.pim_binary_list[1];
    std::string ffn1_pim_bin = transformer_params.pim_binary_list[2];
    std::string ffn2_pim_bin = transformer_params.pim_binary_list[3];
#ifdef SEPARATE
    std::string q_pim_bin = transformer_params.pim_binary_list[1];
    std::string k_pim_bin = transformer_params.pim_binary_list[1];
    std::string v_pim_bin = transformer_params.pim_binary_list[1];
#endif

#ifndef SEPARATE

    ////////////////////////////////////////////////////////////////////////////////////

#if defined(TRANSFORMER_BREAKDOWN) || defined(TRANSFORMER_BREAKDOWN_AMM)
    time1 = W_time();
#endif
    // step 1: QKV LUT
    struct ggml_tensor* lut_qkv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, attn_params.token_tile_size * 3, attn_params.n_tile_size, qkv_params.lut_params.dpu_num);
    load_binary(dpu_set, qkv_pim_bin);
    amm_host(dpu_set, ((float*)input->data), ((float*)layer_weight->qkv_centroid->data), ((lut_data_type*)layer_weight->qkv_lut_table->data), ((float*)layer_weight->qkv_bias->data), ((float*)lut_qkv->data), qkv_params.index_params, qkv_params.lut_params);
#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.amm_latency += time2 - time1;
#elif defined(TRANSFORMER_BREAKDOWN_AMM)
    time2 = W_time();
    transformer_amm_profiles.qkv_projection_latency += time2 - time1;
#endif

    ////////////////////////////////////////////////////////////////////////////////////

#ifdef TRANSFORMER_BREAKDOWN
    time1 = W_time();
#endif
    // step 2: Self Attention
    struct ggml_tensor* Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, attn_params.head_dim, attn_params.seq_len, attn_params.head_num, attn_params.batch_size);
    struct ggml_tensor* K = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, attn_params.head_dim, attn_params.seq_len, attn_params.head_num, attn_params.batch_size);
    struct ggml_tensor* V = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, attn_params.seq_len, attn_params.head_dim, attn_params.head_num, attn_params.batch_size);
    struct ggml_tensor* O_non_project = pim_lut_attention(ctx, lut_qkv, Q, K, V, attn_params, qkv_params.lut_params.feature_mtile_size);
#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.non_amm_latency += time2 - time1;
#endif

#else

    ////////////////////////////////////////////////////////////////////////////////////

#if defined(TRANSFORMER_BREAKDOWN) || defined(TRANSFORMER_BREAKDOWN_AMM)
    time1 = W_time();
#endif
    // step 1: QKV LUT
    struct ggml_tensor* lut_q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, attn_params.token_dim, attn_params.n);
    load_binary(dpu_set, q_pim_bin);
    amm_host(dpu_set, ((float*)input->data), ((float*)layer_weight->q_centroid->data), ((lut_data_type*)layer_weight->q_lut_table->data), ((float*)layer_weight->q_bias->data), ((float*)lut_q->data), q_params.index_params, q_params.lut_params);

    struct ggml_tensor* lut_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, attn_params.token_dim, attn_params.n);
    load_binary(dpu_set, k_pim_bin);
    amm_host(dpu_set, ((float*)input->data), ((float*)layer_weight->k_centroid->data), ((lut_data_type*)layer_weight->k_lut_table->data), ((float*)layer_weight->k_bias->data), ((float*)lut_k->data), k_params.index_params, k_params.lut_params);

    struct ggml_tensor* lut_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, attn_params.token_dim, attn_params.n);
    load_binary(dpu_set, v_pim_bin);
    amm_host(dpu_set, ((float*)input->data), ((float*)layer_weight->v_centroid->data), ((lut_data_type*)layer_weight->v_lut_table->data), ((float*)layer_weight->v_bias->data), ((float*)lut_v->data), v_params.index_params, v_params.lut_params);

#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.amm_latency += time2 - time1;
#elif defined(TRANSFORMER_BREAKDOWN_AMM)
    time2 = W_time();
    transformer_amm_profiles.qkv_projection_latency += time2 - time1;
#endif

    ////////////////////////////////////////////////////////////////////////////////////

#ifdef TRANSFORMER_BREAKDOWN
    time1 = W_time();
#endif
    // step 2: Self Attention
    struct ggml_tensor* Q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, attn_params.head_dim, attn_params.seq_len, attn_params.head_num, attn_params.batch_size);
    struct ggml_tensor* K = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, attn_params.head_dim, attn_params.seq_len, attn_params.head_num, attn_params.batch_size);
    struct ggml_tensor* V = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, attn_params.seq_len, attn_params.head_dim, attn_params.head_num, attn_params.batch_size);
    
    // reorder data
    #pragma omp parallel for num_threads(transformer_params.num_threads)
    for(uint32_t tmp_row=0; tmp_row<o_params.lut_params.n; ++tmp_row)
    {
        uint32_t tmp_n_tile_id = tmp_row / o_params.lut_params.n_stile_size;
        uint32_t intra_n_tile_id = tmp_row % o_params.lut_params.n_stile_size;
        
        uint32_t tmp_batch_id = tmp_row / attn_params.batch_size;
        uint32_t tmp_intra_batch_id = tmp_row % attn_params.batch_size;

        uint32_t tmp_head_id = 0;
        uint32_t tmp_intra_head_id = 0;

        for(uint32_t tmp_dim=0; tmp_dim<o_params.lut_params.output_feature_len; tmp_dim += o_params.lut_params.feature_stile_size)
        {
            uint32_t tmp_feature_tile_id = tmp_dim / o_params.lut_params.feature_stile_size;
            uint32_t tmp_dpu_tile_offset = tmp_n_tile_id * o_params.lut_params.n_stile_size * o_params.lut_params.output_feature_len
                                         + tmp_feature_tile_id * o_params.lut_params.n_stile_size * o_params.lut_params.feature_stile_size;

            for(uint32_t tmp_intra_dim=0; tmp_intra_dim<o_params.lut_params.feature_stile_size; tmp_intra_dim+=o_params.lut_params.feature_mtile_size)
            {
                uint32_t tmp_intra_feature_tile_id = tmp_intra_dim / o_params.lut_params.feature_mtile_size;
                uint32_t tmp_intra_dpu_tile_offset = tmp_intra_feature_tile_id * o_params.lut_params.n_stile_size * o_params.lut_params.feature_mtile_size
                                                   + intra_n_tile_id * o_params.lut_params.feature_mtile_size;

                for(uint32_t tmp_offset=0; tmp_offset<o_params.lut_params.feature_mtile_size; ++tmp_offset)
                {
                    uint32_t new_offset = tmp_batch_id*attn_params.head_dim*attn_params.seq_len*attn_params.head_num
                                        + tmp_head_id*attn_params.head_dim*attn_params.seq_len
                                        + tmp_intra_batch_id*attn_params.head_dim
                                        + tmp_intra_head_id;
                    
                    uint32_t new_offset1 = tmp_batch_id*attn_params.seq_len*attn_params.head_dim*attn_params.head_num
                                         + tmp_head_id*attn_params.seq_len*attn_params.head_dim
                                         + tmp_intra_head_id*attn_params.seq_len
                                         + tmp_intra_batch_id;

                    ((float*)Q->data)[new_offset] = ((float*)lut_q->data)[tmp_dpu_tile_offset + tmp_intra_dpu_tile_offset + tmp_offset];
                    ((float*)K->data)[new_offset] = ((float*)lut_k->data)[tmp_dpu_tile_offset + tmp_intra_dpu_tile_offset + tmp_offset];
                    ((float*)V->data)[new_offset1] = ((float*)lut_v->data)[tmp_dpu_tile_offset + tmp_intra_dpu_tile_offset + tmp_offset];
                
                    tmp_intra_head_id++;
                    if(tmp_intra_head_id==attn_params.head_dim)
                    {
                        tmp_intra_head_id = 0;
                        tmp_head_id++;
                    }
                }
            }
        }
    }

    struct ggml_tensor* O_non_project = pim_lut_attention(ctx, nullptr, Q, K, V, attn_params, qkv_params.lut_params.feature_mtile_size);
#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.non_amm_latency += time2 - time1;
#endif

#endif

    ////////////////////////////////////////////////////////////////////////////////////

#if defined(TRANSFORMER_BREAKDOWN) || defined(TRANSFORMER_BREAKDOWN_AMM)
    time1 = W_time();
#endif
    // step 3: O projection
    struct ggml_tensor* O = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, attn_params.token_dim, attn_params.n);
    load_binary(dpu_set, o_pim_bin);
    amm_host(dpu_set, ((float*)O_non_project->data), ((float*)layer_weight->o_centroid->data), ((lut_data_type*)layer_weight->o_lut_table->data), ((float*)layer_weight->o_bias->data), ((float*)O->data), o_params.index_params, o_params.lut_params);
#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.amm_latency += time2 - time1;
#elif defined(TRANSFORMER_BREAKDOWN_AMM)
    time2 = W_time();
    transformer_amm_profiles.o_projection_latency += time2 - time1;
#endif

    ////////////////////////////////////////////////////////////////////////////////////

#ifdef TRANSFORMER_BREAKDOWN
    time1 = W_time();
#endif
    struct ggml_tensor* ffn_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, o_params.lut_params.output_feature_len, o_params.lut_params.n);
    
    // reorder data
    #pragma omp parallel for num_threads(transformer_params.num_threads)
    for(uint32_t tmp_row=0; tmp_row<o_params.lut_params.n; ++tmp_row)
    {
        uint32_t tmp_n_tile_id = tmp_row / o_params.lut_params.n_stile_size;
        uint32_t intra_n_tile_id = tmp_row % o_params.lut_params.n_stile_size;

        for(uint32_t tmp_dim=0; tmp_dim<o_params.lut_params.output_feature_len; tmp_dim += o_params.lut_params.feature_stile_size)
        {
            uint32_t tmp_feature_tile_id = tmp_dim / o_params.lut_params.feature_stile_size;
            uint32_t tmp_dpu_tile_offset = tmp_n_tile_id * o_params.lut_params.n_stile_size * o_params.lut_params.output_feature_len
                                         + tmp_feature_tile_id * o_params.lut_params.n_stile_size * o_params.lut_params.feature_stile_size;

            for(uint32_t tmp_intra_dim=0; tmp_intra_dim<o_params.lut_params.feature_stile_size; tmp_intra_dim+=o_params.lut_params.feature_mtile_size)
            {
                uint32_t tmp_intra_feature_tile_id = tmp_intra_dim / o_params.lut_params.feature_mtile_size;
                uint32_t tmp_intra_dpu_tile_offset = tmp_intra_feature_tile_id * o_params.lut_params.n_stile_size * o_params.lut_params.feature_mtile_size
                                                   + intra_n_tile_id * o_params.lut_params.feature_mtile_size;

                memcpy((((float*)ffn_input->data) + tmp_row*o_params.lut_params.output_feature_len + tmp_dim + tmp_intra_dim),
                       (((float*)O->data) + tmp_dpu_tile_offset + tmp_intra_dpu_tile_offset),
                       sizeof(float) * o_params.lut_params.feature_mtile_size);
            }
        }
    }

    // step 4: Attention Norm
    struct ggml_tensor* normed_attention_output = norm(ctx, layer_weight->attention_norm, layer_weight->attention_norm_bias, ffn_input, attn_params);

    // step 5: Residual
    #pragma omp parallel for num_threads(transformer_params.num_threads)
    for(uint32_t tmp_row=0; tmp_row<o_params.lut_params.n; ++tmp_row)
    {
        for(uint32_t tmp_dim=0; tmp_dim<o_params.lut_params.output_feature_len; ++tmp_dim)
        {
            ((float*)ffn_input->data)[tmp_row*o_params.lut_params.output_feature_len + tmp_dim] 
            = ((float*)input->data)[tmp_row*o_params.lut_params.output_feature_len + tmp_dim]
            + ((float*)normed_attention_output->data)[tmp_row*o_params.lut_params.output_feature_len + tmp_dim];
        }
    }
#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.non_amm_latency += time2 - time1;
#endif

    ////////////////////////////////////////////////////////////////////////////////////

#if defined(TRANSFORMER_BREAKDOWN) || defined(TRANSFORMER_BREAKDOWN_AMM)
    time1 = W_time();
#endif
    // step 6: FFN1 Lut
    struct ggml_tensor* lut_ffn1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn1_params.lut_params.output_feature_len, ffn1_params.lut_params.n);
    load_binary(dpu_set, ffn1_pim_bin);
    amm_host(dpu_set, ((float*)ffn_input->data), ((float*)layer_weight->ffn1_centroid->data), ((lut_data_type*)layer_weight->ffn1_lut_table->data), ((float*)layer_weight->ffn1_bias->data), ((float*)lut_ffn1->data), ffn1_params.index_params, ffn1_params.lut_params);
#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.amm_latency += time2 - time1;
#elif defined(TRANSFORMER_BREAKDOWN_AMM)
    time2 = W_time();
    transformer_amm_profiles.ffn1_latency += time2 - time1;
#endif

    ////////////////////////////////////////////////////////////////////////////////////

#ifdef TRANSFORMER_BREAKDOWN
    time1 = W_time();
#endif
    // step 7: Activation
    struct ggml_tensor* lut_ffn1_activated = gelu(ctx, lut_ffn1, transformer_params.num_threads);

    // data reorder
    struct ggml_tensor* lut_ffn1_activated_reshaped = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn1_params.lut_params.output_feature_len, ffn1_params.lut_params.n);
    #pragma omp parallel for num_threads(transformer_params.num_threads)
    for(uint32_t tmp_row=0; tmp_row<ffn1_params.lut_params.n; ++tmp_row)
    {
        uint32_t tmp_n_tile_id = tmp_row / ffn1_params.lut_params.n_stile_size;
        uint32_t intra_n_tile_id = tmp_row % ffn1_params.lut_params.n_stile_size;

        for(uint32_t tmp_dim=0; tmp_dim<ffn1_params.lut_params.output_feature_len; tmp_dim += ffn1_params.lut_params.feature_stile_size)
        {
            uint32_t tmp_feature_tile_id = tmp_dim / ffn1_params.lut_params.feature_stile_size;
            uint32_t tmp_dpu_tile_offset = tmp_n_tile_id * ffn1_params.lut_params.n_stile_size * ffn1_params.lut_params.output_feature_len
                                         + tmp_feature_tile_id * ffn1_params.lut_params.n_stile_size * ffn1_params.lut_params.feature_stile_size;

            for(uint32_t tmp_intra_dim=0; tmp_intra_dim<ffn1_params.lut_params.feature_stile_size; tmp_intra_dim+=ffn1_params.lut_params.feature_mtile_size)
            {
                uint32_t tmp_intra_feature_tile_id = tmp_intra_dim / ffn1_params.lut_params.feature_mtile_size;
                uint32_t tmp_intra_dpu_tile_offset = tmp_intra_feature_tile_id * ffn1_params.lut_params.n_stile_size * ffn1_params.lut_params.feature_mtile_size
                                                   + intra_n_tile_id * ffn1_params.lut_params.feature_mtile_size;

                memcpy((((float*)lut_ffn1_activated_reshaped->data) + tmp_row*ffn1_params.lut_params.output_feature_len + tmp_dim + tmp_intra_dim),
                       (((float*)lut_ffn1_activated->data) + tmp_dpu_tile_offset + tmp_intra_dpu_tile_offset),
                       sizeof(float) * ffn1_params.lut_params.feature_mtile_size);
            }
        }
    }
#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.non_amm_latency += time2 - time1;
#endif

    ////////////////////////////////////////////////////////////////////////////////////

#if defined(TRANSFORMER_BREAKDOWN) || defined(TRANSFORMER_BREAKDOWN_AMM)
    time1 = W_time();
#endif
    // step 8: ffn2 lut
    struct ggml_tensor* lut_ffn2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn2_params.lut_params.output_feature_len, ffn2_params.lut_params.n);
    load_binary(dpu_set, ffn2_pim_bin);
    amm_host(dpu_set, ((float*)lut_ffn1_activated_reshaped->data), ((float*)layer_weight->ffn2_centroid->data), ((lut_data_type*)layer_weight->ffn2_lut_table->data), ((float*)layer_weight->ffn2_bias->data), ((float*)lut_ffn2->data), ffn2_params.index_params, ffn2_params.lut_params);
#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.amm_latency += time2 - time1;
#elif defined(TRANSFORMER_BREAKDOWN_AMM)
    time2 = W_time();
    transformer_amm_profiles.ffn2_latency += time2 - time1;
#endif

    ////////////////////////////////////////////////////////////////////////////////////

#ifdef TRANSFORMER_BREAKDOWN
    time1 = W_time();
#endif
    struct ggml_tensor* layer_output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ffn2_params.lut_params.output_feature_len, ffn2_params.lut_params.n);

    // data reorder
    #pragma omp parallel for num_threads(transformer_params.num_threads)
    for(uint32_t tmp_row=0; tmp_row<ffn2_params.lut_params.n; ++tmp_row)
    {
        uint32_t tmp_n_tile_id = tmp_row / ffn2_params.lut_params.n_stile_size;
        uint32_t intra_n_tile_id = tmp_row % ffn2_params.lut_params.n_stile_size;

        for(uint32_t tmp_dim=0; tmp_dim<ffn2_params.lut_params.output_feature_len; tmp_dim += ffn2_params.lut_params.feature_stile_size)
        {
            uint32_t tmp_feature_tile_id = tmp_dim / ffn2_params.lut_params.feature_stile_size;
            uint32_t tmp_dpu_tile_offset = tmp_n_tile_id * ffn2_params.lut_params.n_stile_size * ffn2_params.lut_params.output_feature_len
                                         + tmp_feature_tile_id * ffn2_params.lut_params.n_stile_size * ffn2_params.lut_params.feature_stile_size;

            for(uint32_t tmp_intra_dim=0; tmp_intra_dim<ffn2_params.lut_params.feature_stile_size; tmp_intra_dim+=ffn2_params.lut_params.feature_mtile_size)
            {
                uint32_t tmp_intra_feature_tile_id = tmp_intra_dim / ffn2_params.lut_params.feature_mtile_size;
                uint32_t tmp_intra_dpu_tile_offset = tmp_intra_feature_tile_id * ffn2_params.lut_params.n_stile_size * ffn2_params.lut_params.feature_mtile_size
                                                   + intra_n_tile_id * ffn2_params.lut_params.feature_mtile_size;

                memcpy((((float*)layer_output->data) + tmp_row*ffn2_params.lut_params.output_feature_len + tmp_dim + tmp_intra_dim),
                       (((float*)lut_ffn2->data) + tmp_dpu_tile_offset + tmp_intra_dpu_tile_offset),
                       sizeof(float) * ffn2_params.lut_params.feature_mtile_size);
            }
        }
    }

    // step 9: FFN Output Norm
    struct ggml_tensor* normed_ffn_output = norm(ctx, layer_weight->ffn_norm, layer_weight->ffn_norm_bias, layer_output, attn_params);

    // step 10: residual with normed ffn input
    #pragma omp parallel for num_threads(transformer_params.num_threads)
    for(uint32_t tmp_row=0; tmp_row<ffn2_params.lut_params.n; ++tmp_row)
    {
        for(uint32_t tmp_dim=0; tmp_dim<ffn2_params.lut_params.output_feature_len; ++tmp_dim)
        {
            ((float*)layer_output->data)[tmp_row*ffn2_params.lut_params.output_feature_len + tmp_dim] 
            = ((float*)ffn_input->data)[tmp_row*o_params.lut_params.output_feature_len + tmp_dim]
            + ((float*)normed_ffn_output->data)[tmp_row*ffn2_params.lut_params.output_feature_len + tmp_dim];
        }
    }
#ifdef TRANSFORMER_BREAKDOWN
    time2 = W_time();
    transformer_profiles.non_amm_latency += time2 - time1;
#endif

#ifdef TRANSFORMER_BREAKDOWN
    printf("amm time %.6f, non amm time %.6f\n", transformer_profiles.amm_latency, transformer_profiles.non_amm_latency);
#elif defined(TRANSFORMER_BREAKDOWN_AMM)
    printf("qkv projection time %.6f, o projection time %.6f, ffn1 time %.6f, ffn2 time %.6f\n", 
            transformer_amm_profiles.qkv_projection_latency, transformer_amm_profiles.o_projection_latency, transformer_amm_profiles.ffn1_latency, transformer_amm_profiles.ffn2_latency);
#endif

    return layer_output;
}
