#pragma once
#include <stdint.h>
#include <string>

typedef int8_t lut_data_type;
typedef uint16_t index_data_type;
typedef int32_t output_data_type;


enum LUTLoadType {STATIC, FINE_GRAIN, COARSE_GRAIN};

struct IndexCalcParams
{
    uint32_t n;
    uint32_t input_feature_len;
    uint32_t sub_vec_len;
    uint32_t num_codebook;
    uint32_t num_centroid;
    uint32_t num_threads;

    uint32_t n_stile_size;
    uint32_t cb_mtile_size;
};

struct LUTParams
{
    uint32_t n;
    uint32_t output_feature_len;
    uint32_t num_codebook;
    uint32_t num_centroid;

    float scale;
    float bias;

    uint32_t num_threads;
    uint32_t dpu_num;

    uint32_t input_parallelism;
    uint32_t lut_parallelism;
    uint32_t feature_stile_size;
    uint32_t n_stile_size;

    uint32_t n_mtile_size;
    uint32_t feature_mtile_size;
    uint32_t cb_mtile_size;
    LUTLoadType lut_load_type;
    uint32_t feature_load_tile_size;
    uint32_t cb_load_tile_size;

};

struct AMMParams
{
    IndexCalcParams index_params;
    LUTParams lut_params;
};

struct AttentionParams
{
    uint32_t seq_len;
    uint32_t batch_size;
    uint32_t n;

    uint32_t head_num;
    uint32_t head_dim;
    uint32_t token_dim;

    uint32_t dpu_num;
    uint32_t input_parallelism;
    uint32_t lut_parallelism;

    uint32_t n_tile_size;
    uint32_t token_tile_size;

    uint32_t num_threads;
};

struct TransformerParams
{
    std::string pim_binary_list[4];
    AMMParams amm_param_list[7];

    AttentionParams attention_params;

    uint32_t layer_num;

    uint32_t num_threads;
};

typedef struct {
    uint32_t input_height;
} dpu_arguments_t;


