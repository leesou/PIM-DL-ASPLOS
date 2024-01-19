#pragma once
#include <string>

struct PIMParams
{
    int pe_num;
    int parallelism;
};

struct NetworkParams
{
    int seq_len;
    int batch_size;
    int head_num;
    int head_dim;
    int token_dim;
    int ffn_hidden_dim;
    int layer_num;
};

struct AMMParams
{
    int num_codebook;
    int num_centroid;
    int n;
    int input_feature_len;
    int output_feature_len;
};

struct KernelParams
{
    int n_stile_size = -1;
    int feature_stile_size = -1;
    int input_parallelism = -1;
    int lut_parallelism = -1;

    int loop_order = -1;
    int lut_load_type = -1;

    int n_mtile_size = -1;
    int feature_mtile_size = -1;
    int cb_mtile_size = -1;

    int feature_load_tile_size = -1;
    int cb_load_tile_size = -1;
};


void tune_single_kernel(std::string input_path, std::string output_path, bool verbose);
