#ifndef BERT_H
#define BERT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <memory.h>
#include <omp.h>
#include <random>
#include <map>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif


//
// Struct declaration
//

////////////////////////////////////// for bert base

struct inference_params
{
    int32_t n_threads = 32;
    int32_t n_tokens = 512;
    int32_t n_batch = 64;
};

struct transformer_hparams {
    uint32_t n_embd  = 768;
    uint32_t n_intermediate = 3072;
    uint32_t n_head  = 12;
    uint32_t n_layer = 1;
    uint32_t data_type = 2; // 0: FP32, 1: FP16, 2: Q8

    bool operator!=(const transformer_hparams & other) const {
        return memcmp(this, &other, sizeof(transformer_hparams));
    }
};

//////////////////////////////////////// for bert large

// struct inference_params
// {
//     int32_t n_threads = 32;
//     int32_t n_tokens = 512;
//     int32_t n_batch = 64;
// };

// struct transformer_hparams {
//     uint32_t n_embd  = 1024;
//     uint32_t n_intermediate = 4096;
//     uint32_t n_head  = 16;
//     uint32_t n_layer = 1;
//     uint32_t data_type = 2; // 0: FP32, 1: FP16, 2: Q8

//     bool operator!=(const transformer_hparams & other) const {
//         return memcmp(this, &other, sizeof(transformer_hparams));
//     }
// };

//////////////////////////////////////// for vit huge

// struct inference_params
// {
//     int32_t n_threads = 32;
//     int32_t n_tokens = 264;
//     int32_t n_batch = 128;
// };

// struct transformer_hparams {
//     uint32_t n_embd  = 1280;
//     uint32_t n_intermediate = 5120;
//     uint32_t n_head  = 16;
//     uint32_t n_layer = 1;
//     uint32_t data_type = 2; // 0: FP32, 1: FP16, 2: Q8

//     bool operator!=(const transformer_hparams & other) const {
//         return memcmp(this, &other, sizeof(transformer_hparams));
//     }
// };

// struct inference_params
// {
//     int32_t n_threads = 32;
//     int32_t n_tokens = 512;
//     int32_t n_batch = 128;
// };

// struct transformer_hparams {
//     uint32_t n_embd  = 1280;
//     uint32_t n_intermediate = 5120;
//     uint32_t n_head  = 16;
//     uint32_t n_layer = 1;
//     uint32_t data_type = 2; // 0: FP32, 1: FP16, 2: Q8

//     bool operator!=(const transformer_hparams & other) const {
//         return memcmp(this, &other, sizeof(transformer_hparams));
//     }
// };

//
// Struct declaration
//

struct transformer_layer
{
    // normalization
    struct ggml_tensor *ln_att_w;
    struct ggml_tensor *ln_att_b;

    struct ggml_tensor *ln_out_w;
    struct ggml_tensor *ln_out_b;

    // attention
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;
};

struct transformer_model
{
    transformer_hparams hparams;

    // embedding normalization
    struct ggml_tensor *ln_e_w;
    struct ggml_tensor *ln_e_b;

    // layer weights
    std::vector<transformer_layer> layers;

    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct transformer_ctx
{
    transformer_model model;
};



//
// Randomize functions
//

struct random_normal_distribution {
    std::mt19937 gen;
    std::normal_distribution<float> nd;
    float min;
    float max;
};

void init_random_normal_distribution(
    struct random_normal_distribution * rnd, 
    int seed, 
    float mean, 
    float std, 
    float min, 
    float max
);

struct ggml_tensor * randomize_tensor(
    struct ggml_tensor * tensor,
    int ndims,
    const int64_t ne[],
    float fmin,
    float fmax
);

struct ggml_tensor * randomize_tensor_normal(
    struct ggml_tensor * tensor,
    int ndims,
    const int64_t ne[],
    struct random_normal_distribution * rnd
);


//
// Model setup
//

void quantize(ggml_tensor* original_tensor, ggml_tensor* quantized_tensor, ggml_type wtype);

struct transformer_ctx* init_transformer_ctx();

void free_transformer_ctx(transformer_ctx* ctx);


//
// Model inference
//

struct ggml_tensor* forward(
    struct transformer_ctx* ctx,
    struct ggml_context* ctx0, // used for store intermediate/output data tensors
    struct ggml_tensor* embedding_input, // shape is (n_embd, n_tokens * n_batch)
    const int n_threads,
    const int n_tokens,
    const int n_batch
);

struct ggml_tensor* projection_forward(
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    struct ggml_context* ctx0,
    struct ggml_tensor* input,
    const int n_threads
);

struct ggml_tensor* ffn_forward(
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    struct ggml_context* ctx0,
    struct ggml_tensor* input,
    const int n_threads
);

#ifdef __cplusplus
}
#endif

#endif // BERT_H
