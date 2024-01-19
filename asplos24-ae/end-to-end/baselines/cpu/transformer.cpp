#include "transformer.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <thread>
#include <algorithm>
#include <random>



//
// Randomize functions
//

float frand() {
    return (float)rand()/(float)RAND_MAX;
}

void init_random_normal_distribution(struct random_normal_distribution * rnd, int seed, float mean, float std, float min, float max) {
    rnd->gen = std::mt19937(seed);
    rnd->nd = std::normal_distribution<float>{mean, std};
    rnd->min = min;
    rnd->max = max;
}

float frand_normal(struct random_normal_distribution * rnd) {
    const float r = rnd->nd(rnd->gen);
    return ((r < rnd->min) ? (rnd->min) : (r > rnd->max) ? (rnd->max) : r);
}

struct ggml_tensor * randomize_tensor(
        struct ggml_tensor * tensor,
        int ndims,
        const int64_t ne[],
        float fmin,
        float fmax) {

    switch (ndims) {
        case 1:
            #pragma omp parallel for num_threads(40)
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)tensor->data)[i0] = frand()*(fmax - fmin) + fmin;
            }
            break;
        case 2:
            #pragma omp parallel for num_threads(40)
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)tensor->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                }
            }
            break;
        case 3:
            #pragma omp parallel for num_threads(40)
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)tensor->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    }
                }
            }
            break;
        case 4:
            #pragma omp parallel for num_threads(40)
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)tensor->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };

    return tensor;
}

struct ggml_tensor * randomize_tensor_normal(
        struct ggml_tensor * tensor,
        int ndims,
        const int64_t ne[],
        struct random_normal_distribution * rnd) {
    switch (ndims) {
        case 1:
            #pragma omp parallel for num_threads(40)
            for (int i0 = 0; i0 < ne[0]; i0++) {
                ((float *)tensor->data)[i0] = frand_normal(rnd);
            }
            break;
        case 2:
            #pragma omp parallel for num_threads(40)
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                    ((float *)tensor->data)[i1*ne[0] + i0] = frand_normal(rnd);
                }
            }
            break;
        case 3:
            #pragma omp parallel for num_threads(40)
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                        ((float *)tensor->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand_normal(rnd);
                    }
                }
            }
            break;
        case 4:
            #pragma omp parallel for num_threads(40)
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                            ((float *)tensor->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand_normal(rnd);
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };

    return tensor;
}


//
// Model setup
//

void quantize(ggml_tensor* original_tensor, ggml_tensor* quantized_tensor, ggml_type wtype)
{
    // only 2D weight matrices will be quantized
    assert(original_tensor->type == GGML_TYPE_F32);
    assert(original_tensor->n_dims <= 2 && quantized_tensor->n_dims <= 2 && original_tensor->n_dims == quantized_tensor->n_dims);
    if(original_tensor->n_dims == 1)
        assert(wtype == GGML_TYPE_F32);
    
    if(original_tensor->n_dims == 1)
        assert(original_tensor->ne[0] == quantized_tensor->ne[0]);
    else
        assert(original_tensor->ne[0] == quantized_tensor->ne[0] && original_tensor->ne[1] == quantized_tensor->ne[1]);

    int element_num = 0;
    if(original_tensor->n_dims == 1)
        element_num = original_tensor->ne[0];
    else
        element_num = original_tensor->ne[0] * original_tensor->ne[1];

    switch(wtype)
    {
    case GGML_TYPE_F32:
        memcpy(quantized_tensor->data, original_tensor->data, sizeof(float) * element_num);
        break;
    case GGML_TYPE_F16:
        ggml_fp32_to_fp16_row((float*)(original_tensor->data), (ggml_fp16_t*)(quantized_tensor->data), element_num);
        break;
    case GGML_TYPE_Q8_0:
        ggml_quantize_q8_0((float*)(original_tensor->data), quantized_tensor->data, element_num, original_tensor->ne[0], std::vector<int64_t>(1 << 8, 0).data());
        break;
    default:
        exit(-1);
    }
}

transformer_ctx* init_transformer_ctx()
{
    transformer_ctx* new_transformer = new transformer_ctx;
    transformer_model & model = new_transformer->model;

    // select the data type of weight tensors
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.data_type)
    {
    case 0:
        wtype = GGML_TYPE_F32;
        break;
    case 1:
        wtype = GGML_TYPE_F16;
        break;
    case 2:
        wtype = GGML_TYPE_Q8_0;
        break;
    default:
        fprintf(stderr, "%s: invalid data type %d\n",
                __func__, model.hparams.data_type);
        free_transformer_ctx(new_transformer);
        return nullptr;
    }

    auto &ctx = model.ctx;
    // calculate model's memory requirement
    size_t model_mem_req = 0;
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;

        // Calculate size requirements
        model_mem_req += 2 * n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_e_*

        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_*

        model_mem_req += 4 * n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // kqvo weights
        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // kqvo bias

        model_mem_req += 2 * n_layer * (n_embd * n_intermediate * ggml_type_sizef(wtype)); // ff_*_w
        model_mem_req += n_layer * (n_intermediate * ggml_type_sizef(GGML_TYPE_F32)); // ff_i_b
        model_mem_req += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ff_o_b

        model_mem_req += (5 + 16 * n_layer) * 256; // object overhead

        printf("%s: memory size required by model = %6.2f MB\n", __func__, model_mem_req / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = model_mem_req * 8, // leave space for conducting quantization
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx)
        {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            free_transformer_ctx(new_transformer);
            return nullptr;
        }
    }

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;

        model.ln_e_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_e_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        // map by name
        model.tensors["embeddings.LayerNorm.weight"] = model.ln_e_w;
        model.tensors["embeddings.LayerNorm.bias"] = model.ln_e_b;

        model.layers.resize(n_layer);
        for (int i = 0; i < n_layer; ++i)
        {
            auto &layer = model.layers[i];

            layer.ln_att_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_att_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.q_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.k_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.v_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.o_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.ff_i_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_intermediate);
            layer.ff_i_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_intermediate);

            layer.ff_o_w = ggml_new_tensor_2d(ctx, wtype, n_intermediate, n_embd);
            layer.ff_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.query.weight"] = layer.q_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.query.bias"] = layer.q_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.key.weight"] = layer.k_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.key.bias"] = layer.k_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.value.weight"] = layer.v_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.value.bias"] = layer.v_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight"] = layer.ln_att_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias"] = layer.ln_att_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.dense.weight"] = layer.o_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.dense.bias"] = layer.o_b;

            model.tensors["encoder.layer." + std::to_string(i) + ".intermediate.dense.weight"] = layer.ff_i_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".intermediate.dense.bias"] = layer.ff_i_b;

            model.tensors["encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight"] = layer.ln_out_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias"] = layer.ln_out_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.dense.weight"] = layer.ff_o_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.dense.bias"] = layer.ff_o_b;
        }
    }

    // init weights
    {
        struct random_normal_distribution rnd;
        init_random_normal_distribution(&rnd, 1337, 0.0f, 5.0f, -10.0f, +10.0f);
        
        const auto &hparams = model.hparams;
        const int n_layer = hparams.n_layer;
        const int n_embd = hparams.n_embd;
        const int n_intermediate = hparams.n_intermediate;

        ggml_tensor* ln_e_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        randomize_tensor_normal(ln_e_w, ln_e_w->n_dims, ln_e_w->ne, &rnd);
        quantize(ln_e_w, model.ln_e_w, model.ln_e_w->type);
        ggml_tensor* ln_e_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        randomize_tensor_normal(ln_e_b, ln_e_b->n_dims, ln_e_b->ne, &rnd);
        quantize(ln_e_b, model.ln_e_b, model.ln_e_b->type);

        for(int i=0; i<n_layer; ++i)
        {
            auto &layer = model.layers[i];

            // generate unquantized FP32 data first, then convert to corresponding data types

            // layer normalization weight/bias
            ggml_tensor* ln_att_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            randomize_tensor_normal(ln_att_w, ln_att_w->n_dims, ln_att_w->ne, &rnd);
            quantize(ln_att_w, layer.ln_att_w, layer.ln_att_w->type);

            ggml_tensor* ln_att_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            randomize_tensor_normal(ln_att_b, ln_att_b->n_dims, ln_att_b->ne, &rnd);
            quantize(ln_att_b, layer.ln_att_b, layer.ln_att_b->type);

            ggml_tensor* ln_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            randomize_tensor_normal(ln_out_w, ln_out_w->n_dims, ln_out_w->ne, &rnd);
            quantize(ln_out_w, layer.ln_out_w, layer.ln_out_w->type);

            ggml_tensor* ln_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            randomize_tensor_normal(ln_out_b, ln_out_b->n_dims, ln_out_b->ne, &rnd);
            quantize(ln_out_b, layer.ln_out_b, layer.ln_out_b->type);

            // QKVO projection weight/bias
            ggml_tensor* q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
            randomize_tensor_normal(q_w, q_w->n_dims, q_w->ne, &rnd);
            quantize(q_w, layer.q_w, layer.q_w->type);

            ggml_tensor* q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            randomize_tensor_normal(q_b, q_b->n_dims, q_b->ne, &rnd);
            quantize(q_b, layer.q_b, layer.q_b->type);

            ggml_tensor* k_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
            randomize_tensor_normal(k_w, k_w->n_dims, k_w->ne, &rnd);
            quantize(k_w, layer.k_w, layer.k_w->type);

            ggml_tensor* k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            randomize_tensor_normal(k_b, k_b->n_dims, k_b->ne, &rnd);
            quantize(k_b, layer.k_b, layer.k_b->type);

            ggml_tensor* v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
            randomize_tensor_normal(v_w, v_w->n_dims, v_w->ne, &rnd);
            quantize(v_w, layer.v_w, layer.v_w->type);

            ggml_tensor* v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            randomize_tensor_normal(v_b, v_b->n_dims, v_b->ne, &rnd);
            quantize(v_b, layer.v_b, layer.v_b->type);

            ggml_tensor* o_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_embd);
            randomize_tensor_normal(o_w, o_w->n_dims, o_w->ne, &rnd);
            quantize(o_w, layer.o_w, layer.o_w->type);

            ggml_tensor* o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            randomize_tensor_normal(o_b, o_b->n_dims, o_b->ne, &rnd);
            quantize(o_b, layer.o_b, layer.o_b->type);

            // FFN1/2 weight/bias
            ggml_tensor* ff_i_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_intermediate);
            randomize_tensor_normal(ff_i_w, ff_i_w->n_dims, ff_i_w->ne, &rnd);
            quantize(ff_i_w, layer.ff_i_w, layer.ff_i_w->type);

            ggml_tensor* ff_i_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_intermediate);
            randomize_tensor_normal(ff_i_b, ff_i_b->n_dims, ff_i_b->ne, &rnd);
            quantize(ff_i_b, layer.ff_i_b, layer.ff_i_b->type);

            ggml_tensor* ff_o_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_intermediate, n_embd);
            randomize_tensor_normal(ff_o_w, ff_o_w->n_dims, ff_o_w->ne, &rnd);
            quantize(ff_o_w, layer.ff_o_w, layer.ff_o_w->type);

            ggml_tensor* ff_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            randomize_tensor_normal(ff_o_b, ff_o_b->n_dims, ff_o_b->ne, &rnd);
            quantize(ff_o_b, layer.ff_o_b, layer.ff_o_b->type);
        }
    }

    return new_transformer;
}

void free_transformer_ctx(transformer_ctx* ctx)
{
    ggml_free(ctx->model.ctx);
    delete ctx;
}


//
// Model inference
//

void assert_shape_1d(struct ggml_tensor * tensor, int64_t ne0) {
    GGML_ASSERT(tensor->n_dims == 1);
    GGML_ASSERT(tensor->ne[0] == ne0);
}

void assert_shape_2d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1) {
    GGML_ASSERT(tensor->n_dims == 2);
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
}

void assert_shape_3d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2) {
    GGML_ASSERT(tensor->n_dims == 3);
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
}

void assert_shape_4d(struct ggml_tensor * tensor, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    GGML_ASSERT(tensor->n_dims == 4);
    GGML_ASSERT(tensor->ne[0] == ne0);
    GGML_ASSERT(tensor->ne[1] == ne1);
    GGML_ASSERT(tensor->ne[2] == ne2);
    GGML_ASSERT(tensor->ne[3] == ne3);
}

struct ggml_tensor* forward(
    struct transformer_ctx* ctx,
    struct ggml_context* ctx0, // used for store intermediate/output data tensors
    struct ggml_tensor* embedding_input, // shape is (n_embd, n_tokens * n_batch)
    const int n_threads,
    const int n_tokens,
    const int n_batch
)
{
    const int N = n_tokens;
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    const transformer_model& model = ctx->model;
    const auto &hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_intermediate = hparams.n_intermediate;
    const int n_layer = hparams.n_layer;
    const int n_head = hparams.n_head;

    struct ggml_tensor* inpL = embedding_input;
    assert_shape_2d(inpL, n_embd, N * n_batch);
    // embd norm
    {
        inpL = ggml_norm(ctx0, inpL);
        inpL = ggml_add(ctx0,
                        ggml_mul(ctx0,
                                 ggml_repeat(ctx0, model.ln_e_w, inpL),
                                                                 inpL),
                        ggml_repeat(ctx0, model.ln_e_b, inpL));
    }
    
    for(int il=0; il<n_layer; ++il)
    {
        struct ggml_tensor* inpSA = inpL;
        struct ggml_tensor* cur = inpL;

        // self-attention
        {
            // calculate Q
            struct ggml_tensor* Qcur = ggml_reshape_4d(ctx0, 
                                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, cur),
                                                                ggml_mul_mat(ctx0, model.layers[il].q_w, cur)),
                                                       n_embd/n_head, n_head, N, n_batch);
            struct ggml_tensor* Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
            assert_shape_4d(Q, n_embd/n_head, N, n_head, n_batch);

            // calculate K
            struct ggml_tensor* Kcur = ggml_reshape_4d(ctx0,
                                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, cur),
                                                                ggml_mul_mat(ctx0, model.layers[il].k_w, cur)),
                                                       n_embd/n_head, n_head, N, n_batch);
            struct ggml_tensor* K = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
            assert_shape_4d(K, n_embd/n_head, N, n_head, n_batch);

            // calculate V
            struct ggml_tensor* Vcur = ggml_reshape_4d(ctx0,
                                                       ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, cur),
                                                                ggml_mul_mat(ctx0, model.layers[il].v_w, cur)),
                                                       n_embd/n_head, n_head, N, n_batch);
            // std::cout << Vcur->ne[0] << " " << Vcur->ne[1] << " " << Vcur->ne[2] << " " << Vcur->ne[3] << "\n";
            struct ggml_tensor* V = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_permute(ctx0, Vcur, 0, 2, 1, 3)));
            // std::cout << V->ne[0] << " " << V->ne[1] << " " << V->ne[2] << " " << V->ne[3] << "\n";
            assert_shape_4d(V, N, n_embd/n_head, n_head, n_batch);

            // calculate QK
            struct ggml_tensor* KQ = ggml_mul_mat(ctx0, K, Q);
            assert_shape_4d(KQ, N, N, n_head, n_batch);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled = ggml_scale(ctx0,
                                                        KQ,
                                                        ggml_new_f32(ctx0, 1.0f/sqrtf(float(n_embd)/n_head)));
            assert_shape_4d(KQ_scaled, N, N, n_head, n_batch);

            // KQ_soft_max = soft_max(KQ_scaled)
            struct ggml_tensor* KQ_soft_max = ggml_soft_max(ctx0, KQ_scaled);
            assert_shape_4d(KQ_soft_max, N, N, n_head, n_batch);

            // calculate attention value
            struct ggml_tensor* KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
            assert_shape_4d(KQV, n_embd/n_head, N, n_head, n_batch);
            struct ggml_tensor* KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
            assert_shape_4d(KQV_merged, n_embd/n_head, n_head, N, n_batch);

            // projection
            cur = ggml_reshape_2d(ctx0, ggml_cont(ctx0, KQV_merged), n_embd, N*n_batch);
            assert_shape_2d(cur, n_embd, N*n_batch);
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].o_b, cur),
                           ggml_mul_mat(ctx0, model.layers[il].o_w, cur));
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // residual
        cur = ggml_add(ctx0, cur, inpSA);
        assert_shape_2d(cur, n_embd, N*n_batch);

        // attention output norm
        {
            cur = ggml_norm(ctx0, cur);
            assert_shape_2d(cur, n_embd, N*n_batch);

            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_att_w, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_att_b, cur));
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        struct ggml_tensor* inpFF = cur;
        assert_shape_2d(inpFF, n_embd, N*n_batch);

        // feed forward network
        {
            // ffn layer 1
            cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].ff_i_b, cur),
                           cur);
            assert_shape_2d(cur, n_intermediate, N*n_batch);
            
            // gelu function
            cur = ggml_gelu(ctx0, cur);
            assert_shape_2d(cur, n_intermediate, N*n_batch);

            // ffn layer 2
            cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].ff_o_b, cur),
                           cur);
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        // residual
        cur = ggml_add(ctx0, cur, inpFF);
        assert_shape_2d(cur, n_embd, N*n_batch);

        // ffn output norm
        {
            cur = ggml_norm(ctx0, cur);
            assert_shape_2d(cur, n_embd, N*n_batch);

            cur = ggml_add(ctx0,
                               ggml_mul(ctx0,
                                        ggml_repeat(ctx0, model.layers[il].ln_out_w, cur),
                                        cur),
                               ggml_repeat(ctx0, model.layers[il].ln_out_b, cur));
            assert_shape_2d(cur, n_embd, N*n_batch);
        }

        inpL = cur;
        assert_shape_2d(inpL, n_embd, N*n_batch);
    }

    ggml_tensor* output = inpL;

    // run the computation
    ggml_build_forward_expand(&gf, output);
    ggml_graph_compute(ctx0, &gf);

    return output;
}

struct ggml_tensor* projection_forward(
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    struct ggml_context* ctx0,
    struct ggml_tensor* input,
    const int n_threads
)
{
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor* result = ggml_add(ctx0, ggml_repeat(ctx0, bias, input), ggml_mul_mat(ctx0, weight, input));

    // run the computation
    ggml_build_forward_expand(&gf, result);
    ggml_graph_compute(ctx0, &gf);

    return result;
}

struct ggml_tensor* ffn_forward(
    struct ggml_tensor* weight,
    struct ggml_tensor* bias,
    struct ggml_context* ctx0,
    struct ggml_tensor* input,
    const int n_threads
)
{
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor* non_bias_result = ggml_mul_mat(ctx0, weight, input);
    struct ggml_tensor* result = ggml_add(ctx0, ggml_repeat(ctx0, bias, non_bias_result), non_bias_result); 

    // run the computation
    ggml_build_forward_expand(&gf, result);
    ggml_graph_compute(ctx0, &gf);

    return result;
}
