#include "transformer.h"
#include "ggml.h"

#include <unistd.h>
#include <iostream>
#include <vector>
#include <string>

#define WARMUP_ROUND 10
#define RECORD_ROUND 20

int main(int argc, char** argv)
{
    int64_t t_model_init_us = 0;
    int64_t t_model_init_start_us = ggml_time_us();

    inference_params default_inference_params;
    transformer_hparams default_transformer_hparams;
#ifdef USE_INPUT_INFERENCE_PARAMS
    std::string n_tokens_string = argv[1];
    std::string n_batch_string = argv[2];
    std::string n_embd_string = argv[3];
    std::string n_head_string = argv[4];
    default_inference_params.n_tokens = std::stoi(n_tokens_string);
    default_inference_params.n_batch = std::stoi(n_batch_string);
    default_transformer_hparams.n_embd = std::stoi(n_embd_string);
    default_transformer_hparams.n_intermediate = std::stoi(n_embd_string) * 4;
    default_transformer_hparams.n_head = std::stoi(n_head_string);
#endif
    std::cout << "n_threads " << default_inference_params.n_threads << ", n_tokens " << default_inference_params.n_tokens << ", n_batch " << default_inference_params.n_batch << std::endl;
    std::cout << "n_embd " << default_transformer_hparams.n_embd << ", n_intermediate " << default_transformer_hparams.n_intermediate << ", n_head " << default_transformer_hparams.n_head << ", n_layer " << default_transformer_hparams.n_layer << ", data_type " << default_transformer_hparams.data_type << std::endl;
    // init transformer ctx
    transformer_ctx* tctx;
    tctx = init_transformer_ctx(default_transformer_hparams);
    
    t_model_init_us = ggml_time_us() - t_model_init_start_us;
    std::cout << "transformer context init time is " << t_model_init_us / 1000000.0f << " s" << std::endl;

    struct random_normal_distribution rnd;
    init_random_normal_distribution(&rnd, 1337, 0.0f, 5.0f, -10.0f, +10.0f);

    size_t compute_size = 1024ll*1024ll*1024ll*96ll;
    uint8_t * compute_addr = new uint8_t[compute_size];

#ifdef PROFILE_TRANSFORMER
    int64_t total_inference_us = 0;
#else
    struct ggml_tensor* fused_qkv_weight = ggml_new_tensor_2d(tctx->model.ctx, GGML_TYPE_F32, 
                                                   default_transformer_hparams.n_embd,
                                                   default_transformer_hparams.n_embd * 3);
    struct ggml_tensor* fused_qkv_bias = ggml_new_tensor_1d(tctx->model.ctx, GGML_TYPE_F32, 
                                            default_transformer_hparams.n_embd * 3);
    struct ggml_tensor* fused_qkv_weight_quantized = ggml_new_tensor_2d(tctx->model.ctx, GGML_TYPE_Q8_0, 
                                                   default_transformer_hparams.n_embd,
                                                   default_transformer_hparams.n_embd * 3);
    randomize_tensor_normal(fused_qkv_weight, fused_qkv_weight->n_dims, fused_qkv_weight->ne, &rnd);
    quantize(fused_qkv_weight, fused_qkv_weight_quantized, GGML_TYPE_Q8_0);
    randomize_tensor_normal(fused_qkv_bias, fused_qkv_bias->n_dims, fused_qkv_bias->ne, &rnd);

    int64_t total_qkv_project_us = 0;
    int64_t total_o_project_us = 0;
    int64_t total_ffn1_us = 0;
    int64_t total_ffn2_us = 0;
    int64_t total_qkv_fused_project_us = 0;
#endif
    for(int i=0; i<WARMUP_ROUND+RECORD_ROUND; ++i)
    {
        std::cout << "--------------------------------------------\n";

        ////////////////////////////////////////////////////////////////// data init

        int64_t t_data_init_us = 0;
        int64_t t_data_init_start_us = ggml_time_us();
        // init data context
        
        struct ggml_init_params params = {
            /*.mem_size   =*/ compute_size,
            /*.mem_buffer =*/ compute_addr,
            /*.no_alloc   =*/ false,
        };
        struct ggml_context* ctx0 = ggml_init(params);
        t_data_init_us = ggml_time_us() - t_data_init_start_us;
        std::cout << "data context init time is " << t_data_init_us / 1000000.0 << " s" << std::endl;

        int64_t t_input_init_us = 0;
        int64_t t_input_init_start_us = ggml_time_us();
        // generate input tensor
        struct ggml_tensor* input = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 
                                                   default_transformer_hparams.n_embd,
                                                   default_inference_params.n_tokens * default_inference_params.n_batch);
        randomize_tensor_normal(input, input->n_dims, input->ne, &rnd);
        t_input_init_us = ggml_time_us() - t_input_init_start_us;
        std::cout << "input context init time is " << t_input_init_us / 1000000.0 << " s" << std::endl;

        ////////////////////////////////////////////////////////////////// inference

#ifdef PROFILE_TRANSFORMER
        int64_t tmp_inference_us = 0;
        int64_t inference_start_us = ggml_time_us();

        forward(
            tctx,
            ctx0,
            input,
            default_inference_params.n_threads,
            default_inference_params.n_tokens,
            default_inference_params.n_batch
        );

        tmp_inference_us = ggml_time_us() - inference_start_us;
        std::cout << "iteration " << i << ", inference latency " << tmp_inference_us / 1000000.0 << " s" << std::endl;
        if(i >= WARMUP_ROUND)
            total_inference_us += tmp_inference_us;
#else
        int64_t qkv_project_us = 0;
        int64_t qkv_project_start_us = ggml_time_us();
        struct ggml_tensor* q_output = projection_forward(tctx->model.layers[0].q_w, tctx->model.layers[0].q_b, ctx0, input, default_inference_params.n_threads);
        struct ggml_tensor* k_output = projection_forward(tctx->model.layers[0].k_w, tctx->model.layers[0].k_b, ctx0, input, default_inference_params.n_threads);
        struct ggml_tensor* v_output = projection_forward(tctx->model.layers[0].v_w, tctx->model.layers[0].v_b, ctx0, input, default_inference_params.n_threads);
        qkv_project_us = ggml_time_us() - qkv_project_start_us;
        total_qkv_project_us += qkv_project_us;

        int64_t o_project_us = 0;
        int64_t o_project_start_us = ggml_time_us();
        struct ggml_tensor* o_output = projection_forward(tctx->model.layers[0].o_w, tctx->model.layers[0].o_b, ctx0, input, default_inference_params.n_threads);
        o_project_us = ggml_time_us() - o_project_start_us;
        total_o_project_us += o_project_us;

        int64_t ffn1_us = 0;
        int64_t ffn1_start_us = ggml_time_us();
        struct ggml_tensor* ffn1_output = ffn_forward(tctx->model.layers[0].ff_i_w, tctx->model.layers[0].ff_i_b, ctx0, input, default_inference_params.n_threads);
        ffn1_us = ggml_time_us() - ffn1_start_us;
        total_ffn1_us += ffn1_us;

        int64_t ffn2_us = 0;
        int64_t ffn2_start_us = ggml_time_us();
        struct ggml_tensor* ffn2_output = ffn_forward(tctx->model.layers[0].ff_o_w, tctx->model.layers[0].ff_o_b, ctx0, ffn1_output, default_inference_params.n_threads);
        ffn2_us = ggml_time_us() - ffn2_start_us;
        total_ffn2_us += ffn2_us;

        int64_t qkv_fused_project_us = 0;
        int64_t qkv_fused_project_start_us = ggml_time_us();
        struct ggml_tensor* qkv_fused_output = ffn_forward(fused_qkv_weight_quantized, fused_qkv_bias, ctx0, input, default_inference_params.n_threads);
        qkv_fused_project_us = ggml_time_us() - qkv_fused_project_start_us;
        total_qkv_fused_project_us += qkv_fused_project_us;

        printf("qkv projection time %.6f, o projection time %.6f, ffn1 time %.6f, ffn2 time %.6f, fused qkv time %.6f\n",
                qkv_project_us / 1000000.0, o_project_us / 1000000.0, ffn1_us / 1000000.0, ffn2_us / 1000000.0, qkv_fused_project_us / 1000000.0);
#endif
        
        ggml_free(ctx0);
    }
    free_transformer_ctx(tctx);

    std::cout << "--------------------------------------------\n";
#ifdef PROFILE_TRANSFORMER
    std::cout << "average latency of " << RECORD_ROUND << " iterations " << total_inference_us / (RECORD_ROUND * 1000000.0) << " s" << std::endl;
#else
    printf("total qkv projection time %.6f, total o projection time %.6f, total ffn1 time %.6f, total ffn2 time %.6f, total fused qkv time %.6f\n",
            total_qkv_project_us / 1000000.0, total_o_project_us / 1000000.0, total_ffn1_us / 1000000.0, total_ffn2_us / 1000000.0, total_qkv_fused_project_us / 1000000.0);
#endif
    return 0;
}