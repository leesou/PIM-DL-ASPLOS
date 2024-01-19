#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cassert>
#include <limits>
#include <omp.h>
#include <memory.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <sstream>

#include "ggml.h"

#include "parser.h"
#include "utils.h"
#include "defines.h"
#include "dpu_common.h"
#include "amm_host.h"
#include "transformer_layer.h"
extern "C"{
    #include <dpu.h>
}

using namespace std;

#define MSR_RAPL_POWER_UNIT "0x606"
#define MSR_PKG_ENERGY_STATUS "0x611"
#define MSR_DRAM_ENERGY_STATUS "0x619"
#ifdef MEASURE_ENERGY
    #define WARMUP_ROUND 0
    #define RECORD_ROUND 1
#else
    #define WARMUP_ROUND 5
    #define RECORD_ROUND 10
#endif

string execute_command(string command)
{
    FILE* pipe = popen(command.c_str(), "r");
    if(!pipe)
    {
        cout << "popen failed!" << endl;
        exit(-1);
    }

    char buffer[128];
    string result = "";
    while(fgets(buffer, sizeof(buffer), pipe) != nullptr)
    {
        result += buffer;
    }

    pclose(pipe);
    return result;
}

uint64_t read_msr(int cpu, string reg) 
{
    stringstream ss;
    ss << "sudo rdmsr -x " << reg << " -p " << cpu;
    string cmd_string = ss.str();
    string result = execute_command(cmd_string);
    uint64_t value = stoull(result, nullptr, 16);

    return value;
}

////////////////////////////////////////////// Data Generation

void generate_float_tensor(ggml_tensor* float_tensor, uint32_t float_tensor_size, uint32_t num_threads)
{
    float* data_ptr = (float*)float_tensor->data;

    std::mt19937 gen(233);

    uniform_real_distribution<float> dis(-5.0f, 5.0f);
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<float_tensor_size; ++i)
        data_ptr[i] = dis(gen);
}

void generate_lut_tensor(ggml_tensor* lut_table, uint32_t lut_table_size, uint32_t num_threads)
{
    lut_data_type* data_ptr = (lut_data_type*)lut_table->data;

    std::mt19937 gen(233);

    lut_data_type lower_bound = numeric_limits<lut_data_type>::min();
    lut_data_type upper_bound = numeric_limits<lut_data_type>::max();

    uniform_int_distribution<lut_data_type> dis(lower_bound, upper_bound);
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<lut_table_size; ++i)
        data_ptr[i] = dis(gen);
}

void generate_transformer_layer_weight(TransformerLayer& transformer_layer, TransformerParams& transformer_params)
{
    // generate attention norm weight
    transformer_layer.attention_norm = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    generate_float_tensor(transformer_layer.attention_norm, transformer_params.attention_params.token_dim, transformer_params.num_threads);
    transformer_layer.attention_norm_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    generate_float_tensor(transformer_layer.attention_norm_bias, transformer_params.attention_params.token_dim, transformer_params.num_threads);

#ifndef SEPARATE
    // generate qkv lut weight
    transformer_layer.qkv_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                        transformer_params.amm_param_list[0].index_params.sub_vec_len,
                                                        transformer_params.amm_param_list[0].index_params.num_centroid,
                                                        transformer_params.amm_param_list[0].index_params.num_codebook);
    uint32_t qkv_centroid_element_num = transformer_params.amm_param_list[0].index_params.sub_vec_len
                                      * transformer_params.amm_param_list[0].index_params.num_centroid
                                      * transformer_params.amm_param_list[0].index_params.num_codebook;
    generate_float_tensor(transformer_layer.qkv_centroid, qkv_centroid_element_num, transformer_params.num_threads);
    transformer_layer.qkv_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                         transformer_params.amm_param_list[0].lut_params.output_feature_len,
                                                         transformer_params.amm_param_list[0].lut_params.num_centroid,
                                                         transformer_params.amm_param_list[0].lut_params.num_codebook);
    uint32_t qkv_lut_element_num = transformer_params.amm_param_list[0].lut_params.output_feature_len
                                 * transformer_params.amm_param_list[0].lut_params.num_centroid
                                 * transformer_params.amm_param_list[0].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.qkv_lut_table, qkv_lut_element_num, transformer_params.num_threads);
    transformer_layer.qkv_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[0].lut_params.output_feature_len);
    uint32_t qkv_bias_element_num = transformer_params.amm_param_list[0].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.qkv_bias, qkv_bias_element_num, transformer_params.num_threads);
#else
    // generate q lut weight
    transformer_layer.q_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[4].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[4].index_params.num_centroid,
                                                      transformer_params.amm_param_list[4].index_params.num_codebook);
    uint32_t q_centroid_element_num = transformer_params.amm_param_list[4].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[4].index_params.num_centroid
                                    * transformer_params.amm_param_list[4].index_params.num_codebook;
    generate_float_tensor(transformer_layer.q_centroid, q_centroid_element_num, transformer_params.num_threads);
    transformer_layer.q_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[4].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[4].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[4].lut_params.num_codebook);
    uint32_t q_lut_element_num = transformer_params.amm_param_list[4].lut_params.output_feature_len
                               * transformer_params.amm_param_list[4].lut_params.num_centroid
                               * transformer_params.amm_param_list[4].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.q_lut_table, q_lut_element_num, transformer_params.num_threads);
    transformer_layer.q_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[4].lut_params.output_feature_len);
    uint32_t q_bias_element_num = transformer_params.amm_param_list[4].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.q_bias, q_bias_element_num, transformer_params.num_threads);

    // generate k lut weight
    transformer_layer.k_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[5].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[5].index_params.num_centroid,
                                                      transformer_params.amm_param_list[5].index_params.num_codebook);
    uint32_t k_centroid_element_num = transformer_params.amm_param_list[5].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[5].index_params.num_centroid
                                    * transformer_params.amm_param_list[5].index_params.num_codebook;
    generate_float_tensor(transformer_layer.k_centroid, k_centroid_element_num, transformer_params.num_threads);
    transformer_layer.k_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[5].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[5].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[5].lut_params.num_codebook);
    uint32_t k_lut_element_num = transformer_params.amm_param_list[5].lut_params.output_feature_len
                               * transformer_params.amm_param_list[5].lut_params.num_centroid
                               * transformer_params.amm_param_list[5].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.k_lut_table, k_lut_element_num, transformer_params.num_threads);
    transformer_layer.k_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[5].lut_params.output_feature_len);
    uint32_t k_bias_element_num = transformer_params.amm_param_list[5].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.k_bias, k_bias_element_num, transformer_params.num_threads);

    // generate v lut weight
    transformer_layer.v_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[6].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[6].index_params.num_centroid,
                                                      transformer_params.amm_param_list[6].index_params.num_codebook);
    uint32_t v_centroid_element_num = transformer_params.amm_param_list[6].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[6].index_params.num_centroid
                                    * transformer_params.amm_param_list[6].index_params.num_codebook;
    generate_float_tensor(transformer_layer.v_centroid, v_centroid_element_num, transformer_params.num_threads);
    transformer_layer.v_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[6].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[6].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[6].lut_params.num_codebook);
    uint32_t v_lut_element_num = transformer_params.amm_param_list[6].lut_params.output_feature_len
                               * transformer_params.amm_param_list[6].lut_params.num_centroid
                               * transformer_params.amm_param_list[6].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.v_lut_table, v_lut_element_num, transformer_params.num_threads);
    transformer_layer.v_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[6].lut_params.output_feature_len);
    uint32_t v_bias_element_num = transformer_params.amm_param_list[6].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.v_bias, v_bias_element_num, transformer_params.num_threads);
#endif

    // generate o lut weight
    transformer_layer.o_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                      transformer_params.amm_param_list[1].index_params.sub_vec_len,
                                                      transformer_params.amm_param_list[1].index_params.num_centroid,
                                                      transformer_params.amm_param_list[1].index_params.num_codebook);
    uint32_t o_centroid_element_num = transformer_params.amm_param_list[1].index_params.sub_vec_len
                                    * transformer_params.amm_param_list[1].index_params.num_centroid
                                    * transformer_params.amm_param_list[1].index_params.num_codebook;
    generate_float_tensor(transformer_layer.o_centroid, o_centroid_element_num, transformer_params.num_threads);
    transformer_layer.o_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                       transformer_params.amm_param_list[1].lut_params.output_feature_len,
                                                       transformer_params.amm_param_list[1].lut_params.num_centroid,
                                                       transformer_params.amm_param_list[1].lut_params.num_codebook);
    uint32_t o_lut_element_num = transformer_params.amm_param_list[1].lut_params.output_feature_len
                               * transformer_params.amm_param_list[1].lut_params.num_centroid
                               * transformer_params.amm_param_list[1].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.o_lut_table, o_lut_element_num, transformer_params.num_threads);
    transformer_layer.o_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[1].lut_params.output_feature_len);
    uint32_t o_bias_element_num = transformer_params.amm_param_list[1].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.o_bias, o_bias_element_num, transformer_params.num_threads);

    // generate ffn norm weight
    transformer_layer.ffn_norm = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    generate_float_tensor(transformer_layer.ffn_norm, transformer_params.attention_params.token_dim, transformer_params.num_threads);
    transformer_layer.ffn_norm_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.attention_params.token_dim);
    generate_float_tensor(transformer_layer.ffn_norm_bias, transformer_params.attention_params.token_dim, transformer_params.num_threads);


    // generate ffn1 lut weight
    transformer_layer.ffn1_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                         transformer_params.amm_param_list[2].index_params.sub_vec_len,
                                                         transformer_params.amm_param_list[2].index_params.num_centroid,
                                                         transformer_params.amm_param_list[2].index_params.num_codebook);
    uint32_t ffn1_centroid_element_num = transformer_params.amm_param_list[2].index_params.sub_vec_len
                                       * transformer_params.amm_param_list[2].index_params.num_centroid
                                       * transformer_params.amm_param_list[2].index_params.num_codebook;
    generate_float_tensor(transformer_layer.ffn1_centroid, ffn1_centroid_element_num, transformer_params.num_threads);
    transformer_layer.ffn1_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                          transformer_params.amm_param_list[2].lut_params.output_feature_len,
                                                          transformer_params.amm_param_list[2].lut_params.num_centroid,
                                                          transformer_params.amm_param_list[2].lut_params.num_codebook);
    uint32_t ffn1_lut_element_num = transformer_params.amm_param_list[2].lut_params.output_feature_len
                                  * transformer_params.amm_param_list[2].lut_params.num_centroid
                                  * transformer_params.amm_param_list[2].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.ffn1_lut_table, ffn1_lut_element_num, transformer_params.num_threads);
    transformer_layer.ffn1_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[2].lut_params.output_feature_len);
    uint32_t ffn1_bias_element_num = transformer_params.amm_param_list[2].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.ffn1_bias, ffn1_bias_element_num, transformer_params.num_threads);

    // generate ffn2 lut weight
    transformer_layer.ffn2_centroid = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_F32, 
                                                         transformer_params.amm_param_list[3].index_params.sub_vec_len,
                                                         transformer_params.amm_param_list[3].index_params.num_centroid,
                                                         transformer_params.amm_param_list[3].index_params.num_codebook);
    uint32_t ffn2_centroid_element_num = transformer_params.amm_param_list[3].index_params.sub_vec_len
                                       * transformer_params.amm_param_list[3].index_params.num_centroid
                                       * transformer_params.amm_param_list[3].index_params.num_codebook;
    generate_float_tensor(transformer_layer.ffn2_centroid, ffn2_centroid_element_num, transformer_params.num_threads);
    transformer_layer.ffn2_lut_table = ggml_new_tensor_3d(transformer_layer.ctx, GGML_TYPE_I8,
                                                          transformer_params.amm_param_list[3].lut_params.output_feature_len,
                                                          transformer_params.amm_param_list[3].lut_params.num_centroid,
                                                          transformer_params.amm_param_list[3].lut_params.num_codebook);
    uint32_t ffn2_lut_element_num = transformer_params.amm_param_list[3].lut_params.output_feature_len
                                  * transformer_params.amm_param_list[3].lut_params.num_centroid
                                  * transformer_params.amm_param_list[3].lut_params.num_codebook;
    generate_lut_tensor(transformer_layer.ffn2_lut_table, ffn2_lut_element_num, transformer_params.num_threads);
    transformer_layer.ffn2_bias = ggml_new_tensor_1d(transformer_layer.ctx, GGML_TYPE_F32, transformer_params.amm_param_list[3].lut_params.output_feature_len);
    uint32_t ffn2_bias_element_num = transformer_params.amm_param_list[3].lut_params.output_feature_len;
    generate_float_tensor(transformer_layer.ffn2_bias, ffn2_bias_element_num, transformer_params.num_threads);
}

////////////////////////////////////////////// Main function entry

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        printf("usage: %s config_yaml_path\n", argv[0]);
        exit(-1);
    }
    string config_yaml_path(argv[1]);

    double time1, time2;
    double total_time = 0.0;
#ifdef MEASURE_ENERGY
    double total_energy = 0.0;
#endif

    ////////////////////////////////////////////// transformer param parsing

    time1 = W_time();
    // dump transformer params
    TransformerParams transformer_params;
    parse_transformer_configs(transformer_params, config_yaml_path);
    time2 = W_time();
    printf("parameter parsing time %.6f\n", time2-time1);

    ////////////////////////////////////////////// transformer weight init

    time1 = W_time();
    // init transformer layer weight
    TransformerLayer transformer_layer;
    size_t    layer_size = 1024ll*1024ll*1024ll*1ll;
    uint8_t*  layer_addr = new uint8_t[layer_size];
    struct ggml_init_params layer_ctx_params = {
            /*.mem_size   =*/ layer_size,
            /*.mem_buffer =*/ layer_addr,
            /*.no_alloc   =*/ false,
    };
    transformer_layer.ctx = ggml_init(layer_ctx_params);
    generate_transformer_layer_weight(transformer_layer, transformer_params);
    time2 = W_time();
    printf("weight initialization time %.6f\n", time2-time1);

    ////////////////////////////////////////////// DPU allocate

    time1 = W_time();
    dpu_set_t dpu_set;
    uint32_t allocated_dpu_num;
    if(transformer_params.attention_params.dpu_num == NR_DPUS_PER_RANK)
        allocated_dpu_num = allocate_rank(&dpu_set, transformer_params.attention_params.dpu_num / NR_DPUS_PER_RANK);
    else
        allocated_dpu_num = allocate_dpu(&dpu_set, transformer_params.attention_params.dpu_num);
    time2 = W_time();
    printf("dpu allocate time %.6f\n", time2-time1);

    ////////////////////////////////////////////// inference

    size_t    data_size = 1024ll*1024ll*1024ll*64ll;
    uint8_t*  data_addr = new uint8_t[data_size];

    for(uint32_t i=0; i<WARMUP_ROUND+RECORD_ROUND; ++i)
    {
        std::cout << "--------------------------------------------\n";
        time1 = W_time();
        struct ggml_init_params data_ctx_params = {
                /*.mem_size   =*/ data_size,
                /*.mem_buffer =*/ data_addr,
                /*.no_alloc   =*/ false,
        };
        struct ggml_context * ctx0 = ggml_init(data_ctx_params);
        struct ggml_tensor* input_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, transformer_params.attention_params.token_dim, transformer_params.attention_params.n);
        generate_float_tensor(input_tensor, transformer_params.attention_params.n * transformer_params.attention_params.token_dim, transformer_params.num_threads);
        time2 = W_time();
        printf("input initialization time %.6f\n", time2-time1);

#ifdef MEASURE_ENERGY
        uint64_t power_unit_0 = read_msr(0, MSR_RAPL_POWER_UNIT);
        uint32_t energy_units_0 = ((power_unit_0 >> 8) & 0x1F);
        double energy_scale_0 = 1.0 / (1 << energy_units_0);
        uint64_t power_unit_1 = read_msr(10, MSR_RAPL_POWER_UNIT);
        uint32_t energy_units_1 = ((power_unit_1 >> 8) & 0x1F);
        double energy_scale_1 = 1.0 / (1 << energy_units_1);

        uint64_t start_pkg_energy_status_0 = read_msr(0, MSR_PKG_ENERGY_STATUS);
        uint64_t start_dram_energy_status_0 = read_msr(0, MSR_DRAM_ENERGY_STATUS);
        uint64_t start_pkg_energy_status_1 = read_msr(10, MSR_PKG_ENERGY_STATUS);
        uint64_t start_dram_energy_status_1 = read_msr(10, MSR_DRAM_ENERGY_STATUS);
#endif

        time1 = W_time();
        struct ggml_tensor* layer_output = pim_lut_transformer_layer(&dpu_set, ctx0, &transformer_layer, input_tensor, transformer_params);
        time2 = W_time();
        printf("iteration %d, inference time %.6f\n", i, time2-time1);

#ifdef MEASURE_ENERGY
        uint64_t end_pkg_energy_status_0 = read_msr(0, MSR_PKG_ENERGY_STATUS);
        uint64_t end_dram_energy_status_0 = read_msr(0, MSR_DRAM_ENERGY_STATUS);
        uint64_t end_pkg_energy_status_1 = read_msr(10, MSR_PKG_ENERGY_STATUS);
        uint64_t end_dram_energy_status_1 = read_msr(10, MSR_DRAM_ENERGY_STATUS);

        double cpu_energy = double(end_pkg_energy_status_0-start_pkg_energy_status_0)*energy_scale_0 + double(end_pkg_energy_status_1-start_pkg_energy_status_1)*energy_scale_1;
        double dram_energy = double(end_dram_energy_status_0-start_dram_energy_status_0)*energy_scale_0 + double(end_dram_energy_status_1-start_dram_energy_status_1)*energy_scale_1;
        double pim_energy = 13.2*8 * (time2-time1);
        printf("iteration %d, total energy %.6f\n", i, cpu_energy+dram_energy+pim_energy);
#endif

        if(i>=WARMUP_ROUND)
            total_time += time2 - time1;
#ifdef MEASURE_ENERGY
        if(i>=WARMUP_ROUND)
            total_energy += cpu_energy + dram_energy + pim_energy;
#endif

        fflush(stdout);
    }

    std::cout << "--------------------------------------------\n";
    printf("%d rounds' average inference time %.6f\n", RECORD_ROUND, total_time / RECORD_ROUND);
#ifdef MEASURE_ENERGY
    printf("%d rounds' average energy %.6f\n", RECORD_ROUND, total_energy / RECORD_ROUND);
#endif
}
