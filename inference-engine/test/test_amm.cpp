#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cassert>
#include <limits>
#include <omp.h>
#include <memory.h>

#include "utils.h"
#include "defines.h"
#include "dpu_common.h"
#include "parser.h"
#include "amm_host.h"
extern "C"{
    #include <dpu.h>
}

using namespace std;

#define WARMUP_ROUND 10
#define RECORD_ROUND 20

////////////////////////////////////////////// Data Generation

void generate_input_tensor(float* input_tensor, uint32_t input_tensor_size, uint32_t num_threads)
{
    std::mt19937 gen(233);

    uniform_real_distribution<float> dis(-5.0f, 5.0f);
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<input_tensor_size; ++i)
        input_tensor[i] = dis(gen);
}

void generate_centroid_tensor(float* centroid_tensor, uint32_t centroid_tensor_size, uint32_t num_threads)
{
    std::mt19937 gen(233);

    uniform_real_distribution<float> dis(-10.0f, 10.0f);
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<centroid_tensor_size; ++i)
        centroid_tensor[i] = dis(gen);
}

void generate_lut_data(lut_data_type* lut_table, uint32_t lut_table_size, uint32_t num_threads)
{ 
    std::mt19937 gen(233);

    lut_data_type lower_bound = numeric_limits<lut_data_type>::min();
    lut_data_type upper_bound = numeric_limits<lut_data_type>::max();

    uniform_int_distribution<lut_data_type> dis(lower_bound, upper_bound);
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<lut_table_size; ++i)
        lut_table[i] = dis(gen);
}

void generate_bias_tensor(float* bias_tensor, uint32_t bias_tensor_size, uint32_t num_threads)
{
    std::mt19937 gen(233);

    uniform_real_distribution<float> dis(-10.0f, 10.0f);
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<bias_tensor_size; ++i)
        bias_tensor[i] = dis(gen);
}

//////////////////////////////////////////////

void reorder_pim_lut(lut_data_type* cpu_lut_table, lut_data_type* dpu_lut_table, LUTParams& lut_params)
{
    if(lut_params.lut_load_type != STATIC)
    {
        #pragma omp parallel for num_threads(lut_params.num_threads)
        for(uint32_t tmp_cb=0; tmp_cb<lut_params.num_codebook; ++tmp_cb)
        {
            for(uint32_t tmp_ct=0; tmp_ct<lut_params.num_centroid; ++tmp_ct)
            {
                for(uint32_t tmp_f=0; tmp_f<lut_params.output_feature_len; ++tmp_f)
                {
                    uint32_t tmp_f_load_tile_id = tmp_f / lut_params.feature_load_tile_size;
                    uint32_t tmp_f_intra_load_tile_id = tmp_f % lut_params.feature_load_tile_size;

                    uint32_t tmp_cpu_lut_offset = tmp_cb * lut_params.num_centroid * lut_params.output_feature_len
                                                + tmp_ct * lut_params.output_feature_len
                                                + tmp_f;
                    uint32_t tmp_dpu_lut_offset = tmp_f_load_tile_id * lut_params.num_codebook * lut_params.num_centroid * lut_params.feature_load_tile_size
                                                + tmp_cb * lut_params.num_centroid * lut_params.feature_load_tile_size
                                                + tmp_ct * lut_params.feature_load_tile_size
                                                + tmp_f_intra_load_tile_id;
                    dpu_lut_table[tmp_dpu_lut_offset] = cpu_lut_table[tmp_cpu_lut_offset];
                }
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(lut_params.num_threads)
        for(uint32_t tmp_cb=0; tmp_cb<lut_params.num_codebook; ++tmp_cb)
        {
            for(uint32_t tmp_ct=0; tmp_ct<lut_params.num_centroid; ++tmp_ct)
            {
                for(uint32_t tmp_f=0; tmp_f<lut_params.output_feature_len; ++tmp_f)
                {
                    uint32_t tmp_f_stile_id = tmp_f / lut_params.feature_stile_size;
                    uint32_t tmp_f_intra_stile_id = tmp_f % lut_params.feature_stile_size;

                    uint32_t tmp_cpu_lut_offset = tmp_cb * lut_params.num_centroid * lut_params.output_feature_len
                                                + tmp_ct * lut_params.output_feature_len
                                                + tmp_f;
                    uint32_t tmp_dpu_lut_offset = tmp_f_stile_id * lut_params.num_codebook * lut_params.num_centroid * lut_params.feature_stile_size
                                                + tmp_cb * lut_params.num_centroid * lut_params.feature_stile_size
                                                + tmp_ct * lut_params.feature_stile_size
                                                + tmp_f_intra_stile_id;
                    dpu_lut_table[tmp_dpu_lut_offset] = cpu_lut_table[tmp_cpu_lut_offset];
                }
            }
        }
    }
}

//////////////////////////////////////////////

// just for correctness check
void amm_cpu(float* input_tensor, float* centroid_tensor, lut_data_type* lut_table, float* bias_tensor, float* output_tensor,
                    IndexCalcParams index_params, LUTParams lut_params)
{
    // step 1: calculate index
    index_data_type* lut_input_index = new index_data_type[index_params.n * index_params.num_codebook];
    memset(lut_input_index, 0, sizeof(index_data_type) * index_params.n * index_params.num_codebook);

    #pragma omp parallel for num_threads(index_params.num_threads)
    for(uint32_t i=0; i<index_params.n; ++i)
    {
        float* distance_vec = new float[index_params.num_centroid];
        for(uint32_t tmp_codebook=0; tmp_codebook<index_params.num_codebook; ++tmp_codebook)
        {
            memset(distance_vec, 0, sizeof(float) * index_params.num_centroid);
            for(uint32_t tmp_centroid=0; tmp_centroid<index_params.num_centroid; ++tmp_centroid)
            {
                float tmp_dist = 0;
                for(uint32_t tmp_dim = 0; tmp_dim<index_params.sub_vec_len; ++tmp_dim)
                    tmp_dist += (input_tensor[i*index_params.input_feature_len + index_params.sub_vec_len*tmp_codebook + tmp_dim] - centroid_tensor[tmp_codebook*index_params.num_centroid*index_params.sub_vec_len + tmp_centroid*index_params.sub_vec_len + tmp_dim])
                                * (input_tensor[i*index_params.input_feature_len + index_params.sub_vec_len*tmp_codebook + tmp_dim] - centroid_tensor[tmp_codebook*index_params.num_centroid*index_params.sub_vec_len + tmp_centroid*index_params.sub_vec_len + tmp_dim]);
                distance_vec[tmp_centroid] = tmp_dist;
            }    

            float min_dist = distance_vec[0];
            index_data_type tmp_index = 0;
            for(uint32_t tmp_centroid=1; tmp_centroid<index_params.num_centroid; ++tmp_centroid)
            {
                if((min_dist - distance_vec[tmp_centroid]) > 1e-6)
                {
                    min_dist = distance_vec[tmp_centroid];
                    tmp_index = tmp_centroid;
                }
            }
            lut_input_index[i*index_params.num_codebook + tmp_codebook] = tmp_index;

        }
    }

    // step 2: lut
    output_data_type* lut_tensor = static_cast<output_data_type*>(aligned_alloc(64, lut_params.n * lut_params.output_feature_len * sizeof(output_data_type)));
    memset(lut_tensor, 0, sizeof(output_data_type) * lut_params.n * lut_params.output_feature_len);
    #pragma omp parallel for num_threads(lut_params.num_threads)
    for(uint32_t i=0; i<lut_params.n; ++i)
    {
        for(uint32_t tmp_codebook=0; tmp_codebook<lut_params.num_codebook; ++tmp_codebook)
        {
            for(uint32_t j=0; j<lut_params.output_feature_len; ++j)
                lut_tensor[i*lut_params.output_feature_len + j] += lut_table[tmp_codebook * lut_params.num_centroid * lut_params.output_feature_len + lut_input_index[i * lut_params.num_codebook + tmp_codebook] * lut_params.output_feature_len + j];
        }
    }

    // scale output to float values
    #pragma omp parallel for num_threads(lut_params.num_threads)
    for(int i=0; i<lut_params.n; ++i)
        for(int j=0; j<lut_params.output_feature_len; ++j)
            output_tensor[i*lut_params.output_feature_len+j] = lut_tensor[i*lut_params.output_feature_len+j]*lut_params.scale + lut_params.bias + bias_tensor[j];
    free(lut_tensor);
}

//////////////////////////////////////////////

void check_dpu_result(float* cpu_output_float_data, float* dpu_output_float_data, LUTParams lut_params)
{
    for(int i=0; i<lut_params.n; ++i)
    {
        for(int j=0; j<lut_params.output_feature_len; ++j)
        {
            int dpu_id = (int)(i / lut_params.n_stile_size) * lut_params.lut_parallelism + (int)(j / lut_params.feature_stile_size);
            int intra_dpu_row = i % lut_params.n_stile_size;
            int intra_dpu_dim = j % lut_params.feature_stile_size;
            int intra_dpu_mtile_id = intra_dpu_dim / lut_params.feature_mtile_size;
            int intra_dpu_intra_mtile_dim = intra_dpu_dim % lut_params.feature_mtile_size;

            int dpu_tensor_offset = dpu_id * lut_params.n_stile_size * lut_params.feature_stile_size 
                                  + intra_dpu_mtile_id * lut_params.n_stile_size * lut_params.feature_mtile_size 
                                  + intra_dpu_row * lut_params.feature_mtile_size 
                                  + intra_dpu_intra_mtile_dim;
            if(fabs((cpu_output_float_data[i*lut_params.output_feature_len + j] - dpu_output_float_data[dpu_tensor_offset])) > 1e-3)
            {
                cout << "check result failed at (i, j, dpu_id, intra_dpu_mtile_id, intra_dpu_row, intra_dpu_intra_mtile_dim)" 
                    << " " << i << " " << j << " "  << dpu_id << " " << intra_dpu_mtile_id << " " << intra_dpu_row << " " << intra_dpu_intra_mtile_dim << " "
                    << cpu_output_float_data[i*lut_params.output_feature_len + j] << " " << dpu_output_float_data[dpu_tensor_offset] << endl;
                exit(-1);
            }
        }
    }
    cout << "check dpu result success" << endl;
}

void dump_result(float* cpu_output, float* dpu_output, LUTParams lut_params)
{
    ofstream dpu_result_file("dpu_result.log");
    for(int i=0; i<lut_params.dpu_num; ++i)
    {
        for(int j=0; j<lut_params.n_stile_size; ++j)
        {
            for(int k=0; k<lut_params.feature_stile_size; ++k)
                dpu_result_file << dpu_output[i*lut_params.n_stile_size*lut_params.feature_stile_size + j*lut_params.feature_stile_size + k] << " ";
            dpu_result_file << endl;
        }
        dpu_result_file << endl;
    }

    ofstream cpu_result_file("cpu_result.log");
    for(int i=0; i<lut_params.n; ++i)
    {
        for(int j=0; j<lut_params.output_feature_len; ++j)
        {
            cpu_result_file << cpu_output[i*lut_params.output_feature_len + j] << " ";
        }
        cpu_result_file << endl;
    }
}


int main(int argc, char** argv)
{
    if(argc != 3)
    {
        printf("usage: %s kernel_config_yaml_path dpu_binary_path\n", argv[0]);
        exit(-1);
    }
    string kernel_config_yaml_path(argv[1]);
    string dpu_binary_path(argv[2]);

    AMMParams amm_params;
    parse_amm_configs(amm_params, kernel_config_yaml_path);

    float* input_tensor = static_cast<float*>(aligned_alloc(64, amm_params.index_params.n * amm_params.index_params.input_feature_len * sizeof(float)));
    float* centroid_tensor = static_cast<float*>(aligned_alloc(64, amm_params.index_params.num_codebook * amm_params.index_params.num_centroid * amm_params.index_params.sub_vec_len * sizeof(float)));
    float* bias_tensor = static_cast<float*>(aligned_alloc(64, amm_params.lut_params.output_feature_len * sizeof(float)));
    lut_data_type* cpu_lut_table = static_cast<lut_data_type*>(aligned_alloc(64, amm_params.lut_params.output_feature_len * amm_params.lut_params.num_codebook * amm_params.lut_params.num_centroid * sizeof(lut_data_type)));
    lut_data_type* dpu_lut_table = new lut_data_type[amm_params.lut_params.output_feature_len * amm_params.lut_params.num_codebook * amm_params.lut_params.num_centroid];
    memset(cpu_lut_table, 0, sizeof(lut_data_type) * amm_params.lut_params.output_feature_len * amm_params.lut_params.num_codebook * amm_params.lut_params.num_centroid);
    memset(dpu_lut_table, 0, sizeof(lut_data_type) * amm_params.lut_params.output_feature_len * amm_params.lut_params.num_codebook * amm_params.lut_params.num_centroid);
    float* dpu_output_tensor = static_cast<float*>(aligned_alloc(64, amm_params.lut_params.n * amm_params.lut_params.output_feature_len * sizeof(float)));
    float* cpu_output_tensor = new float[amm_params.lut_params.n * amm_params.lut_params.output_feature_len];
    memset(dpu_output_tensor, 0, sizeof(float) * amm_params.lut_params.n * amm_params.lut_params.output_feature_len);
    memset(cpu_output_tensor, 0, sizeof(float) * amm_params.lut_params.n * amm_params.lut_params.output_feature_len);

    generate_input_tensor(input_tensor, amm_params.index_params.n * amm_params.index_params.input_feature_len, amm_params.index_params.num_threads);
    generate_centroid_tensor(centroid_tensor, amm_params.index_params.num_codebook * amm_params.index_params.num_centroid * amm_params.index_params.sub_vec_len, amm_params.index_params.num_threads);
    generate_lut_data(cpu_lut_table, amm_params.lut_params.output_feature_len * amm_params.lut_params.num_codebook * amm_params.lut_params.num_centroid, amm_params.lut_params.num_threads);
    generate_bias_tensor(bias_tensor, amm_params.lut_params.output_feature_len, amm_params.lut_params.num_threads);
    reorder_pim_lut(cpu_lut_table, dpu_lut_table, amm_params.lut_params);
    cout << "data generation finished" << endl;

    double time1, time2;
    time1 = W_time();
    dpu_set_t dpu_set;
    uint32_t allocated_dpu_num;
    if(amm_params.lut_params.dpu_num == NR_DPUS_PER_RANK)
        allocated_dpu_num = allocate_rank(&dpu_set, amm_params.lut_params.dpu_num / NR_DPUS_PER_RANK);
    else
        allocated_dpu_num = allocate_dpu(&dpu_set, amm_params.lut_params.dpu_num);
    time2 = W_time() - time1;
    cout << "dpu allocate time is " << time2 << endl;
    time1 = W_time();
    load_binary(&dpu_set, dpu_binary_path);
    time2 = W_time() - time1;
    cout << "binary load time is " << time2 << endl;

    double pim_total = 0.0;
    double cpu_total = 0.0;
    for(uint32_t tmp_round=0; tmp_round<WARMUP_ROUND+RECORD_ROUND; ++tmp_round)
    {
        cout << "---------- ROUND " << tmp_round << " ----------" << endl;

        time1 = W_time();
        amm_host(&dpu_set, input_tensor, centroid_tensor, dpu_lut_table, bias_tensor, dpu_output_tensor, amm_params.index_params, amm_params.lut_params);
        time2 = W_time() - time1;
        cout << "PIM AMM time " << time2 << endl;
        if(tmp_round>=WARMUP_ROUND)
            pim_total += time2;

#ifdef RUN_VERIFICATION
        time1 = W_time();
        amm_cpu(input_tensor, centroid_tensor, cpu_lut_table, bias_tensor, cpu_output_tensor, amm_params.index_params, amm_params.lut_params);
        time2 = W_time() - time1;
        cout << "CPU AMM time " << time2 << endl;
        if(tmp_round>=WARMUP_ROUND)
            cpu_total += time2;

        check_dpu_result(cpu_output_tensor, dpu_output_tensor, amm_params.lut_params);
#endif
    }

    cout << "warm up rounds " << WARMUP_ROUND << ", record rounds " << RECORD_ROUND << endl;
    cout << "PIM AVG AMM time " << pim_total/RECORD_ROUND << endl;
#ifdef RUN_VERIFICATION
    cout << "CPU AVG AMM time " << cpu_total/RECORD_ROUND << endl;
#endif

    return 0;
}
