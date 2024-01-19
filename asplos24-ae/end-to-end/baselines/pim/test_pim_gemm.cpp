#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <cassert>
#include <limits>
#include <omp.h>
#include <memory.h>

#include "dpu_common.h"
#include "utils.h"
extern "C"{
    #include <dpu.h>
}

using namespace std;

typedef float data_type;

#define WARMUP_ROUND 5
#define RECORD_ROUND 5

////////////////////////////////////////////// Data Generation

void generate_data(data_type* tensor, uint32_t element_num, uint32_t num_threads)
{
    // std::random_device rd;  
    std::mt19937 gen(233);

    uniform_real_distribution<data_type> dis(-5.0f, 5.0f);
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<element_num; ++i)
        tensor[i] = dis(gen);
}


int main(int argc, char** argv)
{
    if(argc != 2)
    {
        printf("usage: %s dpu_binary_path\n", argv[0]);
        exit(-1);
    }
    string dpu_binary_path(argv[1]);

    data_type* input_matrix = new data_type[ROW * HIDDEN_DIM];
    data_type* weight_matrix = new data_type[FEATURE_LEN * HIDDEN_DIM];
    data_type* output_matrix = new data_type[ROW * FEATURE_LEN];
    uint32_t input_element_num = ROW * HIDDEN_DIM;
    uint32_t weight_element_num = FEATURE_LEN * HIDDEN_DIM;
    uint32_t output_element_num = ROW * FEATURE_LEN;
    generate_data(input_matrix, input_element_num, 40);
    generate_data(weight_matrix, weight_element_num, 40);

    uint32_t input_offset = N_STILE_SIZE * HIDDEN_DIM;
    uint32_t weight_offset = FEATURE_STILE_SIZE * HIDDEN_DIM;
    uint32_t output_offset = N_STILE_SIZE *FEATURE_STILE_SIZE;

    printf("%d %d %d %d %d %d\n", ROW, HIDDEN_DIM, FEATURE_LEN, DPU_NUM, INPUT_PARALLELISM, LUT_PARALLELISM);
    printf("%d %d %d %d %d %d\n", input_element_num, weight_element_num, output_element_num, input_offset, weight_offset, output_offset);

    double time1, time2;
    time1 = W_time();
    dpu_set_t dpu_set;
    uint32_t allocated_dpu_num;
    // if(DPU_NUM % NR_DPUS_PER_RANK == 0)
    if(DPU_NUM == NR_DPUS_PER_RANK)
        allocated_dpu_num = allocate_rank(&dpu_set, DPU_NUM / NR_DPUS_PER_RANK);
    else
        allocated_dpu_num = allocate_dpu(&dpu_set, DPU_NUM);
    time2 = W_time() - time1;
    cout << "dpu allocate time is " << time2 << endl;

    double total_time = 0.;
    for(uint32_t tmp_round=0; tmp_round<WARMUP_ROUND+RECORD_ROUND; ++tmp_round)
    {
        cout << "---------- ROUND " << tmp_round << " ----------" << endl;
        time1 = W_time();
        load_binary(&dpu_set, dpu_binary_path);
        time2 = W_time() - time1;
        cout << "binary load time is " << time2 << endl;

        uint32_t each_dpu;
        dpu_set_t dpu;

        double send_input_time, send_weight_time, kernel_time, read_output_time;

        time1 = W_time();
        // send input
        DPU_FOREACH(dpu_set, dpu, each_dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &input_matrix[input_offset * (uint32_t)(each_dpu / LUT_PARALLELISM)]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_matrix", 0, sizeof(data_type) * input_offset, DPU_XFER_ASYNC));
        dpu_sync(dpu_set);
        send_input_time = W_time() - time1;

        time1 = W_time();
        // send weight
        DPU_FOREACH(dpu_set, dpu, each_dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &weight_matrix[weight_offset * (uint32_t)(each_dpu % LUT_PARALLELISM)]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "weight_matrix", 0, sizeof(data_type) * weight_offset, DPU_XFER_ASYNC));
        dpu_sync(dpu_set);
        send_weight_time = W_time() - time1;

        time1 = W_time();
        // compute
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
        kernel_time = W_time() - time1;

        time1 = W_time();
        // fetch output
        DPU_FOREACH(dpu_set, dpu, each_dpu)
        {
            DPU_ASSERT(dpu_prepare_xfer(dpu, &output_matrix[output_offset * each_dpu]));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "output_matrix", 0, sizeof(data_type) * output_offset, DPU_XFER_ASYNC));
        dpu_sync(dpu_set);
        read_output_time = W_time() - time1;

        printf("send input time %.6f, send weight time %.6f, read output time %.6f, kernel time %.6f\n", send_input_time, send_weight_time, read_output_time, kernel_time);
        if(tmp_round>=WARMUP_ROUND)
            total_time += send_input_time + send_weight_time + read_output_time + kernel_time;
    
    }
    cout << "warm up rounds " << WARMUP_ROUND << ", record rounds " << RECORD_ROUND << endl;
    cout << "PIM AVG GEMM time " << total_time/RECORD_ROUND << endl;

    free_dpu_set(&dpu_set);

    return 0;
}
