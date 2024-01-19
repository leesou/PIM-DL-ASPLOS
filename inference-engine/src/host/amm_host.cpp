#include "amm_host.h"
#include "index_calc_host.h"
#include "pim_lut_host.h"
#include "utils.h"
#include <iostream>
#include <immintrin.h>
#include <stdlib.h>


void amm_host(dpu_set_t* dpu_set,
              float* input_tensor, float* centroid_tensor, lut_data_type* lut_table, float* bias_tensor, float* output_tensor,
              IndexCalcParams index_params, LUTParams lut_params)
{
#ifdef AMM_BREAKDOWN
    double time1, time2;
    time1 = W_time();
#endif
    // step 1: calculate input index of lut operation
    index_data_type * lut_input_index = new index_data_type[index_params.n * index_params.num_codebook];
    memset(lut_input_index, 0, sizeof(index_data_type) * index_params.n * index_params.num_codebook);
    index_calculation(input_tensor, centroid_tensor, lut_input_index, index_params);
#ifdef AMM_BREAKDOWN
    time2 = W_time();
    amm_profiles.index_calc_latency += time2 - time1;
    printf("index calculation time %.6f\n", amm_profiles.index_calc_latency);
#endif

    // step 2: conduct pim lut operation
    pim_lut(lut_params, dpu_set, lut_input_index, lut_table, bias_tensor, output_tensor);
}

