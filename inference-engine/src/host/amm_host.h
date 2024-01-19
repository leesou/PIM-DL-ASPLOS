#pragma once
#include "defines.h"
extern "C"{
    #include <dpu.h>
}


void amm_host(dpu_set_t* dpu_set,
              float* input_tensor, float* centroid_tensor, lut_data_type* lut_table, float* bias_tensor, float* output_tensor,
              IndexCalcParams index_params, LUTParams lut_params);

