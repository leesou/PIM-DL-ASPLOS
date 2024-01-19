#pragma once
extern "C"{
    #include <dpu.h>
}
#include "defines.h"

void prepare_parameters(dpu_set_t* dpu_set, dpu_arguments_t* dpu_arguments);

void prepare_lut_table(LUTParams lut_params, dpu_set_t* dpu_set, lut_data_type* lut_table);

void prepare_input_index(LUTParams lut_params, dpu_set_t* dpu_set, index_data_type* input_index);

void pim_lut(LUTParams lut_params, dpu_set_t* dpu_set,
             index_data_type* input_index, lut_data_type* lut_table, float* bias_tensor, float* output_tensor);

