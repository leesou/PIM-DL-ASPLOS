#ifndef DPU_CONFIG_H


typedef int8_t lut_data_type;
typedef uint16_t index_data_type;
typedef int32_t output_data_type;

// need to change with the typedef above accordingly
#define INDEX_SIZE 2
#define LUT_SIZE 1
#define OUTPUT_SIZE 4

typedef struct {
    uint32_t input_height;
} dpu_arguments_t;


#define DPU_CONFIG_H
#endif
