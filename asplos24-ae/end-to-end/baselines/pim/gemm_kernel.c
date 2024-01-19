#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <barrier.h>

typedef float data_type;

#define ROWS_PER_TASKLET ((N_MTILE_SIZE) / (NR_TASKLETS))

#define INPUT_ELEMENT_NUM ((N_STILE_SIZE) * (HIDDEN_DIM))
#define WEIGHT_ELEMENT_NUM ((FEATURE_STILE_SIZE) * (HIDDEN_DIM))
#define OUTPUT_ELEMENT_NUM ((N_STILE_SIZE) * (FEATURE_STILE_SIZE))

#define INPUT_BUFFER_ELEMENT_NUM ((N_MTILE_SIZE) * (HIDDEN_MTILE_SIZE))
#define WEIGHT_BUFFER_ELEMENT_NUM ((FEATURE_MTILE_SIZE) * (HIDDEN_MTILE_SIZE))
#define OUTPUT_BUFFER_ELEMENT_NUM ((N_MTILE_SIZE) * (FEATURE_MTILE_SIZE))

/*------------------input matrix-----------------------*/
__mram_noinit data_type input_matrix[INPUT_ELEMENT_NUM];

/*------------------weight matrix-----------------------*/
__mram_noinit data_type weight_matrix[WEIGHT_ELEMENT_NUM];

/*------------------output matrix-----------------------*/
__mram_noinit data_type output_matrix[OUTPUT_ELEMENT_NUM];


/*------------------WRAM Buffers (shared by all tasklets)-----------------------*/
__dma_aligned data_type input_buffer[INPUT_BUFFER_ELEMENT_NUM];
__dma_aligned data_type weight_buffer[WEIGHT_BUFFER_ELEMENT_NUM];
__dma_aligned data_type output_buffer[OUTPUT_BUFFER_ELEMENT_NUM];


BARRIER_INIT(load_barrier, NR_TASKLETS);
BARRIER_INIT(load_barrier1, NR_TASKLETS);


int main()
{
    // tmp taklet id
    uint32_t tasklet_id = me();

    uint32_t input_buffer_offset = ((INPUT_BUFFER_ELEMENT_NUM) / NR_TASKLETS) * tasklet_id;
    uint32_t weight_buffer_offset = ((WEIGHT_BUFFER_ELEMENT_NUM) / NR_TASKLETS) * tasklet_id;
    uint32_t output_buffer_offset = ((OUTPUT_BUFFER_ELEMENT_NUM) / NR_TASKLETS) * tasklet_id;

    uint32_t input_mram_offset = 0;
    uint32_t output_mram_offset = 0;

    for(uint32_t tmp_row=0; tmp_row<N_STILE_SIZE; tmp_row+=N_MTILE_SIZE)
    {
        for(uint32_t tmp_feature=0; tmp_feature<FEATURE_STILE_SIZE; tmp_feature+=FEATURE_MTILE_SIZE)
        {
            // reset output
            for(uint32_t tmp_offset=0; tmp_offset<output_buffer_offset; tmp_offset++)
                output_buffer[output_buffer_offset+tmp_offset] = 0;

            uint32_t weight_mram_offset = 0;
            uint32_t input_mram_offset1 = 0;
            for(uint32_t tmp_hidden_dim=0; tmp_hidden_dim<HIDDEN_DIM; tmp_hidden_dim+=HIDDEN_MTILE_SIZE)
            {
                // load input
                // mram_read(&input_matrix[input_mram_offset+input_mram_offset1+input_buffer_offset], &input_buffer[input_buffer_offset], ((INPUT_BUFFER_ELEMENT_NUM * sizeof(data_type)) / NR_TASKLETS));
                // load weight
                // mram_read(&weight_matrix[weight_mram_offset+weight_buffer_offset], &weight_buffer[weight_buffer_offset], ((WEIGHT_BUFFER_ELEMENT_NUM * sizeof(data_type)) / NR_TASKLETS));
            
                for(uint32_t tmp_intra_row=0; tmp_intra_row<ROWS_PER_TASKLET; ++tmp_intra_row)
                {
                    uint32_t tmp_input_offset = input_buffer_offset + tmp_intra_row * HIDDEN_MTILE_SIZE;
                    for(uint32_t tmp_intra_feature=0; tmp_intra_feature<FEATURE_MTILE_SIZE; ++tmp_intra_feature)
                    {
                        uint32_t tmp_weight_offset = tmp_intra_feature * HIDDEN_MTILE_SIZE;
                        uint32_t tmp_output_offset = output_buffer_offset + tmp_intra_row * FEATURE_MTILE_SIZE + tmp_intra_feature;
                        for(uint32_t tmp_intra_hidden_dim=0; tmp_intra_hidden_dim<HIDDEN_MTILE_SIZE; tmp_intra_hidden_dim++)
                        {
                            output_buffer[tmp_output_offset] += input_buffer[tmp_input_offset+tmp_intra_hidden_dim] * weight_buffer[tmp_weight_offset+tmp_intra_hidden_dim];
                        }
                    }
                }

                weight_mram_offset += WEIGHT_BUFFER_ELEMENT_NUM;
                input_mram_offset1 += INPUT_BUFFER_ELEMENT_NUM;
            }  

            // save output
            mram_write(&output_buffer[output_buffer_offset], &output_matrix[output_mram_offset+output_buffer_offset], ((OUTPUT_BUFFER_ELEMENT_NUM * sizeof(data_type)) / NR_TASKLETS));
            output_mram_offset += OUTPUT_BUFFER_ELEMENT_NUM;
        }

        input_mram_offset += ((N_MTILE_SIZE) * (HIDDEN_DIM));
    }

    return 0;
}
