#include <mram.h>
#include <alloc.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <barrier.h>

#include "dpu_configs.h"





#define INPUT_STILE_SIZE (N_STILE_SIZE * CB_MTILE_SIZE)
#define OUTPUT_STILE_SIZE (N_STILE_SIZE * FEATURE_MTILE_SIZE)
#define INPUT_MTILE_SIZE (N_MTILE_SIZE * CB_MTILE_SIZE)
#define OUTPUT_MTILE_SIZE (N_MTILE_SIZE * FEATURE_MTILE_SIZE)
#define LUT_MTILE_SIZE (NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE)
#define LUT_LOAD_MTILE_SIZE (NUM_CODEBOOK * NUM_CENTROID * FEATURE_LOAD_TILE_SIZE)
#define LUT_LOAD_MMTILE_SIZE (CB_MTILE_SIZE * NUM_CENTROID * FEATURE_LOAD_TILE_SIZE)
#define LUT_LOAD_TILE_SIZE (CB_LOAD_TILE_SIZE * NUM_CENTROID * FEATURE_LOAD_TILE_SIZE)
#define LUT_LOAD_CBTILE_SIZE (NUM_CENTROID * FEATURE_LOAD_TILE_SIZE)
#define LUT_CBTILE_SIZE (NUM_CENTROID * FEATURE_STILE_SIZE)
#define LUT_CBMTILE_SIZE (CB_MTILE_SIZE * NUM_CENTROID * FEATURE_STILE_SIZE)
#define INPUT_BUFFER_SIZE_PER_TASKLET (INPUT_MTILE_SIZE / NR_TASKLETS)
#define OUTPUT_BUFFER_SIZE_PER_TASKLET (OUTPUT_MTILE_SIZE / NR_TASKLETS)
#ifdef STATIC_LUT_TABLE
#define LUT_BUFFER_SIZE_PER_TASKLET (FEATURE_STILE_SIZE * NUM_CODEBOOK * NUM_CENTROID / NR_TASKLETS)
#elif defined(FINE_GRAIN)
#define LUT_BUFFER_SIZE_PER_TASKLET (FEATURE_LOAD_TILE_SIZE)
#else
#define LUT_BUFFER_SIZE_PER_TASKLET (FEATURE_LOAD_TILE_SIZE * CB_LOAD_TILE_SIZE * NUM_CENTROID / NR_TASKLETS)
#endif
#define INPUT_BUFFER_BYTE_PER_TASKLET  (INPUT_BUFFER_SIZE_PER_TASKLET * INDEX_SIZE)
#define OUTPUT_BUFFER_BYTE_PER_TASKLET (OUTPUT_BUFFER_SIZE_PER_TASKLET * OUTPUT_SIZE)
#define LUT_BUFFER_BYTE_PER_TASKLET (LUT_BUFFER_SIZE_PER_TASKLET * LUT_SIZE)
#define MAX_INPUT_PER_RW (2048 / INDEX_SIZE)
#define MAX_OUTPUT_PER_RW (2048 / OUTPUT_SIZE)
#define MAX_LUT_PER_RW (2048 / LUT_SIZE)
#define N_PER_TASKLET (N_MTILE_SIZE / NR_TASKLETS)




/*------------------LUT table-----------------------*/
__mram_noinit lut_data_type lut_table[FEATURE_STILE_SIZE * NUM_CODEBOOK * NUM_CENTROID];

/*------------------Index data-----------------------*/
__mram_noinit index_data_type input_index[N_STILE_SIZE * NUM_CODEBOOK];

/*------------------Output data-----------------------*/
__mram_noinit output_data_type output_data[N_STILE_SIZE * FEATURE_STILE_SIZE];

/*------------------Operation Parameters-----------------------*/
__host dpu_arguments_t dpu_arguments;




/*------------------WRAM Buffers (shared by all tasklets)-----------------------*/
__dma_aligned index_data_type input_index_buffer[INPUT_MTILE_SIZE];
__dma_aligned output_data_type output_result_buffer[OUTPUT_MTILE_SIZE];
#ifdef STATIC_LUT_TABLE
__dma_aligned lut_data_type lut_table_buffer[FEATURE_STILE_SIZE * NUM_CODEBOOK * NUM_CENTROID];
#elif defined(FINE_GRAIN)
__dma_aligned lut_data_type lut_table_buffer[FEATURE_LOAD_TILE_SIZE * NR_TASKLETS];
#else
__dma_aligned lut_data_type lut_table_buffer[FEATURE_LOAD_TILE_SIZE * CB_LOAD_TILE_SIZE * NUM_CENTROID];
#endif




/*------------------Barriers for tasklet sync (shared by all tasklets)-----------------------*/
BARRIER_INIT(lut_load_barrier, NR_TASKLETS);
BARRIER_INIT(lut_load_barrier1, NR_TASKLETS);




void lut_kernel();




int main()
{
    lut_kernel();
    return 0;
}



#ifdef LOOP_ORDER_NFC

void lut_kernel()
{
    // tmp taklet id
    uint32_t tasklet_id = me();

#ifdef STATIC_LUT_TABLE
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // load lut table first
    uint32_t lut_tensor_offset = lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
    uint32_t tmp_lut_offset = 0;
    uint32_t tmp_lut_byte_offset = 0;
    for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
    {
        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
        tmp_lut_offset += MAX_LUT_PER_RW;
    }
    mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
    mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif
    barrier_wait(&lut_load_barrier);

    // tile offset, changing along with loop iteration
    uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
    uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE 
    for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
    {
        uint32_t output_stile_offset = 0;
        for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
        {
            // reset output buffer
            // when oc is the innermost outer iterator, we can just reset output buffer instead of loading output data
            for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                output_result_buffer[tmp_output_offset] = 0;

            uint32_t input_stile_offset = 0;
            uint32_t lut_cbmtile_offset = 0;
            for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // read lut and compute
                uint32_t tmp_input_row_offset = 0;
                uint32_t tmp_output_row_offset = 0;
                for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                {
                    uint32_t lut_cbtile_offset = 0;
                    for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                    {
                        uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];
                        for(uint32_t tmp_f=0; tmp_f<FEATURE_MTILE_SIZE; ++tmp_f)
                        {
                            output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_f] += lut_table_buffer[lut_cbmtile_offset + lut_cbtile_offset + tmp_index + tmp_init_feature + tmp_f];
                        }

                        lut_cbtile_offset += LUT_CBTILE_SIZE;
                    }

                    tmp_input_row_offset += CB_MTILE_SIZE;
                    tmp_output_row_offset += FEATURE_MTILE_SIZE;
                }

                // update input's stile offset
                input_stile_offset += INPUT_STILE_SIZE;
                lut_cbmtile_offset += LUT_CBMTILE_SIZE;
            }

            // save output buffer to DRAM
            uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_output_offset = 0;
            uint32_t tmp_output_byte_offset = 0;
            for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
            {
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                tmp_output_offset += MAX_OUTPUT_PER_RW;
            }
            mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
            mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            // update output's stile offset
            output_stile_offset += OUTPUT_STILE_SIZE;
        }

        // update input & output intra each stile's mtile offset
        input_mtile_offset += INPUT_MTILE_SIZE;
        output_mtile_offset += OUTPUT_MTILE_SIZE;
    }
#elif defined(FINE_GRAIN)
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // tile offset, changing along with loop iteration
    uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
    uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE 
    for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
    {
        uint32_t output_stile_offset = 0;
        uint32_t lut_mtile_offset = 0;
        for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
        {
            // reset output buffer
            // when oc is the innermost outer iterator, we can just reset output buffer instead of loading output data
            for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                output_result_buffer[tmp_output_offset] = 0;
            
            uint32_t input_stile_offset = 0;
            uint32_t lut_load_mmtile_offset = 0;
            for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    // all tasklets compute the same row
                    uint32_t tmp_input_row_offset = 0;
                    uint32_t tmp_output_row_offset = 0;
                    for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                    {
                        uint32_t lut_load_cbtile_offset = 0;
                        for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                        {
                            uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];

                            // load lut table
                            uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_cbtile_offset + tmp_index;
                            mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);

                            // conduct computation
                            for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                            {
                                output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[lut_buffer_offset + tmp_f];
                            }

                            lut_load_cbtile_offset += LUT_LOAD_CBTILE_SIZE;
                        }

                        tmp_input_row_offset += CB_MTILE_SIZE;
                        tmp_output_row_offset += FEATURE_MTILE_SIZE;
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // update lut load mmtile offset
                lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
                // update input's stile offset
                input_stile_offset += INPUT_STILE_SIZE;
            }

            // save output buffer to DRAM
            uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_output_offset = 0;
            uint32_t tmp_output_byte_offset = 0;
            for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
            {
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                tmp_output_offset += MAX_OUTPUT_PER_RW;
            }
            mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
            mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            // update lut table mtile's offset
            lut_mtile_offset += LUT_MTILE_SIZE;
            // update output's stile offset
            output_stile_offset += OUTPUT_STILE_SIZE;
        }
    
        // update input & output intra each stile's mtile offset
        input_mtile_offset += INPUT_MTILE_SIZE;
        output_mtile_offset += OUTPUT_MTILE_SIZE;
    }
#else
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // tile offset, changing along with loop iteration
    uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
    uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE 
    for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
    {
        uint32_t output_stile_offset = 0;
        uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
        for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
        {
            // reset output buffer
            // when oc is the innermost outer iterator, we can just reset output buffer instead of loading output data
            for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                output_result_buffer[tmp_output_offset] = 0;

            uint32_t input_stile_offset = 0;
            uint32_t lut_load_mmtile_offset = 0;
            for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    uint32_t lut_load_tile_offset = 0;
                    for(uint32_t tmp_init_load_cb=0; tmp_init_load_cb<CB_MTILE_SIZE; tmp_init_load_cb+=CB_LOAD_TILE_SIZE)
                    {
                        // load lut table if it is stored in MRAM
                        uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_tile_offset + lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
                        uint32_t tmp_lut_offset = 0;
                        uint32_t tmp_lut_byte_offset = 0;
                        for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
                        {
                            mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
                            tmp_lut_offset += MAX_LUT_PER_RW;
                        }
                        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                        mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif

                        // add a memory barrier to ensure all tasklets finishes LUT read
			            barrier_wait(&lut_load_barrier);
                        uint32_t tmp_input_row_offset = 0;
                        uint32_t tmp_output_row_offset = 0;
                        // conduct computation
                        for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                        {
                            for(uint32_t tmp_cb=0; tmp_cb<CB_LOAD_TILE_SIZE; ++tmp_cb)
                            {
                                uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_init_load_cb + tmp_cb];
                                for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                                {
                                    output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[tmp_index + tmp_f];
                                }
                            }

                            tmp_input_row_offset += CB_MTILE_SIZE;
                            tmp_output_row_offset += FEATURE_MTILE_SIZE;
                        }
                        // add a memory barrier to ensure all computation have been finished before reading new lut data
                        barrier_wait(&lut_load_barrier1);

                        // update lut load tile offset
                        lut_load_tile_offset += LUT_LOAD_TILE_SIZE;                        
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // update lut load mmtile offset
                lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
                // update input's stile offset
                input_stile_offset += INPUT_STILE_SIZE;
            }

            // save output buffer to DRAM
            uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_output_offset = 0;
            uint32_t tmp_output_byte_offset = 0;
            for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
            {
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                tmp_output_offset += MAX_OUTPUT_PER_RW;
            }
            mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
            mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            // update lut table mtile's offset
            lut_mtile_offset += LUT_MTILE_SIZE;
            // update output's stile offset
            output_stile_offset += OUTPUT_STILE_SIZE;
        }
        
        // update input & output intra each stile's mtile offset
        input_mtile_offset += INPUT_MTILE_SIZE;
        output_mtile_offset += OUTPUT_MTILE_SIZE;
    }
#endif

}

#elif defined(LOOP_ORDER_NCF)

void lut_kernel()
{
    // tmp taklet id
    uint32_t tasklet_id = me();

#ifdef STATIC_LUT_TABLE
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // load lut table first
    uint32_t lut_tensor_offset = lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
    uint32_t tmp_lut_offset = 0;
    uint32_t tmp_lut_byte_offset = 0;
    for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
    {
        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
        tmp_lut_offset += MAX_LUT_PER_RW;
    }
    mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
    mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif
    barrier_wait(&lut_load_barrier);

    // tile offset, changing along with loop iteration
    uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
    uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE 
    for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
    {
        uint32_t input_stile_offset = 0;
        uint32_t lut_cbmtile_offset = 0;
        for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
        {
            uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
            // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_input_offset = 0;
            uint32_t tmp_input_byte_offset = 0;
            for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
            {
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                tmp_input_offset += MAX_INPUT_PER_RW;
            }    
            mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
            mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            uint32_t output_stile_offset = 0;
            for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
            {
                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // read lut and compute
                uint32_t tmp_input_row_offset = 0;
                uint32_t tmp_output_row_offset = 0;
                for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                {
                    uint32_t lut_cbtile_offset = 0;
                    for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                    {
                        uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];
                        for(uint32_t tmp_f=0; tmp_f<FEATURE_MTILE_SIZE; ++tmp_f)
                        {
                            output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_f] += lut_table_buffer[lut_cbmtile_offset + lut_cbtile_offset + tmp_index + tmp_init_feature + tmp_f];
                        }

                        lut_cbtile_offset += LUT_CBTILE_SIZE;
                    }

                    tmp_input_row_offset += CB_MTILE_SIZE;
                    tmp_output_row_offset += FEATURE_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update output's stile offset
                output_stile_offset += OUTPUT_STILE_SIZE;
            }

            // update input's stile offset
            input_stile_offset += INPUT_STILE_SIZE;
            lut_cbmtile_offset += LUT_CBMTILE_SIZE;
        }

        // update input & output intra each stile's mtile offset
        input_mtile_offset += INPUT_MTILE_SIZE;
        output_mtile_offset += OUTPUT_MTILE_SIZE;
    }

#elif defined(FINE_GRAIN)
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    // tile offset, changing along with loop iteration
    uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
    uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE 
    for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
    {
        uint32_t input_stile_offset = 0;
        uint32_t lut_load_mmtile_offset = 0;
        for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
        {
            uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
            // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_input_offset = 0;
            uint32_t tmp_input_byte_offset = 0;
            for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
            {
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                tmp_input_offset += MAX_INPUT_PER_RW;
            }    
            mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
            mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            uint32_t output_stile_offset = 0;
            uint32_t lut_mtile_offset = 0;
            for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
            {
                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    // all tasklets compute the same row
                    uint32_t tmp_input_row_offset = 0;
                    uint32_t tmp_output_row_offset = 0;
                    for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                    {
                        uint32_t lut_load_cbtile_offset = 0;
                        for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                        {
                            uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];

                            // load lut table
                            uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_cbtile_offset + tmp_index;
                            mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);

                            // conduct computation
                            for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                            {
                                output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[lut_buffer_offset + tmp_f];
                            }

                            lut_load_cbtile_offset += LUT_LOAD_CBTILE_SIZE;
                        }

                        tmp_input_row_offset += CB_MTILE_SIZE;
                        tmp_output_row_offset += FEATURE_MTILE_SIZE;
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }


                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif


                // update lut table mtile's offset
                lut_mtile_offset += LUT_MTILE_SIZE;
                // update output's stile offset
                output_stile_offset += OUTPUT_STILE_SIZE;
            }

            // update lut load mmtile offset
            lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
            // update input's stile offset
            input_stile_offset += INPUT_STILE_SIZE;
        }

        // update input & output intra each stile's mtile offset
        input_mtile_offset += INPUT_MTILE_SIZE;
        output_mtile_offset += OUTPUT_MTILE_SIZE;
    }

#else
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // tile offset, changing along with loop iteration
    uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
    uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
    for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
    {
        uint32_t input_stile_offset = 0;
        uint32_t lut_load_mmtile_offset = 0;
        for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
        {
            uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
            // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_input_offset = 0;
            uint32_t tmp_input_byte_offset = 0;
            for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
            {
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                tmp_input_offset += MAX_INPUT_PER_RW;
            }    
            mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
            mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            uint32_t output_stile_offset = 0;
            uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
            for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
            {
                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    uint32_t lut_load_tile_offset = 0;
                    for(uint32_t tmp_init_load_cb=0; tmp_init_load_cb<CB_MTILE_SIZE; tmp_init_load_cb+=CB_LOAD_TILE_SIZE)
                    {
                        // load lut table if it is stored in MRAM
                        uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_tile_offset + lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
                        uint32_t tmp_lut_offset = 0;
                        uint32_t tmp_lut_byte_offset = 0;
                        for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
                        {
                            mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
                            tmp_lut_offset += MAX_LUT_PER_RW;
                        }
                        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                        mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif

                        // add a memory barrier to ensure all tasklets finishes LUT read
			            barrier_wait(&lut_load_barrier);
                        uint32_t tmp_input_row_offset = 0;
                        uint32_t tmp_output_row_offset = 0;
                        // conduct computation
                        for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                        {
                            for(uint32_t tmp_cb=0; tmp_cb<CB_LOAD_TILE_SIZE; ++tmp_cb)
                            {
                                uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_init_load_cb + tmp_cb];
                                for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                                {
                                    output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[tmp_index + tmp_f];
                                }
                            }

                            tmp_input_row_offset += CB_MTILE_SIZE;
                            tmp_output_row_offset += FEATURE_MTILE_SIZE;
                        }
                        // add a memory barrier to ensure all computation have been finished before reading new lut data
                        barrier_wait(&lut_load_barrier1);

                        // update lut load tile offset
                        lut_load_tile_offset += LUT_LOAD_TILE_SIZE;                        
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update lut table mtile's offset
                lut_mtile_offset += LUT_MTILE_SIZE;
                // update output's stile offset
                output_stile_offset += OUTPUT_STILE_SIZE;
            }

            // update lut load mmtile offset
            lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
            // update input's stile offset
            input_stile_offset += INPUT_STILE_SIZE;
        }

        // update input & output intra each stile's mtile offset
        input_mtile_offset += INPUT_MTILE_SIZE;
        output_mtile_offset += OUTPUT_MTILE_SIZE;
    }
#endif

}

#elif defined(LOOP_ORDER_FNC)

void lut_kernel()
{
    // tmp taklet id
    uint32_t tasklet_id = me();

#ifdef STATIC_LUT_TABLE
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // load lut table first
    uint32_t lut_tensor_offset = lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
    uint32_t tmp_lut_offset = 0;
    uint32_t tmp_lut_byte_offset = 0;
    for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
    {
        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
        tmp_lut_offset += MAX_LUT_PER_RW;
    }
    mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
    mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif
    barrier_wait(&lut_load_barrier);

    uint32_t output_stile_offset = 0;
    for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
    {
        // tile offset, changing along with loop iteration
        uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
        uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE 
        for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
        {
            // reset output buffer
            // when oc is the innermost outer iterator, we can just reset output buffer instead of loading output data
            for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                output_result_buffer[tmp_output_offset] = 0;
            
            uint32_t input_stile_offset = 0;
            uint32_t lut_cbmtile_offset = 0;
            for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load lut and compute
                uint32_t tmp_input_row_offset = 0;
                uint32_t tmp_output_row_offset = 0;
                for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                {
                    uint32_t lut_cbtile_offset = 0;
                    for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                    {
                        uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];
                        for(uint32_t tmp_f=0; tmp_f<FEATURE_MTILE_SIZE; ++tmp_f)
                        {
                            output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_f] += lut_table_buffer[lut_cbmtile_offset + lut_cbtile_offset + tmp_index + tmp_init_feature + tmp_f];
                        }

                        lut_cbtile_offset += LUT_CBTILE_SIZE;
                    }

                    tmp_input_row_offset += CB_MTILE_SIZE;
                    tmp_output_row_offset += FEATURE_MTILE_SIZE;
                }

                // update input's stile offset
                input_stile_offset += INPUT_STILE_SIZE;
                lut_cbmtile_offset += LUT_CBMTILE_SIZE;
            }

            // save output buffer to DRAM
            uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_output_offset = 0;
            uint32_t tmp_output_byte_offset = 0;
            for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
            {
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                tmp_output_offset += MAX_OUTPUT_PER_RW;
            }
            mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
            mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            // update input & output intra each stile's mtile offset
            input_mtile_offset += INPUT_MTILE_SIZE;
            output_mtile_offset += OUTPUT_MTILE_SIZE;
        }

        // update output's stile offset
        output_stile_offset += OUTPUT_STILE_SIZE;
    }
#elif defined(FINE_GRAIN)
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    uint32_t output_stile_offset = 0;
    uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
    for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
    {
        // tile offset, changing along with loop iteration
        uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
        uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE 
        for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
        {
            // reset output buffer
            // when oc is the innermost outer iterator, we can just reset output buffer instead of loading output data
            for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                output_result_buffer[tmp_output_offset] = 0;
            
            uint32_t input_stile_offset = 0;
            uint32_t lut_load_mmtile_offset = 0;
            for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    // all tasklets compute the same row
                    uint32_t tmp_input_row_offset = 0;
                    uint32_t tmp_output_row_offset = 0;
                    for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                    {
                        uint32_t lut_load_cbtile_offset = 0;
                        for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                        {
                            uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];

                            // load lut table
                            uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_cbtile_offset + tmp_index;
                            mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);

                            // conduct computation
                            for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                            {
                                output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[lut_buffer_offset + tmp_f];
                            }

                            lut_load_cbtile_offset += LUT_LOAD_CBTILE_SIZE;
                        }

                        tmp_input_row_offset += CB_MTILE_SIZE;
                        tmp_output_row_offset += FEATURE_MTILE_SIZE;
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // update lut load mmtile offset
                lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
                // update input's stile offset
                input_stile_offset += INPUT_STILE_SIZE;
            }

            // save output buffer to DRAM
            uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_output_offset = 0;
            uint32_t tmp_output_byte_offset = 0;
            for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
            {
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                tmp_output_offset += MAX_OUTPUT_PER_RW;
            }
            mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
            mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            // update input & output intra each stile's mtile offset
            input_mtile_offset += INPUT_MTILE_SIZE;
            output_mtile_offset += OUTPUT_MTILE_SIZE;
        }

        // update lut table mtile's offset
        lut_mtile_offset += LUT_MTILE_SIZE;
        // update output's stile offset
        output_stile_offset += OUTPUT_STILE_SIZE;
    }
#else
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    uint32_t output_stile_offset = 0;
    uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
    for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
    {
        // tile offset, changing along with loop iteration
        uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
        uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE 
        for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
        {
            // reset output buffer
            // when oc is the innermost outer iterator, we can just reset output buffer instead of loading output data
            for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                output_result_buffer[tmp_output_offset] = 0;
            
            uint32_t input_stile_offset = 0;
            uint32_t lut_load_mmtile_offset = 0;
            for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    uint32_t lut_load_tile_offset = 0;
                    for(uint32_t tmp_init_load_cb=0; tmp_init_load_cb<CB_MTILE_SIZE; tmp_init_load_cb+=CB_LOAD_TILE_SIZE)
                    {
                        // load lut table if it is stored in MRAM
                        uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_tile_offset + lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
                        uint32_t tmp_lut_offset = 0;
                        uint32_t tmp_lut_byte_offset = 0;
                        for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
                        {
                            mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
                            tmp_lut_offset += MAX_LUT_PER_RW;
                        }
                        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                        mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif

                        // add a memory barrier to ensure all tasklets finishes LUT read
			            barrier_wait(&lut_load_barrier);
                        uint32_t tmp_input_row_offset = 0;
                        uint32_t tmp_output_row_offset = 0;
                        // conduct computation
                        for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                        {
                            for(uint32_t tmp_cb=0; tmp_cb<CB_LOAD_TILE_SIZE; ++tmp_cb)
                            {
                                uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_init_load_cb + tmp_cb];
                                for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                                {
                                    output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[tmp_index + tmp_f];
                                }
                            }

                            tmp_input_row_offset += CB_MTILE_SIZE;
                            tmp_output_row_offset += FEATURE_MTILE_SIZE;
                        }
                        // add a memory barrier to ensure all computation have been finished before reading new lut data
                        barrier_wait(&lut_load_barrier1);

                        // update lut load tile offset
                        lut_load_tile_offset += LUT_LOAD_TILE_SIZE;                        
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // update lut load mmtile offset
                lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
                // update input's stile offset
                input_stile_offset += INPUT_STILE_SIZE;
            }

            // save output buffer to DRAM
            uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_output_offset = 0;
            uint32_t tmp_output_byte_offset = 0;
            for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
            {
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                tmp_output_offset += MAX_OUTPUT_PER_RW;
            }
            mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
            mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            // update input & output intra each stile's mtile offset
            input_mtile_offset += INPUT_MTILE_SIZE;
            output_mtile_offset += OUTPUT_MTILE_SIZE;
        }

        // update lut table mtile's offset
        lut_mtile_offset += LUT_MTILE_SIZE;
        // update output's stile offset
        output_stile_offset += OUTPUT_STILE_SIZE;
    }
#endif

}

#elif defined(LOOP_ORDER_FCN)

void lut_kernel()
{
    // tmp taklet id
    uint32_t tasklet_id = me();

#ifdef STATIC_LUT_TABLE
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // load lut table first
    uint32_t lut_tensor_offset = lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
    uint32_t tmp_lut_offset = 0;
    uint32_t tmp_lut_byte_offset = 0;
    for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
    {
        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
        tmp_lut_offset += MAX_LUT_PER_RW;
    }
    mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
    mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif
    barrier_wait(&lut_load_barrier);

    uint32_t output_stile_offset = 0;
    uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
    for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
    {
        uint32_t input_stile_offset = 0;
        uint32_t lut_cbmtile_offset = 0;
        for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
        {
            // tile offset, changing along with loop iteration
            uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
            uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
            for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t tmp_input_row_offset = 0;
                uint32_t tmp_output_row_offset = 0;
                for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                {
                    uint32_t lut_cbtile_offset = 0;
                    for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                    {
                        uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];
                        for(uint32_t tmp_f=0; tmp_f<FEATURE_MTILE_SIZE; ++tmp_f)
                        {
                            output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_f] += lut_table_buffer[lut_cbmtile_offset + lut_cbtile_offset + tmp_index + tmp_init_feature + tmp_f];
                        }

                        lut_cbtile_offset += LUT_CBTILE_SIZE;
                    }

                    tmp_input_row_offset += CB_MTILE_SIZE;
                    tmp_output_row_offset += FEATURE_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update input & output intra each stile's mtile offset
                input_mtile_offset += INPUT_MTILE_SIZE;
                output_mtile_offset += OUTPUT_MTILE_SIZE;
            }

            // update lut load mmtile offset
            lut_cbmtile_offset += LUT_CBMTILE_SIZE;
            // update input's stile offset
            input_stile_offset += INPUT_STILE_SIZE;
        }

        // update lut table mtile's offset
        lut_mtile_offset += LUT_MTILE_SIZE;
        // update output's stile offset
        output_stile_offset += OUTPUT_STILE_SIZE;
    }
#elif defined(FINE_GRAIN)
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    uint32_t output_stile_offset = 0;
    uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
    for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
    {
        uint32_t input_stile_offset = 0;
        uint32_t lut_load_mmtile_offset = 0;
        for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
        {
            // tile offset, changing along with loop iteration
            uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
            uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
            for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    // all tasklets compute the same row
                    uint32_t tmp_input_row_offset = 0;
                    uint32_t tmp_output_row_offset = 0;
                    for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                    {
                        uint32_t lut_load_cbtile_offset = 0;
                        for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                        {
                            uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];

                            // load lut table
                            uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_cbtile_offset + tmp_index;
                            mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);

                            // conduct computation
                            for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                            {
                                output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[lut_buffer_offset + tmp_f];
                            }

                            lut_load_cbtile_offset += LUT_LOAD_CBTILE_SIZE;
                        }

                        tmp_input_row_offset += CB_MTILE_SIZE;
                        tmp_output_row_offset += FEATURE_MTILE_SIZE;
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update input & output intra each stile's mtile offset
                input_mtile_offset += INPUT_MTILE_SIZE;
                output_mtile_offset += OUTPUT_MTILE_SIZE;
            }

            // update lut load mmtile offset
            lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
            // update input's stile offset
            input_stile_offset += INPUT_STILE_SIZE;
        }

        // update lut table mtile's offset
        lut_mtile_offset += LUT_MTILE_SIZE;
        // update output's stile offset
        output_stile_offset += OUTPUT_STILE_SIZE;
    }
#else
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    uint32_t output_stile_offset = 0;
    uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
    for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
    {
        uint32_t input_stile_offset = 0;
        uint32_t lut_load_mmtile_offset = 0;
        for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
        {
            // tile offset, changing along with loop iteration
            uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
            uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
            for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    uint32_t lut_load_tile_offset = 0;
                    for(uint32_t tmp_init_load_cb=0; tmp_init_load_cb<CB_MTILE_SIZE; tmp_init_load_cb+=CB_LOAD_TILE_SIZE)
                    {
                        // load lut table if it is stored in MRAM
                        uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_tile_offset + lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
                        uint32_t tmp_lut_offset = 0;
                        uint32_t tmp_lut_byte_offset = 0;
                        for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
                        {
                            mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
                            tmp_lut_offset += MAX_LUT_PER_RW;
                        }
                        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                        mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif

                        // add a memory barrier to ensure all tasklets finishes LUT read
			            barrier_wait(&lut_load_barrier);
                        uint32_t tmp_input_row_offset = 0;
                        uint32_t tmp_output_row_offset = 0;
                        // conduct computation
                        for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                        {
                            for(uint32_t tmp_cb=0; tmp_cb<CB_LOAD_TILE_SIZE; ++tmp_cb)
                            {
                                uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_init_load_cb + tmp_cb];
                                for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                                {
                                    output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[tmp_index + tmp_f];
                                }
                            }

                            tmp_input_row_offset += CB_MTILE_SIZE;
                            tmp_output_row_offset += FEATURE_MTILE_SIZE;
                        }
                        // add a memory barrier to ensure all computation have been finished before reading new lut data
                        barrier_wait(&lut_load_barrier1);

                        // update lut load tile offset
                        lut_load_tile_offset += LUT_LOAD_TILE_SIZE;                        
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update input & output intra each stile's mtile offset
                input_mtile_offset += INPUT_MTILE_SIZE;
                output_mtile_offset += OUTPUT_MTILE_SIZE;
            }

            // update lut load mmtile offset
            lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
            // update input's stile offset
            input_stile_offset += INPUT_STILE_SIZE;
        }

        // update lut table mtile's offset
        lut_mtile_offset += LUT_MTILE_SIZE;
        // update output's stile offset
        output_stile_offset += OUTPUT_STILE_SIZE;
    }
#endif

}

#elif defined(LOOP_ORDER_CNF)

void lut_kernel()
{
    // tmp taklet id
    uint32_t tasklet_id = me();

#ifdef STATIC_LUT_TABLE
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // load lut table first
    uint32_t lut_tensor_offset = lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
    uint32_t tmp_lut_offset = 0;
    uint32_t tmp_lut_byte_offset = 0;
    for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
    {
        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
        tmp_lut_offset += MAX_LUT_PER_RW;
    }
    mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
    mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif
    barrier_wait(&lut_load_barrier);

    uint32_t input_stile_offset = 0;
    uint32_t lut_cbmtile_offset = 0;
    for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
    {
        // tile offset, changing along with loop iteration
        uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
        uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE 
        for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
        {
            uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
            // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_input_offset = 0;
            uint32_t tmp_input_byte_offset = 0;
            for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
            {
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                tmp_input_offset += MAX_INPUT_PER_RW;
            }    
            mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
            mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            uint32_t output_stile_offset = 0;
            for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
            {
                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // read lut and compute
                uint32_t tmp_input_row_offset = 0;
                uint32_t tmp_output_row_offset = 0;
                for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                {
                    uint32_t lut_cbtile_offset = 0;
                    for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                    {
                        uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];
                        for(uint32_t tmp_f=0; tmp_f<FEATURE_MTILE_SIZE; ++tmp_f)
                        {
                            output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_f] += lut_table_buffer[lut_cbmtile_offset + lut_cbtile_offset + tmp_index + tmp_init_feature + tmp_f];
                        }

                        lut_cbtile_offset += LUT_CBTILE_SIZE;
                    }

                    tmp_input_row_offset += CB_MTILE_SIZE;
                    tmp_output_row_offset += FEATURE_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update output's stile offset
                output_stile_offset += OUTPUT_STILE_SIZE;
            }

            // update input & output intra each stile's mtile offset
            input_mtile_offset += INPUT_MTILE_SIZE;
            output_mtile_offset += OUTPUT_MTILE_SIZE;
        }

        // update input's stile offset
        input_stile_offset += INPUT_STILE_SIZE;
        lut_cbmtile_offset += LUT_CBMTILE_SIZE;
    }
#elif defined(FINE_GRAIN)
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    uint32_t input_stile_offset = 0;
    uint32_t lut_load_mmtile_offset = 0;
    for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
    {
        // tile offset, changing along with loop iteration
        uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
        uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
        for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
        {
            uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
            // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_input_offset = 0;
            uint32_t tmp_input_byte_offset = 0;
            for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
            {
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                tmp_input_offset += MAX_INPUT_PER_RW;
            }    
            mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
            mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            uint32_t output_stile_offset = 0;
            uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
            for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
            {
                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    // all tasklets compute the same row
                    uint32_t tmp_input_row_offset = 0;
                    uint32_t tmp_output_row_offset = 0;
                    for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                    {
                        uint32_t lut_load_cbtile_offset = 0;
                        for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                        {
                            uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];

                            // load lut table
                            uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_cbtile_offset + tmp_index;
                            mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);

                            // conduct computation
                            for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                            {
                                output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[lut_buffer_offset + tmp_f];
                            }

                            lut_load_cbtile_offset += LUT_LOAD_CBTILE_SIZE;
                        }

                        tmp_input_row_offset += CB_MTILE_SIZE;
                        tmp_output_row_offset += FEATURE_MTILE_SIZE;
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update lut table mtile's offset
                lut_mtile_offset += LUT_MTILE_SIZE;
                // update output's stile offset
                output_stile_offset += OUTPUT_STILE_SIZE;
            }

            // update input & output intra each stile's mtile offset
            input_mtile_offset += INPUT_MTILE_SIZE;
            output_mtile_offset += OUTPUT_MTILE_SIZE;
        }

        // update lut load mmtile offset
        lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
        // update input's stile offset
        input_stile_offset += INPUT_STILE_SIZE;
    }
#else
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    uint32_t input_stile_offset = 0;
    uint32_t lut_load_mmtile_offset = 0;
    for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
    {
        // tile offset, changing along with loop iteration
        uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
        uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
        for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
        {
            uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
            // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
            uint32_t tmp_input_offset = 0;
            uint32_t tmp_input_byte_offset = 0;
            for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
            {
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                tmp_input_offset += MAX_INPUT_PER_RW;
            }    
            mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
            mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

            uint32_t output_stile_offset = 0;
            uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
            for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
            {
                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    uint32_t lut_load_tile_offset = 0;
                    for(uint32_t tmp_init_load_cb=0; tmp_init_load_cb<CB_MTILE_SIZE; tmp_init_load_cb+=CB_LOAD_TILE_SIZE)
                    {
                        // load lut table if it is stored in MRAM
                        uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_tile_offset + lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
                        uint32_t tmp_lut_offset = 0;
                        uint32_t tmp_lut_byte_offset = 0;
                        for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
                        {
                            mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
                            tmp_lut_offset += MAX_LUT_PER_RW;
                        }
                        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                        mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif

                        // add a memory barrier to ensure all tasklets finishes LUT read
			            barrier_wait(&lut_load_barrier);
                        uint32_t tmp_input_row_offset = 0;
                        uint32_t tmp_output_row_offset = 0;
                        // conduct computation
                        for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                        {
                            for(uint32_t tmp_cb=0; tmp_cb<CB_LOAD_TILE_SIZE; ++tmp_cb)
                            {
                                uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_init_load_cb + tmp_cb];
                                for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                                {
                                    output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[tmp_index + tmp_f];
                                }
                            }

                            tmp_input_row_offset += CB_MTILE_SIZE;
                            tmp_output_row_offset += FEATURE_MTILE_SIZE;
                        }
                        // add a memory barrier to ensure all computation have been finished before reading new lut data
                        barrier_wait(&lut_load_barrier1);

                        // update lut load tile offset
                        lut_load_tile_offset += LUT_LOAD_TILE_SIZE;                        
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update lut table mtile's offset
                lut_mtile_offset += LUT_MTILE_SIZE;
                // update output's stile offset
                output_stile_offset += OUTPUT_STILE_SIZE;
            }

            // update input & output intra each stile's mtile offset
            input_mtile_offset += INPUT_MTILE_SIZE;
            output_mtile_offset += OUTPUT_MTILE_SIZE;
        }

        // update lut load mmtile offset
        lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
        // update input's stile offset
        input_stile_offset += INPUT_STILE_SIZE;
    }
#endif

}

#elif defined(LOOP_ORDER_CFN)

void lut_kernel()
{
    // tmp taklet id
    uint32_t tasklet_id = me();

#ifdef STATIC_LUT_TABLE
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    // load lut table first
    uint32_t lut_tensor_offset = lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
    uint32_t tmp_lut_offset = 0;
    uint32_t tmp_lut_byte_offset = 0;
    for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
    {
        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
        tmp_lut_offset += MAX_LUT_PER_RW;
    }
    mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
    mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif
    barrier_wait(&lut_load_barrier);

    uint32_t input_stile_offset = 0;
    uint32_t lut_cbmtile_offset = 0;
    for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
    {
        uint32_t output_stile_offset = 0;
        uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
        for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
        {
            // tile offset, changing along with loop iteration
            uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
            uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
            for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t tmp_input_row_offset = 0;
                uint32_t tmp_output_row_offset = 0;
                for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                {
                    uint32_t lut_cbtile_offset = 0;
                    for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                    {
                        uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];
                        for(uint32_t tmp_f=0; tmp_f<FEATURE_MTILE_SIZE; ++tmp_f)
                        {
                            output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_f] += lut_table_buffer[lut_cbmtile_offset + lut_cbtile_offset + tmp_index + tmp_init_feature + tmp_f];
                        }

                        lut_cbtile_offset += LUT_CBTILE_SIZE;
                    }

                    tmp_input_row_offset += CB_MTILE_SIZE;
                    tmp_output_row_offset += FEATURE_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update input & output intra each stile's mtile offset
                input_mtile_offset += INPUT_MTILE_SIZE;
                output_mtile_offset += OUTPUT_MTILE_SIZE;
            }

            // update lut table mtile's offset
            lut_mtile_offset += LUT_MTILE_SIZE;
            // update output's stile offset
            output_stile_offset += OUTPUT_STILE_SIZE;
        }

        // update lut load mmtile offset
        lut_cbmtile_offset += LUT_CBMTILE_SIZE;
        // update input's stile offset
        input_stile_offset += INPUT_STILE_SIZE;
    }
#elif defined(FINE_GRAIN)
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    uint32_t input_stile_offset = 0;
    uint32_t lut_load_mmtile_offset = 0;
    for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
    {
        uint32_t output_stile_offset = 0;
        uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
        for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
        {
            // tile offset, changing along with loop iteration
            uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
            uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
            for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    // all tasklets compute the same row
                    uint32_t tmp_input_row_offset = 0;
                    uint32_t tmp_output_row_offset = 0;
                    for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                    {
                        uint32_t lut_load_cbtile_offset = 0;
                        for(uint32_t tmp_cb=0; tmp_cb<CB_MTILE_SIZE; ++tmp_cb)
                        {
                            uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_cb];

                            // load lut table
                            uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_cbtile_offset + tmp_index;
                            mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);

                            // conduct computation
                            for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                            {
                                output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[lut_buffer_offset + tmp_f];
                            }

                            lut_load_cbtile_offset += LUT_LOAD_CBTILE_SIZE;
                        }

                        tmp_input_row_offset += CB_MTILE_SIZE;
                        tmp_output_row_offset += FEATURE_MTILE_SIZE;
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update input & output intra each stile's mtile offset
                input_mtile_offset += INPUT_MTILE_SIZE;
                output_mtile_offset += OUTPUT_MTILE_SIZE;
            }

            // update lut table mtile's offset
            lut_mtile_offset += LUT_MTILE_SIZE;
            // update output's stile offset
            output_stile_offset += OUTPUT_STILE_SIZE;
        }

        // update lut load mmtile offset
        lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
        // update input's stile offset
        input_stile_offset += INPUT_STILE_SIZE;
    }
#else
    // on-chip buffer offsets of each tasklet
    uint32_t output_buffer_offset = OUTPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t input_buffer_offset = INPUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;
    uint32_t lut_buffer_offset = LUT_BUFFER_SIZE_PER_TASKLET * tasklet_id;

    uint32_t input_stile_offset = 0;
    uint32_t lut_load_mmtile_offset = 0;
    for(uint32_t tmp_init_cb=0; tmp_init_cb<NUM_CODEBOOK; tmp_init_cb+=CB_MTILE_SIZE)
    {
        uint32_t output_stile_offset = 0;
        uint32_t lut_mtile_offset = 0; // each lut mtile's size is NUM_CODEBOOK * NUM_CENTROID * FEATURE_MTILE_SIZE
        for(uint32_t tmp_init_feature=0; tmp_init_feature<FEATURE_STILE_SIZE; tmp_init_feature+=FEATURE_MTILE_SIZE)
        {
            // tile offset, changing along with loop iteration
            uint32_t input_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * CB_MTILE_SIZE
            uint32_t output_mtile_offset = 0; // each mtile's size is N_MTILE_SIZE * FEATURE_MTILE_SIZE
            for(uint32_t tmp_init_row=0; tmp_init_row<N_STILE_SIZE; tmp_init_row+=N_MTILE_SIZE)
            {
                uint32_t input_tensor_offset = input_stile_offset + input_mtile_offset + input_buffer_offset;
                // read input indices
#if INPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_input_offset = 0;
                uint32_t tmp_input_byte_offset = 0;
                for(tmp_input_byte_offset=0; tmp_input_byte_offset<INPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_input_byte_offset+=2048)
                {
                    mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], 2048);
                    tmp_input_offset += MAX_INPUT_PER_RW;
                }    
                mram_read(&input_index[input_tensor_offset+tmp_input_offset], &input_index_buffer[input_buffer_offset+tmp_input_offset], INPUT_BUFFER_BYTE_PER_TASKLET-tmp_input_byte_offset);
#else
                mram_read(&input_index[input_tensor_offset], &input_index_buffer[input_buffer_offset], INPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // load output buffer
                uint32_t output_tensor_offset = output_stile_offset + output_mtile_offset + output_buffer_offset;
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                uint32_t tmp_output_offset = 0;
                uint32_t tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_read(&output_data[output_tensor_offset+tmp_output_offset], &output_result_buffer[output_buffer_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_read(&output_data[output_tensor_offset], &output_result_buffer[output_buffer_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif
                if(tmp_init_cb == 0)
                    for(uint32_t tmp_output_offset=output_buffer_offset; tmp_output_offset<output_buffer_offset+OUTPUT_BUFFER_SIZE_PER_TASKLET; ++tmp_output_offset)
                        output_result_buffer[tmp_output_offset] = 0;

                // load lut and compute
                uint32_t lut_load_mtile_offset = 0;
                for(uint32_t tmp_init_load_feature=0; tmp_init_load_feature<FEATURE_MTILE_SIZE; tmp_init_load_feature+=FEATURE_LOAD_TILE_SIZE)
                {
                    uint32_t lut_load_tile_offset = 0;
                    for(uint32_t tmp_init_load_cb=0; tmp_init_load_cb<CB_MTILE_SIZE; tmp_init_load_cb+=CB_LOAD_TILE_SIZE)
                    {
                        // load lut table if it is stored in MRAM
                        uint32_t lut_tensor_offset = lut_mtile_offset + lut_load_mtile_offset + lut_load_mmtile_offset + lut_load_tile_offset + lut_buffer_offset;
#if LUT_BUFFER_BYTE_PER_TASKLET > 2048
                        uint32_t tmp_lut_offset = 0;
                        uint32_t tmp_lut_byte_offset = 0;
                        for(tmp_lut_byte_offset=0; tmp_lut_byte_offset<LUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_lut_byte_offset+=2048)
                        {
                            mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], 2048);
                            tmp_lut_offset += MAX_LUT_PER_RW;
                        }
                        mram_read(&lut_table[lut_tensor_offset+tmp_lut_offset], &lut_table_buffer[lut_buffer_offset+tmp_lut_offset], LUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                        mram_read(&lut_table[lut_tensor_offset], &lut_table_buffer[lut_buffer_offset], LUT_BUFFER_BYTE_PER_TASKLET);
#endif

                        // add a memory barrier to ensure all tasklets finishes LUT read
			            barrier_wait(&lut_load_barrier);
                        uint32_t tmp_input_row_offset = 0;
                        uint32_t tmp_output_row_offset = 0;
                        // conduct computation
                        for(uint32_t tmp_row=0; tmp_row<N_PER_TASKLET; ++tmp_row)
                        {
                            for(uint32_t tmp_cb=0; tmp_cb<CB_LOAD_TILE_SIZE; ++tmp_cb)
                            {
                                uint32_t tmp_index = input_index_buffer[input_buffer_offset + tmp_input_row_offset + tmp_init_load_cb + tmp_cb];
                                for(uint32_t tmp_f=0; tmp_f<FEATURE_LOAD_TILE_SIZE; ++tmp_f)
                                {
                                    output_result_buffer[output_buffer_offset + tmp_output_row_offset + tmp_init_load_feature + tmp_f] += lut_table_buffer[tmp_index + tmp_f];
                                }
                            }

                            tmp_input_row_offset += CB_MTILE_SIZE;
                            tmp_output_row_offset += FEATURE_MTILE_SIZE;
                        }
                        // add a memory barrier to ensure all computation have been finished before reading new lut data
                        barrier_wait(&lut_load_barrier1);

                        // update lut load tile offset
                        lut_load_tile_offset += LUT_LOAD_TILE_SIZE;                        
                    }

                    // update lut load mtile offset
                    lut_load_mtile_offset += LUT_LOAD_MTILE_SIZE;
                }

                // save output buffer to DRAM
#if OUTPUT_BUFFER_BYTE_PER_TASKLET > 2048
                tmp_output_offset = 0;
                tmp_output_byte_offset = 0;
                for(tmp_output_byte_offset = 0; tmp_output_byte_offset<OUTPUT_BUFFER_BYTE_PER_TASKLET-2048; tmp_output_byte_offset+=2048)
                {
                    mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], 2048);
                    tmp_output_offset += MAX_OUTPUT_PER_RW;
                }
                mram_write(&output_result_buffer[output_buffer_offset+tmp_output_offset], &output_data[output_tensor_offset+tmp_output_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET-2048);
#else
                mram_write(&output_result_buffer[output_buffer_offset], &output_data[output_tensor_offset], OUTPUT_BUFFER_BYTE_PER_TASKLET);
#endif

                // update input & output intra each stile's mtile offset
                input_mtile_offset += INPUT_MTILE_SIZE;
                output_mtile_offset += OUTPUT_MTILE_SIZE;
            }

            // update lut table mtile's offset
            lut_mtile_offset += LUT_MTILE_SIZE;
            // update output's stile offset
            output_stile_offset += OUTPUT_STILE_SIZE;
        }

        // update lut load mmtile offset
        lut_load_mmtile_offset += LUT_LOAD_MMTILE_SIZE;
        // update input's stile offset
        input_stile_offset += INPUT_STILE_SIZE;
    }
#endif

}

#else

void lut_kernel()
{
    // to initialize mram tensors during compiling
    uint32_t tmp_lut = lut_table[0];
    uint32_t tmp_index = input_index[0];
    uint32_t tmp_output = output_data[0];
}

#endif
