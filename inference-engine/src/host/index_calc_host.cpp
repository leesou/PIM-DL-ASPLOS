#include <memory.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "index_calc_host.h"


void index_calculation(float* input_tensor, float* centroid_tensor, index_data_type* lut_input_index, IndexCalcParams index_params)
{
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
                {
                    tmp_dist += (input_tensor[i*index_params.input_feature_len + index_params.sub_vec_len*tmp_codebook+tmp_dim] 
                                - centroid_tensor[tmp_codebook*index_params.num_centroid*index_params.sub_vec_len + tmp_centroid*index_params.sub_vec_len+tmp_dim]) *
                                (input_tensor[i*index_params.input_feature_len + index_params.sub_vec_len*tmp_codebook+tmp_dim] 
                                - centroid_tensor[tmp_codebook*index_params.num_centroid*index_params.sub_vec_len + tmp_centroid*index_params.sub_vec_len+tmp_dim]);
                }    
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

            uint32_t n_tile_id = i / index_params.n_stile_size;
            uint32_t intra_n_tile_id = i % index_params.n_stile_size;
            uint32_t cb_tile_id = tmp_codebook / index_params.cb_mtile_size;
            uint32_t intra_cb_tile_id = tmp_codebook % index_params.cb_mtile_size;
            uint32_t offset = n_tile_id * index_params.n_stile_size * index_params.num_codebook
                            + cb_tile_id * index_params.n_stile_size * index_params.cb_mtile_size
                            + intra_n_tile_id * index_params.cb_mtile_size
                            + intra_cb_tile_id;
            lut_input_index[offset] = tmp_index;

        }
    }
}
