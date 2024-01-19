#include "parser.h"
#include <iostream>
#include <cassert>
#include <stdint.h>
#include <string>


void parse_amm_configs(AMMParams& amm_params, std::string yaml_path)
{
    YAML::Node config;

    // read yaml file
    try{
        config = YAML::LoadFile(yaml_path);
    }catch(YAML::BadFile &e){
        std::cerr<<"Config File Not Open"<<std::endl;
    }

    // parse yaml configs
    try
    {
        amm_params.index_params.n = config["amm_shape_params"]["n"].as<uint32_t>();
        amm_params.index_params.input_feature_len = config["amm_shape_params"]["input_feature_len"].as<uint32_t>();
        amm_params.index_params.num_codebook = config["amm_shape_params"]["num_codebook"].as<uint32_t>();
        amm_params.index_params.num_centroid = config["amm_shape_params"]["num_centroid"].as<uint32_t>();
        amm_params.index_params.sub_vec_len = amm_params.index_params.input_feature_len / amm_params.index_params.num_codebook;
        amm_params.index_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        amm_params.index_params.n_stile_size = config["kernel_params"]["n_stile_size"].as<uint32_t>();
        amm_params.index_params.cb_mtile_size = config["kernel_params"]["cb_mtile_size"].as<uint32_t>();

        amm_params.lut_params.n = config["amm_shape_params"]["n"].as<uint32_t>();
        amm_params.lut_params.output_feature_len = config["amm_shape_params"]["output_feature_len"].as<uint32_t>();
        amm_params.lut_params.num_codebook = config["amm_shape_params"]["num_codebook"].as<uint32_t>();
        amm_params.lut_params.num_centroid = config["amm_shape_params"]["num_centroid"].as<uint32_t>();
        amm_params.lut_params.scale = config["amm_shape_params"]["scale"].as<float>();
        amm_params.lut_params.bias = config["amm_shape_params"]["bias"].as<float>();
        amm_params.lut_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        amm_params.lut_params.dpu_num = config["system_params"]["dpu_num"].as<uint32_t>();
        amm_params.lut_params.n_stile_size = config["kernel_params"]["n_stile_size"].as<uint32_t>();
        amm_params.lut_params.feature_stile_size = config["kernel_params"]["feature_stile_size"].as<uint32_t>();
        amm_params.lut_params.input_parallelism = amm_params.lut_params.n / amm_params.lut_params.n_stile_size;
        amm_params.lut_params.lut_parallelism = amm_params.lut_params.output_feature_len / amm_params.lut_params.feature_stile_size;
        amm_params.lut_params.n_mtile_size = config["kernel_params"]["n_mtile_size"].as<uint32_t>();
        amm_params.lut_params.feature_mtile_size = config["kernel_params"]["feature_mtile_size"].as<uint32_t>();
        amm_params.lut_params.cb_mtile_size = config["kernel_params"]["cb_mtile_size"].as<uint32_t>();
        amm_params.lut_params.feature_load_tile_size = config["kernel_params"]["feature_load_tile_size"].as<uint32_t>();
        amm_params.lut_params.cb_load_tile_size = config["kernel_params"]["cb_load_tile_size"].as<uint32_t>();
        amm_params.lut_params.lut_load_type = LUTLoadType(config["kernel_params"]["lut_load_type"].as<uint32_t>());

    }catch(YAML::TypedBadConversion<std::string> &e){
        std::cerr<<"Error in parsing configs"<<std::endl;
    }

    // check parameter correctness
    // index param
    assert(amm_params.index_params.input_feature_len % amm_params.index_params.sub_vec_len == 0);
    assert(amm_params.index_params.input_feature_len % amm_params.index_params.num_codebook == 0);
    // lut param
    assert(amm_params.lut_params.input_parallelism * amm_params.lut_params.lut_parallelism == amm_params.lut_params.dpu_num);
    assert(amm_params.lut_params.n == amm_params.lut_params.n_stile_size * amm_params.lut_params.input_parallelism);
    assert(amm_params.lut_params.output_feature_len == amm_params.lut_params.feature_stile_size * amm_params.lut_params.lut_parallelism);
    assert(amm_params.lut_params.n_stile_size % amm_params.lut_params.n_mtile_size == 0);
    assert(amm_params.lut_params.feature_stile_size % amm_params.lut_params.feature_mtile_size == 0);
    assert(amm_params.lut_params.num_codebook % amm_params.lut_params.cb_mtile_size == 0);
    assert(amm_params.lut_params.feature_mtile_size % amm_params.lut_params.feature_load_tile_size == 0);
    assert(amm_params.lut_params.cb_mtile_size % amm_params.lut_params.cb_load_tile_size == 0);
    // tasklet alignment
    uint32_t nr_tasklets = config["system_params"]["nr_tasklets"].as<uint32_t>();
    assert(amm_params.lut_params.n_mtile_size % nr_tasklets == 0);

    // print amm params
    std::cout << "------ system params ------\n";
    std::cout << "num threads " << amm_params.index_params.num_threads << ", dpu num " << amm_params.lut_params.dpu_num << ", num tasklets " << nr_tasklets << std::endl;
    std::cout << "------ amm shape params ------\n";
    std::cout << "num codebook " << amm_params.index_params.num_codebook << ", num centroid " << amm_params.index_params.num_centroid << std::endl;
    std::cout << "n " << amm_params.index_params.n << ", input feature len " << amm_params.index_params.input_feature_len << ", output feature len " << amm_params.lut_params.output_feature_len << std::endl;
    std::cout << "scale " << amm_params.lut_params.scale << ", bias " << amm_params.lut_params.bias << std::endl;
    std::cout << "------ kernel params ------\n";
    std::cout << "loop order: " << config["kernel_params"]["loop_order"].as<uint32_t>() << " (0: nfc, 1: ncf, 2: fnc, 3: fcn, 4: cnf, 5: cfn)\n";
    std::cout << "lut load type: " << config["kernel_params"]["lut_load_type"].as<uint32_t>() << " (0: static lut table, 1: fine grain, 2: coarse grain)\n";
    std::cout << "n stile size " << amm_params.lut_params.n_stile_size << ", feature stile size " << amm_params.lut_params.feature_stile_size << std::endl;
    std::cout << "n mtile size " << amm_params.lut_params.n_mtile_size << ", feature mtile size " << amm_params.lut_params.feature_mtile_size << ", cb mtile size " << amm_params.lut_params.cb_mtile_size << std::endl;
    std::cout << "feature load tile size " << amm_params.lut_params.feature_load_tile_size << ", cb load tile size " << amm_params.lut_params.cb_load_tile_size << std::endl;
    std::cout << "------ calculated params ------\n";
    std::cout << "sub vec len " << amm_params.index_params.sub_vec_len << ", input parallelism " << amm_params.lut_params.input_parallelism << ", lut parallelism " << amm_params.lut_params.lut_parallelism << std::endl;
}


void parse_transformer_configs(TransformerParams& transformer_params, std::string yaml_path)
{
    YAML::Node config;

    // read yaml file
    try{
        config = YAML::LoadFile(yaml_path);
    }catch(YAML::BadFile &e){
        std::cerr<<"Config File Not Open"<<std::endl;
    }

    // parse yaml configs
    try
    {
        // thread num
        transformer_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();

        // layer num
        transformer_params.layer_num = config["network_params"]["layer_num"].as<uint32_t>();

        // pim binaries
        transformer_params.pim_binary_list[0] = config["system_params"]["qkv_lut_pim_binary"].as<std::string>();
        transformer_params.pim_binary_list[1] = config["system_params"]["o_lut_pim_binary"].as<std::string>();
        transformer_params.pim_binary_list[2] = config["system_params"]["ffn1_lut_pim_binary"].as<std::string>();
        transformer_params.pim_binary_list[3] = config["system_params"]["ffn2_lut_pim_binary"].as<std::string>();

        // qkv lut
        // lut param
        transformer_params.amm_param_list[0].lut_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.output_feature_len = config["network_params"]["token_dim"].as<uint32_t>() * 3; // QKV
        transformer_params.amm_param_list[0].lut_params.num_codebook = config["kernel_params"]["qkv_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.num_centroid = config["kernel_params"]["qkv_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.scale = config["kernel_params"]["qkv_scale"].as<float>();
        transformer_params.amm_param_list[0].lut_params.bias = config["kernel_params"]["qkv_bias"].as<float>();
        transformer_params.amm_param_list[0].lut_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.dpu_num = config["system_params"]["dpu_num"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.input_parallelism = config["kernel_params"]["qkv_input_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.lut_parallelism = config["kernel_params"]["qkv_lut_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.feature_stile_size = transformer_params.amm_param_list[0].lut_params.output_feature_len / transformer_params.amm_param_list[0].lut_params.lut_parallelism;
        transformer_params.amm_param_list[0].lut_params.n_stile_size = transformer_params.amm_param_list[0].lut_params.n / transformer_params.amm_param_list[0].lut_params.input_parallelism;
        transformer_params.amm_param_list[0].lut_params.n_mtile_size = config["kernel_params"]["qkv_n_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.feature_mtile_size = config["kernel_params"]["qkv_feature_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.cb_mtile_size = config["kernel_params"]["qkv_cb_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.lut_load_type = LUTLoadType(config["kernel_params"]["qkv_lut_load_type"].as<uint32_t>());
        transformer_params.amm_param_list[0].lut_params.feature_load_tile_size = config["kernel_params"]["qkv_feature_load_tile_size"].as<uint32_t>();
        transformer_params.amm_param_list[0].lut_params.cb_load_tile_size = config["kernel_params"]["qkv_cb_load_tile_size"].as<uint32_t>();
        // index calculation param
        transformer_params.amm_param_list[0].index_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[0].index_params.input_feature_len = config["network_params"]["token_dim"].as<uint32_t>();
        transformer_params.amm_param_list[0].index_params.num_codebook = config["kernel_params"]["qkv_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[0].index_params.num_centroid = config["kernel_params"]["qkv_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[0].index_params.sub_vec_len = transformer_params.amm_param_list[0].index_params.input_feature_len /  transformer_params.amm_param_list[0].index_params.num_codebook;
        transformer_params.amm_param_list[0].index_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[0].index_params.n_stile_size = transformer_params.amm_param_list[0].lut_params.n_stile_size;
        transformer_params.amm_param_list[0].index_params.cb_mtile_size = transformer_params.amm_param_list[0].lut_params.cb_mtile_size;

        // o lut
        // lut param
        transformer_params.amm_param_list[1].lut_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.output_feature_len = config["network_params"]["token_dim"].as<uint32_t>(); // O
        transformer_params.amm_param_list[1].lut_params.num_codebook = config["kernel_params"]["o_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.num_centroid = config["kernel_params"]["o_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.scale = config["kernel_params"]["o_scale"].as<float>();
        transformer_params.amm_param_list[1].lut_params.bias = config["kernel_params"]["o_bias"].as<float>();
        transformer_params.amm_param_list[1].lut_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.dpu_num = config["system_params"]["dpu_num"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.input_parallelism = config["kernel_params"]["o_input_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.lut_parallelism = config["kernel_params"]["o_lut_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.feature_stile_size = transformer_params.amm_param_list[1].lut_params.output_feature_len / transformer_params.amm_param_list[1].lut_params.lut_parallelism;
        transformer_params.amm_param_list[1].lut_params.n_stile_size = transformer_params.amm_param_list[1].lut_params.n / transformer_params.amm_param_list[1].lut_params.input_parallelism;
        transformer_params.amm_param_list[1].lut_params.n_mtile_size = config["kernel_params"]["o_n_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.feature_mtile_size = config["kernel_params"]["o_feature_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.cb_mtile_size = config["kernel_params"]["o_cb_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.lut_load_type = LUTLoadType(config["kernel_params"]["o_lut_load_type"].as<uint32_t>());
        transformer_params.amm_param_list[1].lut_params.feature_load_tile_size = config["kernel_params"]["o_feature_load_tile_size"].as<uint32_t>();
        transformer_params.amm_param_list[1].lut_params.cb_load_tile_size = config["kernel_params"]["o_cb_load_tile_size"].as<uint32_t>();
        // index calculation param
        transformer_params.amm_param_list[1].index_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[1].index_params.input_feature_len = config["network_params"]["token_dim"].as<uint32_t>();
        transformer_params.amm_param_list[1].index_params.num_codebook = config["kernel_params"]["o_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[1].index_params.num_centroid = config["kernel_params"]["o_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[1].index_params.sub_vec_len = transformer_params.amm_param_list[1].index_params.input_feature_len /  transformer_params.amm_param_list[1].index_params.num_codebook;
        transformer_params.amm_param_list[1].index_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[1].index_params.n_stile_size = transformer_params.amm_param_list[1].lut_params.n_stile_size;
        transformer_params.amm_param_list[1].index_params.cb_mtile_size = transformer_params.amm_param_list[1].lut_params.cb_mtile_size;

        // ffn1 lut
        // lut param
        transformer_params.amm_param_list[2].lut_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.output_feature_len = config["network_params"]["ffn_hidden_dim"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.num_codebook = config["kernel_params"]["ffn1_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.num_centroid = config["kernel_params"]["ffn1_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.scale = config["kernel_params"]["ffn1_scale"].as<float>();
        transformer_params.amm_param_list[2].lut_params.bias = config["kernel_params"]["ffn1_bias"].as<float>();
        transformer_params.amm_param_list[2].lut_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.dpu_num = config["system_params"]["dpu_num"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.input_parallelism = config["kernel_params"]["ffn1_input_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.lut_parallelism = config["kernel_params"]["ffn1_lut_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.feature_stile_size = transformer_params.amm_param_list[2].lut_params.output_feature_len / transformer_params.amm_param_list[2].lut_params.lut_parallelism;
        transformer_params.amm_param_list[2].lut_params.n_stile_size = transformer_params.amm_param_list[2].lut_params.n / transformer_params.amm_param_list[2].lut_params.input_parallelism;
        transformer_params.amm_param_list[2].lut_params.n_mtile_size = config["kernel_params"]["ffn1_n_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.feature_mtile_size = config["kernel_params"]["ffn1_feature_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.cb_mtile_size = config["kernel_params"]["ffn1_cb_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.lut_load_type = LUTLoadType(config["kernel_params"]["ffn1_lut_load_type"].as<uint32_t>());
        transformer_params.amm_param_list[2].lut_params.feature_load_tile_size = config["kernel_params"]["ffn1_feature_load_tile_size"].as<uint32_t>();
        transformer_params.amm_param_list[2].lut_params.cb_load_tile_size = config["kernel_params"]["ffn1_cb_load_tile_size"].as<uint32_t>();
        // index calculation param
        transformer_params.amm_param_list[2].index_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[2].index_params.input_feature_len = config["network_params"]["token_dim"].as<uint32_t>();
        transformer_params.amm_param_list[2].index_params.num_codebook = config["kernel_params"]["ffn1_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[2].index_params.num_centroid = config["kernel_params"]["ffn1_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[2].index_params.sub_vec_len = transformer_params.amm_param_list[2].index_params.input_feature_len /  transformer_params.amm_param_list[2].index_params.num_codebook;
        transformer_params.amm_param_list[2].index_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[2].index_params.n_stile_size = transformer_params.amm_param_list[2].lut_params.n_stile_size;
        transformer_params.amm_param_list[2].index_params.cb_mtile_size = transformer_params.amm_param_list[2].lut_params.cb_mtile_size;

        // ffn2 lut
        // lut param
        transformer_params.amm_param_list[3].lut_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.output_feature_len = config["network_params"]["token_dim"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.num_codebook = config["kernel_params"]["ffn2_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.num_centroid = config["kernel_params"]["ffn2_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.scale = config["kernel_params"]["ffn2_scale"].as<float>();
        transformer_params.amm_param_list[3].lut_params.bias = config["kernel_params"]["ffn2_bias"].as<float>();
        transformer_params.amm_param_list[3].lut_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.dpu_num = config["system_params"]["dpu_num"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.input_parallelism = config["kernel_params"]["ffn2_input_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.lut_parallelism = config["kernel_params"]["ffn2_lut_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.feature_stile_size = transformer_params.amm_param_list[3].lut_params.output_feature_len / transformer_params.amm_param_list[3].lut_params.lut_parallelism;
        transformer_params.amm_param_list[3].lut_params.n_stile_size = transformer_params.amm_param_list[3].lut_params.n / transformer_params.amm_param_list[3].lut_params.input_parallelism;
        transformer_params.amm_param_list[3].lut_params.n_mtile_size = config["kernel_params"]["ffn2_n_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.feature_mtile_size = config["kernel_params"]["ffn2_feature_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.cb_mtile_size = config["kernel_params"]["ffn2_cb_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.lut_load_type = LUTLoadType(config["kernel_params"]["ffn2_lut_load_type"].as<uint32_t>());
        transformer_params.amm_param_list[3].lut_params.feature_load_tile_size = config["kernel_params"]["ffn2_feature_load_tile_size"].as<uint32_t>();
        transformer_params.amm_param_list[3].lut_params.cb_load_tile_size = config["kernel_params"]["ffn2_cb_load_tile_size"].as<uint32_t>();
        // index calculation param
        transformer_params.amm_param_list[3].index_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[3].index_params.input_feature_len = config["network_params"]["ffn_hidden_dim"].as<uint32_t>();
        transformer_params.amm_param_list[3].index_params.num_codebook = config["kernel_params"]["ffn2_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[3].index_params.num_centroid = config["kernel_params"]["ffn2_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[3].index_params.sub_vec_len = transformer_params.amm_param_list[3].index_params.input_feature_len /  transformer_params.amm_param_list[3].index_params.num_codebook;
        transformer_params.amm_param_list[3].index_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[3].index_params.n_stile_size = transformer_params.amm_param_list[3].lut_params.n_stile_size;
        transformer_params.amm_param_list[3].index_params.cb_mtile_size = transformer_params.amm_param_list[3].lut_params.cb_mtile_size;

        // q lut
        // lut param
        transformer_params.amm_param_list[4].lut_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.output_feature_len = config["network_params"]["token_dim"].as<uint32_t>(); // O
        transformer_params.amm_param_list[4].lut_params.num_codebook = config["kernel_params"]["o_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.num_centroid = config["kernel_params"]["o_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.scale = config["kernel_params"]["o_scale"].as<float>();
        transformer_params.amm_param_list[4].lut_params.bias = config["kernel_params"]["o_bias"].as<float>();
        transformer_params.amm_param_list[4].lut_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.dpu_num = config["system_params"]["dpu_num"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.input_parallelism = config["kernel_params"]["o_input_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.lut_parallelism = config["kernel_params"]["o_lut_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.feature_stile_size = transformer_params.amm_param_list[1].lut_params.output_feature_len / transformer_params.amm_param_list[1].lut_params.lut_parallelism;
        transformer_params.amm_param_list[4].lut_params.n_stile_size = transformer_params.amm_param_list[1].lut_params.n / transformer_params.amm_param_list[1].lut_params.input_parallelism;
        transformer_params.amm_param_list[4].lut_params.n_mtile_size = config["kernel_params"]["o_n_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.feature_mtile_size = config["kernel_params"]["o_feature_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.cb_mtile_size = config["kernel_params"]["o_cb_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.lut_load_type = LUTLoadType(config["kernel_params"]["o_lut_load_type"].as<uint32_t>());
        transformer_params.amm_param_list[4].lut_params.feature_load_tile_size = config["kernel_params"]["o_feature_load_tile_size"].as<uint32_t>();
        transformer_params.amm_param_list[4].lut_params.cb_load_tile_size = config["kernel_params"]["o_cb_load_tile_size"].as<uint32_t>();
        // index calculation param
        transformer_params.amm_param_list[4].index_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[4].index_params.input_feature_len = config["network_params"]["token_dim"].as<uint32_t>();
        transformer_params.amm_param_list[4].index_params.num_codebook = config["kernel_params"]["o_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[4].index_params.num_centroid = config["kernel_params"]["o_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[4].index_params.sub_vec_len = transformer_params.amm_param_list[1].index_params.input_feature_len /  transformer_params.amm_param_list[1].index_params.num_codebook;
        transformer_params.amm_param_list[4].index_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[4].index_params.n_stile_size = transformer_params.amm_param_list[1].lut_params.n_stile_size;
        transformer_params.amm_param_list[4].index_params.cb_mtile_size = transformer_params.amm_param_list[1].lut_params.cb_mtile_size;
    
        // k lut
        // lut param
        transformer_params.amm_param_list[5].lut_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.output_feature_len = config["network_params"]["token_dim"].as<uint32_t>(); // O
        transformer_params.amm_param_list[5].lut_params.num_codebook = config["kernel_params"]["o_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.num_centroid = config["kernel_params"]["o_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.scale = config["kernel_params"]["o_scale"].as<float>();
        transformer_params.amm_param_list[5].lut_params.bias = config["kernel_params"]["o_bias"].as<float>();
        transformer_params.amm_param_list[5].lut_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.dpu_num = config["system_params"]["dpu_num"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.input_parallelism = config["kernel_params"]["o_input_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.lut_parallelism = config["kernel_params"]["o_lut_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.feature_stile_size = transformer_params.amm_param_list[1].lut_params.output_feature_len / transformer_params.amm_param_list[1].lut_params.lut_parallelism;
        transformer_params.amm_param_list[5].lut_params.n_stile_size = transformer_params.amm_param_list[1].lut_params.n / transformer_params.amm_param_list[1].lut_params.input_parallelism;
        transformer_params.amm_param_list[5].lut_params.n_mtile_size = config["kernel_params"]["o_n_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.feature_mtile_size = config["kernel_params"]["o_feature_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.cb_mtile_size = config["kernel_params"]["o_cb_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.lut_load_type = LUTLoadType(config["kernel_params"]["o_lut_load_type"].as<uint32_t>());
        transformer_params.amm_param_list[5].lut_params.feature_load_tile_size = config["kernel_params"]["o_feature_load_tile_size"].as<uint32_t>();
        transformer_params.amm_param_list[5].lut_params.cb_load_tile_size = config["kernel_params"]["o_cb_load_tile_size"].as<uint32_t>();
        // index calculation param
        transformer_params.amm_param_list[5].index_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[5].index_params.input_feature_len = config["network_params"]["token_dim"].as<uint32_t>();
        transformer_params.amm_param_list[5].index_params.num_codebook = config["kernel_params"]["o_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[5].index_params.num_centroid = config["kernel_params"]["o_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[5].index_params.sub_vec_len = transformer_params.amm_param_list[1].index_params.input_feature_len /  transformer_params.amm_param_list[1].index_params.num_codebook;
        transformer_params.amm_param_list[5].index_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[5].index_params.n_stile_size = transformer_params.amm_param_list[1].lut_params.n_stile_size;
        transformer_params.amm_param_list[5].index_params.cb_mtile_size = transformer_params.amm_param_list[1].lut_params.cb_mtile_size;

        // v lut
        // lut param
        transformer_params.amm_param_list[6].lut_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.output_feature_len = config["network_params"]["token_dim"].as<uint32_t>(); // O
        transformer_params.amm_param_list[6].lut_params.num_codebook = config["kernel_params"]["o_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.num_centroid = config["kernel_params"]["o_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.scale = config["kernel_params"]["o_scale"].as<float>();
        transformer_params.amm_param_list[6].lut_params.bias = config["kernel_params"]["o_bias"].as<float>();
        transformer_params.amm_param_list[6].lut_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.dpu_num = config["system_params"]["dpu_num"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.input_parallelism = config["kernel_params"]["o_input_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.lut_parallelism = config["kernel_params"]["o_lut_parallelism"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.feature_stile_size = transformer_params.amm_param_list[1].lut_params.output_feature_len / transformer_params.amm_param_list[1].lut_params.lut_parallelism;
        transformer_params.amm_param_list[6].lut_params.n_stile_size = transformer_params.amm_param_list[1].lut_params.n / transformer_params.amm_param_list[1].lut_params.input_parallelism;
        transformer_params.amm_param_list[6].lut_params.n_mtile_size = config["kernel_params"]["o_n_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.feature_mtile_size = config["kernel_params"]["o_feature_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.cb_mtile_size = config["kernel_params"]["o_cb_mtile_size"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.lut_load_type = LUTLoadType(config["kernel_params"]["o_lut_load_type"].as<uint32_t>());
        transformer_params.amm_param_list[6].lut_params.feature_load_tile_size = config["kernel_params"]["o_feature_load_tile_size"].as<uint32_t>();
        transformer_params.amm_param_list[6].lut_params.cb_load_tile_size = config["kernel_params"]["o_cb_load_tile_size"].as<uint32_t>();
        // index calculation param
        transformer_params.amm_param_list[6].index_params.n = config["network_params"]["seq_len"].as<uint32_t>() * config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.amm_param_list[6].index_params.input_feature_len = config["network_params"]["token_dim"].as<uint32_t>();
        transformer_params.amm_param_list[6].index_params.num_codebook = config["kernel_params"]["o_num_codebook"].as<uint32_t>();
        transformer_params.amm_param_list[6].index_params.num_centroid = config["kernel_params"]["o_num_centroid"].as<uint32_t>();
        transformer_params.amm_param_list[6].index_params.sub_vec_len = transformer_params.amm_param_list[1].index_params.input_feature_len /  transformer_params.amm_param_list[1].index_params.num_codebook;
        transformer_params.amm_param_list[6].index_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();
        transformer_params.amm_param_list[6].index_params.n_stile_size = transformer_params.amm_param_list[1].lut_params.n_stile_size;
        transformer_params.amm_param_list[6].index_params.cb_mtile_size = transformer_params.amm_param_list[1].lut_params.cb_mtile_size;

        // attention
        transformer_params.attention_params.seq_len = config["network_params"]["seq_len"].as<uint32_t>();
        transformer_params.attention_params.batch_size = config["network_params"]["batch_size"].as<uint32_t>();
        transformer_params.attention_params.n = transformer_params.attention_params.seq_len * transformer_params.attention_params.batch_size;
        transformer_params.attention_params.head_num = config["network_params"]["head_num"].as<uint32_t>();
        transformer_params.attention_params.head_dim = config["network_params"]["head_dim"].as<uint32_t>();
        transformer_params.attention_params.token_dim = config["network_params"]["token_dim"].as<uint32_t>();
        transformer_params.attention_params.dpu_num = config["system_params"]["dpu_num"].as<uint32_t>();
        transformer_params.attention_params.input_parallelism = config["kernel_params"]["qkv_input_parallelism"].as<uint32_t>();
        transformer_params.attention_params.lut_parallelism = config["kernel_params"]["qkv_lut_parallelism"].as<uint32_t>();
        transformer_params.attention_params.n_tile_size = transformer_params.amm_param_list[0].lut_params.n_stile_size;
        transformer_params.attention_params.token_tile_size = transformer_params.amm_param_list[0].lut_params.feature_stile_size / 3;
        transformer_params.attention_params.num_threads = config["system_params"]["num_threads"].as<uint32_t>();

    }catch(YAML::TypedBadConversion<std::string> &e){
        std::cerr<<"Error in parsing configs"<<std::endl;
    }

    // print params
    printf("----------Transformer Params----------\n");
    printf("thread num %d\n", transformer_params.num_threads);
    printf("layer num %d\n", transformer_params.layer_num);
    printf("PIM LUT binaries %s %s %s %s\n", transformer_params.pim_binary_list[0].c_str(), transformer_params.pim_binary_list[1].c_str(), transformer_params.pim_binary_list[2].c_str(), transformer_params.pim_binary_list[3].c_str());
    
    printf("----------QKV LUT Params----------\n");
    printf("index calc params:\n");
    printf("num_threads %d\n", transformer_params.amm_param_list[0].index_params.num_threads);
    printf("n %d, input feature len %d\n", transformer_params.amm_param_list[0].index_params.n, transformer_params.amm_param_list[0].index_params.input_feature_len);
    printf("num codebook %d, num centroid %d, sub vec len %d\n", transformer_params.amm_param_list[0].index_params.num_codebook, transformer_params.amm_param_list[0].index_params.num_centroid, transformer_params.amm_param_list[0].index_params.sub_vec_len);
    printf("n stile size %d, cb mtile size %d\n", transformer_params.amm_param_list[0].index_params.n_stile_size, transformer_params.amm_param_list[0].index_params.cb_mtile_size);
    printf("lut kernel params:\n");
    printf("n %d, output feature len %d, num codebook %d, num centroid %d\n", transformer_params.amm_param_list[0].lut_params.n, transformer_params.amm_param_list[0].lut_params.output_feature_len, transformer_params.amm_param_list[0].lut_params.num_codebook, transformer_params.amm_param_list[0].lut_params.num_centroid);
    printf("scale %.6f, bias %.6f\n", transformer_params.amm_param_list[0].lut_params.scale, transformer_params.amm_param_list[0].lut_params.bias);
    printf("num threads %d, dpu num %d\n", transformer_params.amm_param_list[0].lut_params.num_threads, transformer_params.amm_param_list[0].lut_params.dpu_num);
    printf("input parallelism %d, lut parallelism %d, feature stile size %d, n stile size %d\n", transformer_params.amm_param_list[0].lut_params.input_parallelism, transformer_params.amm_param_list[0].lut_params.lut_parallelism, transformer_params.amm_param_list[0].lut_params.feature_stile_size, transformer_params.amm_param_list[0].lut_params.n_stile_size);
    printf("n mtile size %d, feature mtile size %d, cb mtile size %d\n", transformer_params.amm_param_list[0].lut_params.n_mtile_size, transformer_params.amm_param_list[0].lut_params.feature_mtile_size, transformer_params.amm_param_list[0].lut_params.cb_mtile_size);
    printf("lut load type %d, feature load tile size %d, cb load tile size %d\n", transformer_params.amm_param_list[0].lut_params.lut_load_type, transformer_params.amm_param_list[0].lut_params.feature_load_tile_size, transformer_params.amm_param_list[0].lut_params.cb_load_tile_size);
    
    printf("----------O LUT Params----------\n");
    printf("index calc params:\n");
    printf("num_threads %d\n", transformer_params.amm_param_list[1].index_params.num_threads);
    printf("n %d, input feature len %d\n", transformer_params.amm_param_list[1].index_params.n, transformer_params.amm_param_list[1].index_params.input_feature_len);
    printf("num codebook %d, num centroid %d, sub vec len %d\n", transformer_params.amm_param_list[1].index_params.num_codebook, transformer_params.amm_param_list[1].index_params.num_centroid, transformer_params.amm_param_list[1].index_params.sub_vec_len);
    printf("n stile size %d, cb mtile size %d\n", transformer_params.amm_param_list[1].index_params.n_stile_size, transformer_params.amm_param_list[1].index_params.cb_mtile_size);
    printf("lut kernel params:\n");
    printf("n %d, output feature len %d, num codebook %d, num centroid %d\n", transformer_params.amm_param_list[1].lut_params.n, transformer_params.amm_param_list[1].lut_params.output_feature_len, transformer_params.amm_param_list[1].lut_params.num_codebook, transformer_params.amm_param_list[1].lut_params.num_centroid);
    printf("scale %.6f, bias %.6f\n", transformer_params.amm_param_list[1].lut_params.scale, transformer_params.amm_param_list[1].lut_params.bias);
    printf("num threads %d, dpu num %d\n", transformer_params.amm_param_list[1].lut_params.num_threads, transformer_params.amm_param_list[1].lut_params.dpu_num);
    printf("input parallelism %d, lut parallelism %d, feature stile size %d, n stile size %d\n", transformer_params.amm_param_list[1].lut_params.input_parallelism, transformer_params.amm_param_list[1].lut_params.lut_parallelism, transformer_params.amm_param_list[1].lut_params.feature_stile_size, transformer_params.amm_param_list[1].lut_params.n_stile_size);
    printf("n mtile size %d, feature mtile size %d, cb mtile size %d\n", transformer_params.amm_param_list[1].lut_params.n_mtile_size, transformer_params.amm_param_list[1].lut_params.feature_mtile_size, transformer_params.amm_param_list[1].lut_params.cb_mtile_size);
    printf("lut load type %d, feature load tile size %d, cb load tile size %d\n", transformer_params.amm_param_list[1].lut_params.lut_load_type, transformer_params.amm_param_list[1].lut_params.feature_load_tile_size, transformer_params.amm_param_list[1].lut_params.cb_load_tile_size);
    
    printf("----------FFN1 LUT Params----------\n");
    printf("index calc params:\n");
    printf("num_threads %d\n", transformer_params.amm_param_list[2].index_params.num_threads);
    printf("n %d, input feature len %d\n", transformer_params.amm_param_list[2].index_params.n, transformer_params.amm_param_list[2].index_params.input_feature_len);
    printf("num codebook %d, num centroid %d, sub vec len %d\n", transformer_params.amm_param_list[2].index_params.num_codebook, transformer_params.amm_param_list[2].index_params.num_centroid, transformer_params.amm_param_list[2].index_params.sub_vec_len);
    printf("n stile size %d, cb mtile size %d\n", transformer_params.amm_param_list[2].index_params.n_stile_size, transformer_params.amm_param_list[2].index_params.cb_mtile_size);
    printf("lut kernel params:\n");
    printf("n %d, output feature len %d, num codebook %d, num centroid %d\n", transformer_params.amm_param_list[2].lut_params.n, transformer_params.amm_param_list[2].lut_params.output_feature_len, transformer_params.amm_param_list[2].lut_params.num_codebook, transformer_params.amm_param_list[2].lut_params.num_centroid);
    printf("scale %.6f, bias %.6f\n", transformer_params.amm_param_list[2].lut_params.scale, transformer_params.amm_param_list[2].lut_params.bias);
    printf("num threads %d, dpu num %d\n", transformer_params.amm_param_list[2].lut_params.num_threads, transformer_params.amm_param_list[2].lut_params.dpu_num);
    printf("input parallelism %d, lut parallelism %d, feature stile size %d, n stile size %d\n", transformer_params.amm_param_list[2].lut_params.input_parallelism, transformer_params.amm_param_list[2].lut_params.lut_parallelism, transformer_params.amm_param_list[2].lut_params.feature_stile_size, transformer_params.amm_param_list[2].lut_params.n_stile_size);
    printf("n mtile size %d, feature mtile size %d, cb mtile size %d\n", transformer_params.amm_param_list[2].lut_params.n_mtile_size, transformer_params.amm_param_list[2].lut_params.feature_mtile_size, transformer_params.amm_param_list[2].lut_params.cb_mtile_size);
    printf("lut load type %d, feature load tile size %d, cb load tile size %d\n", transformer_params.amm_param_list[2].lut_params.lut_load_type, transformer_params.amm_param_list[2].lut_params.feature_load_tile_size, transformer_params.amm_param_list[2].lut_params.cb_load_tile_size);
    
    printf("----------FFN2 LUT Params----------\n");
    printf("index calc params:\n");
    printf("num_threads %d\n", transformer_params.amm_param_list[3].index_params.num_threads);
    printf("n %d, input feature len %d\n", transformer_params.amm_param_list[3].index_params.n, transformer_params.amm_param_list[3].index_params.input_feature_len);
    printf("num codebook %d, num centroid %d, sub vec len %d\n", transformer_params.amm_param_list[3].index_params.num_codebook, transformer_params.amm_param_list[3].index_params.num_centroid, transformer_params.amm_param_list[3].index_params.sub_vec_len);
    printf("n stile size %d, cb mtile size %d\n", transformer_params.amm_param_list[3].index_params.n_stile_size, transformer_params.amm_param_list[3].index_params.cb_mtile_size);
    printf("lut kernel params:\n");
    printf("n %d, output feature len %d, num codebook %d, num centroid %d\n", transformer_params.amm_param_list[3].lut_params.n, transformer_params.amm_param_list[3].lut_params.output_feature_len, transformer_params.amm_param_list[3].lut_params.num_codebook, transformer_params.amm_param_list[3].lut_params.num_centroid);
    printf("scale %.6f, bias %.6f\n", transformer_params.amm_param_list[3].lut_params.scale, transformer_params.amm_param_list[3].lut_params.bias);
    printf("num threads %d, dpu num %d\n", transformer_params.amm_param_list[3].lut_params.num_threads, transformer_params.amm_param_list[3].lut_params.dpu_num);
    printf("input parallelism %d, lut parallelism %d, feature stile size %d, n stile size %d\n", transformer_params.amm_param_list[3].lut_params.input_parallelism, transformer_params.amm_param_list[3].lut_params.lut_parallelism, transformer_params.amm_param_list[3].lut_params.feature_stile_size, transformer_params.amm_param_list[3].lut_params.n_stile_size);
    printf("n mtile size %d, feature mtile size %d, cb mtile size %d\n", transformer_params.amm_param_list[3].lut_params.n_mtile_size, transformer_params.amm_param_list[3].lut_params.feature_mtile_size, transformer_params.amm_param_list[3].lut_params.cb_mtile_size);
    printf("lut load type %d, feature load tile size %d, cb load tile size %d\n", transformer_params.amm_param_list[3].lut_params.lut_load_type, transformer_params.amm_param_list[3].lut_params.feature_load_tile_size, transformer_params.amm_param_list[3].lut_params.cb_load_tile_size);
    
    printf("----------Attention Params----------\n");
    printf("seq len %d, batch size %d, n %d\n", transformer_params.attention_params.seq_len, transformer_params.attention_params.batch_size, transformer_params.attention_params.n);
    printf("head num %d, head dim %d, token dim %d\n", transformer_params.attention_params.head_num, transformer_params.attention_params.head_dim, transformer_params.attention_params.token_dim);
    printf("dpu num %d, input parallelism %d, lut parallelism %d\n", transformer_params.attention_params.dpu_num, transformer_params.attention_params.input_parallelism, transformer_params.attention_params.lut_parallelism);
    printf("n tile size %d, token tile size %d\n", transformer_params.attention_params.n_tile_size, transformer_params.attention_params.token_tile_size);
    printf("num threads %d\n", transformer_params.attention_params.num_threads);
    
    printf("-----------------------------------\n");
}
