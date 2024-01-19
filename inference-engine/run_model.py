import argparse
import os
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_config_file', type=str)
    parser.add_argument('--model_weight_file', type=str, default='null')
    parser.add_argument('--need_compile', action='store_true', default=True)

    args = parser.parse_args()
    return args

def compile(args):
    with open(args.transformer_config_file, 'r') as f:
        yaml_data = yaml.load(f.read(), Loader=yaml.FullLoader)

    # dpu settings
    tl_var = yaml_data['system_params']['nr_tasklets']

    # shape settings
    seq_len_var= yaml_data['network_params']['seq_len']
    batch_size_var = yaml_data['network_params']['batch_size']
    n_var = seq_len_var * batch_size_var
    token_dim_var = yaml_data['network_params']['token_dim']
    ffn_hidden_dim_var = yaml_data['network_params']['ffn_hidden_dim']

    ## QKV LUT
    # shape settings
    cb_var_qkv = yaml_data['kernel_params']['qkv_num_codebook']
    ct_var_qkv = yaml_data['kernel_params']['qkv_num_centroid']
    # kernel settings
    loop_order_var_qkv = yaml_data['kernel_params']['qkv_loop_order']
    lut_load_type_var_qkv = yaml_data['kernel_params']['qkv_lut_load_type']
    # tile sizes
    input_parallelism_var_qkv = yaml_data['kernel_params']['qkv_input_parallelism']
    lut_parallelism_var_qkv = yaml_data['kernel_params']['qkv_lut_parallelism']
    n_stile_var_qkv = int(n_var / input_parallelism_var_qkv)
    feature_stile_var_qkv = int(token_dim_var*3 / lut_parallelism_var_qkv)
    n_mtile_var_qkv = yaml_data['kernel_params']['qkv_n_mtile_size']
    feature_mtile_var_qkv = yaml_data['kernel_params']['qkv_feature_mtile_size']
    cb_mtile_var_qkv = yaml_data['kernel_params']['qkv_cb_mtile_size']
    feature_load_tile_var_qkv = yaml_data['kernel_params']['qkv_feature_load_tile_size']
    cb_load_tile_var_qkv = yaml_data['kernel_params']['qkv_cb_load_tile_size']

    ## O LUT
    # shape settings
    cb_var_o = yaml_data['kernel_params']['o_num_codebook']
    ct_var_o = yaml_data['kernel_params']['o_num_centroid']
    # kernel settings
    loop_order_var_o = yaml_data['kernel_params']['o_loop_order']
    lut_load_type_var_o = yaml_data['kernel_params']['o_lut_load_type']
    # tile sizes
    input_parallelism_var_o = yaml_data['kernel_params']['o_input_parallelism']
    lut_parallelism_var_o = yaml_data['kernel_params']['o_lut_parallelism']
    n_stile_var_o = int(n_var / input_parallelism_var_o)
    feature_stile_var_o = int(token_dim_var / lut_parallelism_var_o)
    n_mtile_var_o = yaml_data['kernel_params']['o_n_mtile_size']
    feature_mtile_var_o = yaml_data['kernel_params']['o_feature_mtile_size']
    cb_mtile_var_o = yaml_data['kernel_params']['o_cb_mtile_size']
    feature_load_tile_var_o = yaml_data['kernel_params']['o_feature_load_tile_size']
    cb_load_tile_var_o = yaml_data['kernel_params']['o_cb_load_tile_size']

    ## FFN1 LUT
    # shape settings
    cb_var_ffn1 = yaml_data['kernel_params']['ffn1_num_codebook']
    ct_var_ffn1 = yaml_data['kernel_params']['ffn1_num_centroid']
    # kernel settings
    loop_order_var_ffn1 = yaml_data['kernel_params']['ffn1_loop_order']
    lut_load_type_var_ffn1 = yaml_data['kernel_params']['ffn1_lut_load_type']
    # tile sizes
    input_parallelism_var_ffn1 = yaml_data['kernel_params']['ffn1_input_parallelism']
    lut_parallelism_var_ffn1 = yaml_data['kernel_params']['ffn1_lut_parallelism']
    n_stile_var_ffn1 = int(n_var / input_parallelism_var_ffn1)
    feature_stile_var_ffn1 = int(ffn_hidden_dim_var / lut_parallelism_var_ffn1)
    n_mtile_var_ffn1 = yaml_data['kernel_params']['ffn1_n_mtile_size']
    feature_mtile_var_ffn1 = yaml_data['kernel_params']['ffn1_feature_mtile_size']
    cb_mtile_var_ffn1 = yaml_data['kernel_params']['ffn1_cb_mtile_size']
    feature_load_tile_var_ffn1 = yaml_data['kernel_params']['ffn1_feature_load_tile_size']
    cb_load_tile_var_ffn1 = yaml_data['kernel_params']['ffn1_cb_load_tile_size']

    ## FFN2 LUT
    # shape settings
    cb_var_ffn2 = yaml_data['kernel_params']['ffn2_num_codebook']
    ct_var_ffn2 = yaml_data['kernel_params']['ffn2_num_centroid']
    # kernel settings
    loop_order_var_ffn2 = yaml_data['kernel_params']['ffn2_loop_order']
    lut_load_type_var_ffn2 = yaml_data['kernel_params']['ffn2_lut_load_type']
    # tile sizes
    input_parallelism_var_ffn2 = yaml_data['kernel_params']['ffn2_input_parallelism']
    lut_parallelism_var_ffn2 = yaml_data['kernel_params']['ffn2_lut_parallelism']
    n_stile_var_ffn2 = int(n_var / input_parallelism_var_ffn2)
    feature_stile_var_ffn2 = int(token_dim_var / lut_parallelism_var_ffn2)
    n_mtile_var_ffn2 = yaml_data['kernel_params']['ffn2_n_mtile_size']
    feature_mtile_var_ffn2 = yaml_data['kernel_params']['ffn2_feature_mtile_size']
    cb_mtile_var_ffn2 = yaml_data['kernel_params']['ffn2_cb_mtile_size']
    feature_load_tile_var_ffn2 = yaml_data['kernel_params']['ffn2_feature_load_tile_size']
    cb_load_tile_var_ffn2 = yaml_data['kernel_params']['ffn2_cb_load_tile_size']

    print(f"tasklet num {tl_var}")
    print(f"n {n_var}, seq len {seq_len_var}, batch size {batch_size_var}")
    print(f"token din {token_dim_var}, ffn hidden dim {ffn_hidden_dim_var}")
    
    print(f"----------QKV LUT Params----------")
    print(f"num codebook {cb_var_qkv}, num centroid {ct_var_qkv}, loop order {loop_order_var_qkv}, lut load type {lut_load_type_var_qkv}")
    print(f"input parallelism {input_parallelism_var_qkv}, lut parallelism {lut_parallelism_var_qkv}, n stile {n_stile_var_qkv}, feature stile {feature_stile_var_qkv}")
    print(f"n mtile {n_mtile_var_qkv}, feature mtile {feature_mtile_var_qkv}, cb mtile {cb_mtile_var_qkv}")
    print(f"feature load tile {feature_load_tile_var_qkv}, cb load tile {cb_load_tile_var_qkv}")

    print(f"----------O LUT Params----------")
    print(f"num codebook {cb_var_o}, num centroid {ct_var_o}, loop order {loop_order_var_o}, lut load type {lut_load_type_var_o}")
    print(f"input parallelism {input_parallelism_var_o}, lut parallelism {lut_parallelism_var_o}, n stile {n_stile_var_o}, feature stile {feature_stile_var_o}")
    print(f"n mtile {n_mtile_var_o}, feature mtile {feature_mtile_var_o}, cb mtile {cb_mtile_var_o}")
    print(f"feature load tile {feature_load_tile_var_o}, cb load tile {cb_load_tile_var_o}")

    print(f"----------FFN1 LUT Params----------")
    print(f"num codebook {cb_var_ffn1}, num centroid {ct_var_ffn1}, loop order {loop_order_var_ffn1}, lut load type {lut_load_type_var_ffn1}")
    print(f"input parallelism {input_parallelism_var_ffn1}, lut parallelism {lut_parallelism_var_ffn1}, n stile {n_stile_var_ffn1}, feature stile {feature_stile_var_ffn1}")
    print(f"n mtile {n_mtile_var_ffn1}, feature mtile {feature_mtile_var_ffn1}, cb mtile {cb_mtile_var_ffn1}")
    print(f"feature load tile {feature_load_tile_var_ffn1}, cb load tile {cb_load_tile_var_ffn1}")

    print(f"----------FFN2 LUT Params----------")
    print(f"num codebook {cb_var_ffn2}, num centroid {ct_var_ffn2}, loop order {loop_order_var_ffn2}, lut load type {lut_load_type_var_ffn2}")
    print(f"input parallelism {input_parallelism_var_ffn2}, lut parallelism {lut_parallelism_var_ffn2}, n stile {n_stile_var_ffn2}, feature stile {feature_stile_var_ffn2}")
    print(f"n mtile {n_mtile_var_ffn2}, feature mtile {feature_mtile_var_ffn2}, cb mtile {cb_mtile_var_ffn2}")
    print(f"feature load tile {feature_load_tile_var_ffn2}, cb load tile {cb_load_tile_var_ffn2}")

    os.system('mkdir -p build')
    command = f'cmake -DTRANSFORMER=1 -DDYNAMIC=0 -DSEPARATE=1 -DMEASURE_ENERGY=0 -DLATENCY_BREAKDOWN=0 -DLUT_BREAKDOWN=0 -DLAYER_BREAKDOWN=0 -DNR_TASKLETS={tl_var} \
                -DNUM_CODEBOOK_QKV={cb_var_qkv} -DNUM_CENTROID_QKV={ct_var_qkv} \
                -DLOOP_ORDER_QKV={loop_order_var_qkv} -DLUT_LOAD_TYPE_QKV={lut_load_type_var_qkv} \
                -DN_STILE_SIZE_QKV={n_stile_var_qkv} -DFEATURE_STILE_SIZE_QKV={feature_stile_var_qkv} \
                -DN_MTILE_SIZE_QKV={n_mtile_var_qkv} -DFEATURE_MTILE_SIZE_QKV={feature_mtile_var_qkv} -DCB_MTILE_SIZE_QKV={cb_mtile_var_qkv} \
                -DFEATURE_LOAD_TILE_SIZE_QKV={feature_load_tile_var_qkv} -DCB_LOAD_TILE_SIZE_QKV={cb_load_tile_var_qkv} \
                -DNUM_CODEBOOK_O={cb_var_o} -DNUM_CENTROID_O={ct_var_o} \
                -DLOOP_ORDER_O={loop_order_var_o} -DLUT_LOAD_TYPE_O={lut_load_type_var_o} \
                -DN_STILE_SIZE_O={n_stile_var_o} -DFEATURE_STILE_SIZE_O={feature_stile_var_o} \
                -DN_MTILE_SIZE_O={n_mtile_var_o} -DFEATURE_MTILE_SIZE_O={feature_mtile_var_o} -DCB_MTILE_SIZE_O={cb_mtile_var_o} \
                -DFEATURE_LOAD_TILE_SIZE_O={feature_load_tile_var_o} -DCB_LOAD_TILE_SIZE_O={cb_load_tile_var_o} \
                -DNUM_CODEBOOK_FFN1={cb_var_ffn1} -DNUM_CENTROID_FFN1={ct_var_ffn1} \
                -DLOOP_ORDER_FFN1={loop_order_var_ffn1} -DLUT_LOAD_TYPE_FFN1={lut_load_type_var_ffn1} \
                -DN_STILE_SIZE_FFN1={n_stile_var_ffn1} -DFEATURE_STILE_SIZE_FFN1={feature_stile_var_ffn1} \
                -DN_MTILE_SIZE_FFN1={n_mtile_var_ffn1} -DFEATURE_MTILE_SIZE_FFN1={feature_mtile_var_ffn1} -DCB_MTILE_SIZE_FFN1={cb_mtile_var_ffn1} \
                -DFEATURE_LOAD_TILE_SIZE_FFN1={feature_load_tile_var_ffn1} -DCB_LOAD_TILE_SIZE_FFN1={cb_load_tile_var_ffn1} \
                -DNUM_CODEBOOK_FFN2={cb_var_ffn2} -DNUM_CENTROID_FFN2={ct_var_ffn2} \
                -DLOOP_ORDER_FFN2={loop_order_var_ffn2} -DLUT_LOAD_TYPE_FFN2={lut_load_type_var_ffn2} \
                -DN_STILE_SIZE_FFN2={n_stile_var_ffn2} -DFEATURE_STILE_SIZE_FFN2={feature_stile_var_ffn2} \
                -DN_MTILE_SIZE_FFN2={n_mtile_var_ffn2} -DFEATURE_MTILE_SIZE_FFN2={feature_mtile_var_ffn2} -DCB_MTILE_SIZE_FFN2={cb_mtile_var_ffn2} \
                -DFEATURE_LOAD_TILE_SIZE_FFN2={feature_load_tile_var_ffn2} -DCB_LOAD_TILE_SIZE_FFN2={cb_load_tile_var_ffn2} \
                ..'
    os.system('cd build; %s; make -j; cd ../' % command)


def main():
    args = parse_args()
    if args.need_compile or args.need_breakdown:
        compile(args)
    os.system('./build/bin/test_transformer %s %s' % (args.transformer_config_file, args.model_weight_file))


if __name__ == '__main__':
    main()
