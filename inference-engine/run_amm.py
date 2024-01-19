import argparse
import os
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amm_config_file', type=str)
    parser.add_argument('--need_compile', action='store_true', default=True)

    args = parser.parse_args()
    return args


def compile(args):
    with open(args.amm_config_file, 'r') as f:
        yaml_data = yaml.load(f.read(), Loader=yaml.FullLoader)
    # dpu settings
    tl_var = yaml_data['system_params']['nr_tasklets']
    # shape settings
    cb_var = yaml_data['amm_shape_params']['num_codebook']
    ct_var = yaml_data['amm_shape_params']['num_centroid']
    # kernel settings
    loop_order_var = yaml_data['kernel_params']['loop_order']
    lut_load_type_var = yaml_data['kernel_params']['lut_load_type']
    n_stile_var = yaml_data['kernel_params']['n_stile_size']
    feature_stile_var = yaml_data['kernel_params']['feature_stile_size']
    n_mtile_var = yaml_data['kernel_params']['n_mtile_size']
    feature_mtile_var = yaml_data['kernel_params']['feature_mtile_size']
    cb_mtile_var = yaml_data['kernel_params']['cb_mtile_size']
    feature_load_tile_var = yaml_data['kernel_params']['feature_load_tile_size']
    cb_load_tile_var = yaml_data['kernel_params']['cb_load_tile_size']

    print("num tasklet %d" % tl_var)
    print("num codebook %d" % cb_var)
    print("num centroid %d" % ct_var)
    print("loop order %d (0: nfc, 1: ncf, 2: fnc, 3: fcn, 4: cnf, 5: cfn)" % loop_order_var)
    print("lut load type %d (0: static lut table, 1: fine grain, 2: coarse grain)" % lut_load_type_var)
    print("n stile size %d" % n_stile_var)
    print("feature stile size %d" % feature_stile_var)
    print("n mtile size %d" % n_mtile_var)
    print("feature mtile size %d" % feature_mtile_var)
    print("cb mtile size %d" % cb_mtile_var)
    print("feature load tile size %d" % feature_load_tile_var)
    print("cb load tile size %d" % cb_load_tile_var)

    os.system('mkdir -p build')
    command = f'cmake -DTRANSFORMER=0 -DDYNAMIC=1 -DLATENCY_BREAKDOWN=0 -DLUT_BREAKDOWN=0 -DLAYER_BREAKDOWN=0 -DNR_TASKLETS={tl_var} \
                -DNUM_CODEBOOK_D={cb_var} -DNUM_CENTROID_D={ct_var} -DLOOP_ORDER_D={loop_order_var} -DLUT_LOAD_TYPE_D={lut_load_type_var} \
                -DN_STILE_SIZE_D={n_stile_var} -DFEATURE_STILE_SIZE_D={feature_stile_var} -DN_MTILE_SIZE_D={n_mtile_var} -DFEATURE_MTILE_SIZE_D={feature_mtile_var} -DCB_MTILE_SIZE_D={cb_mtile_var} \
                -DFEATURE_LOAD_TILE_SIZE_D={feature_load_tile_var} -DCB_LOAD_TILE_SIZE_D={cb_load_tile_var} \
                ..'
    os.system('cd build; %s; make -j; cd ../' % command)


def main():
    args = parse_args()
    if args.need_compile:
        compile(args)
    os.system('./build/bin/test_amm %s ./build/bin/dpu_bin_dynamic' % args.amm_config_file)


if __name__ == '__main__':
    main()
