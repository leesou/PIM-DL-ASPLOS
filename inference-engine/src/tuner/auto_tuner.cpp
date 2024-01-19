#include <string>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <map>

#include "auto_tuner.h"
#include "yaml-cpp/yaml.h"
using namespace std;

#define THREAD_NUM 40


void init_single_kernel_params(string input_path, AMMParams& amm_params, PIMParams& pim_params, int& num_threads)
{
    YAML::Node config;

    // read yaml file
    try{
        config = YAML::LoadFile(input_path);
    }catch(YAML::BadFile &e){
        std::cerr<<"Config File Not Open"<<std::endl;
    }

    // parse yaml configs
    try
    {
        pim_params.pe_num = config["system_params"]["dpu_num"].as<int>();
        pim_params.parallelism = config["system_params"]["nr_tasklets"].as<int>();
        num_threads = config["system_params"]["num_threads"].as<int>();

        amm_params.input_feature_len = config["amm_shape_params"]["input_feature_len"].as<int>();
        amm_params.n = config["amm_shape_params"]["n"].as<int>();
        amm_params.num_centroid = config["amm_shape_params"]["num_centroid"].as<int>();
        amm_params.num_codebook = config["amm_shape_params"]["num_codebook"].as<int>();
        amm_params.output_feature_len = config["amm_shape_params"]["output_feature_len"].as<int>();

    }catch(YAML::TypedBadConversion<std::string> &e){
        std::cerr<<"Error in parsing configs"<<std::endl;
    }
}

void dump_single_kernel_params(string output_path, AMMParams& amm_params, PIMParams& pim_params, int& num_threads, KernelParams& kernel_params)
{
    ofstream output_file(output_path);
    output_file << "system_params:\n";
    output_file << "  num_threads: " << num_threads << "\n";
    output_file << "  dpu_num: " << pim_params.pe_num << "\n";
    output_file << "  nr_tasklets: " << pim_params.parallelism << "\n";
    output_file << "\n";
    output_file << "amm_shape_params:\n";
    output_file << "  num_codebook: " << amm_params.num_codebook << "\n";
    output_file << "  num_centroid: " << amm_params.num_centroid << "\n";
    output_file << "  n: " << amm_params.n << "\n";
    output_file << "  input_feature_len: " << amm_params.input_feature_len << "\n";
    output_file << "  output_feature_len: " << amm_params.output_feature_len << "\n";
    output_file << "  scale: " << 0.1 << "\n";
    output_file << "  bias: " << 0.2 << "\n";
    output_file << "\n";
    output_file << "kernel_params:\n";
    output_file << "  loop_order: " << kernel_params.loop_order << "\n";
    output_file << "  lut_load_type: " << kernel_params.lut_load_type << "\n";
    output_file << "  n_stile_size: " << kernel_params.n_stile_size << "\n";
    output_file << "  feature_stile_size: " << kernel_params.feature_stile_size << "\n";
    output_file << "  n_mtile_size: " << kernel_params.n_mtile_size << "\n";
    output_file << "  feature_mtile_size: " << kernel_params.feature_mtile_size << "\n";
    output_file << "  cb_mtile_size: " << kernel_params.cb_mtile_size << "\n";
    output_file << "  feature_load_tile_size: " << kernel_params.feature_load_tile_size << "\n";
    output_file << "  cb_load_tile_size: " << kernel_params.cb_load_tile_size << "\n";
}


void get_factors(int x, vector<int>& factors)
{
    factors.clear();
    for(int i=1; i<=x; ++i)
        if(x%i == 0)
            factors.push_back(i);
}

void generate_sub_lut_tiling_space(AMMParams& amm_params, PIMParams& pim_params, vector<KernelParams>& sub_lut_tiling_space)
{
    sub_lut_tiling_space.clear();

    vector<int> pe_num_factors;
    get_factors(pim_params.pe_num, pe_num_factors);

    for(int input_parallelism : pe_num_factors)
    {
        int lut_parallelism = pim_params.pe_num / input_parallelism;

        if(amm_params.n%input_parallelism != 0)
            continue;
        if(amm_params.output_feature_len%lut_parallelism != 0)
            continue;

        KernelParams kernel_params;
        kernel_params.input_parallelism = input_parallelism;
        kernel_params.lut_parallelism = lut_parallelism;
        kernel_params.n_stile_size = amm_params.n / input_parallelism;
        kernel_params.feature_stile_size = amm_params.output_feature_len / lut_parallelism;
        sub_lut_tiling_space.push_back(kernel_params);
    }
}

/* Latency Estimator Begin */

double communication_calculator(AMMParams& amm_params, PIMParams& pim_params, KernelParams& sub_lut_tiling, bool verbose=false)
{
    double input_total_size = double(pim_params.pe_num) * double(sub_lut_tiling.n_stile_size) * double(amm_params.num_codebook) * 2 * 1.0 / (1024 * 1024);
    double lut_total_size = double(pim_params.pe_num) * double(amm_params.num_codebook) * double(amm_params.num_centroid) * double(sub_lut_tiling.feature_stile_size) * 1 * 1.0 / (1024 * 1024);
    double output_total_size = double(pim_params.pe_num) * double(sub_lut_tiling.n_stile_size) * double(sub_lut_tiling.feature_stile_size) * 4 * 1.0 / (1024 * 1024);

    double input_send_latency, lut_send_latency, output_read_latency = 0;
    double bw_input=0, bw_lut=0, bw_output=0;

    // depends on system configurations
    if(input_total_size<=128)
    {
        bw_input = input_total_size/0.0207961034196259;
    }
    else if(input_total_size<=4096)
    {
        bw_input = input_total_size/(5e-05*input_total_size+0.0166);
    }
    else if(input_total_size<=32768)
    {
        bw_input = input_total_size/(3e-05*input_total_size+0.1266);
    }
    else
    {
        bw_input = 29530;
    }
    input_send_latency = input_total_size / bw_input;
    
    // depends on system configurations
    if(lut_total_size<=256)
    {
        bw_lut = lut_total_size/(0.0001*lut_total_size+0.0037);
    }
    else if(lut_total_size<=2048)
    {
        bw_lut = lut_total_size/(7e-05*lut_total_size+0.0152);
    }
    else if(lut_total_size<=32768)
    {
        bw_lut = lut_total_size/(2e-05*lut_total_size+0.108);
    }
    else
    {
        bw_lut = 42926;
    }
    lut_send_latency = lut_total_size / bw_lut;

    // depends on system configurations
    if(output_total_size<=0.25)
    {
        bw_output = 1024 * (0.7614 * log(2) * log2(output_total_size * 1024) + 1.4376);
    }
    else
    {
        bw_output = min(48.85892*output_total_size+4360.88152, 4678.72695);
    }
    output_read_latency = output_total_size / bw_output;

    double total_latency = input_send_latency + lut_send_latency + output_read_latency;
    return total_latency;
}


double calc_inout_latency(double tile_size, bool is_output_reset=false, bool is_write=false)
{
    if(is_output_reset)
        return (0.5*tile_size+61)*(0.1448*log(tile_size)+0.3421);
    if(is_write)
        return (0.5*tile_size+77);
    return (0.5*tile_size+61);
}

double calc_lut_read_latency(double tile_size, int lut_load_type)
{
    if(lut_load_type==0) // static
        return max(9.0415*tile_size+551.86, 2826.28);
    if(lut_load_type==2) // coarse
        return (8.9985*tile_size+537.81);
    return 191981000; // we fuse static lut read latency with compute latency
}

double calc_lut_reduce_latency(KernelParams& kernel_params)
{
    if(kernel_params.lut_load_type==0) // static
    {
        if(kernel_params.feature_mtile_size==1 && kernel_params.n_stile_size!=32)
            return 28.511*kernel_params.feature_mtile_size+75.696;
        return 38.949*kernel_params.feature_mtile_size+103.41;
    }
    else if(kernel_params.lut_load_type==1) // fine
    {
        if(kernel_params.feature_load_tile_size<=8)
            return (211.926*kernel_params.feature_load_tile_size-910.0476);
        else if(kernel_params.feature_load_tile_size>=64)
            return (151.301*kernel_params.feature_load_tile_size-649.711);
        else if(kernel_params.feature_load_tile_size>=32)
            return (127.578*kernel_params.feature_load_tile_size-547.840);
        else
            return (263.59*kernel_params.feature_load_tile_size-1131.9) * 0.544;
    }
    else if(kernel_params.lut_load_type==2) // coarse
    {
        unordered_map<int, unordered_map<int, double>> latency_table = {
            {512, { {1, 51067.17578}, {2, 83819.19141} }},
            {256, { {1, 24655.99219}, {2, 41049.60156}, {4, 57436.98438}, {8, 102510.4688} }},
            {128, { {1, 12940.47363}, {2, 21114.01367}, {4, 29433.19531}, {8, 46137.89844}, {16, 102552.7344} }},
            {64,  { {1, 6195.875}, {2, 10310.23926}, {4, 14408.08008}, {8, 25677.71484}, {16, 51319.8125}, {32, 121828.7344} }},
            {32,  { {1, 2301.281494}, {2, 6059.832031}, {4, 7954.683594}, {8, 12201.95898}, {16, 25710.22656}, {32, 61379.42969}, {64, 265429.9375} }},
            {16,  { {1, 3983.648071}, {2, 4704.496826}, {4, 5351.220215}, {8, 8604.514648}, {16, 13771.4043}, {32, 33026.37891}, {64, 134484.4766}, {128, 265066.5469} }},
            {8,   { {1, 3283.255127}, {2, 3695.32605}, {4, 4210.680176}, {8, 5120.65918}, {16, 9174.716797}, {32, 22397.05664}, {64, 67897.30859}, {128, 133584.5859} }},
            {4,   { {2, 3149.981567}, {4, 3456.32959}, {8, 4083.037842}, {16, 5453.440918}, {32, 14559.79102}, {64, 34833.63672}, {128, 67646}, {256, 133296.4219} }},
            {2,   { {4, 1854.964294}, {8, 2191.868042}, {16, 3550.974609}, {32, 5696.202637}, {64, 17630.43359}, {128, 33838.29102}, {256, 66676.73047} }},
            {1,   { {8, 3225.414856}, {16, 4339.175171}, {32, 5940.764893}, {64, 10682.63428}, {128, 18547.72168}, {256, 34614.73438}, {512, 67431.4375} }}
        };
        return latency_table[kernel_params.cb_load_tile_size][kernel_params.feature_load_tile_size];
    }
    return 191981000;
}

double computation_calculator(AMMParams& amm_params, PIMParams& pim_params, KernelParams& kernel_params)
{
    double total_latency = 0;
    if(kernel_params.lut_load_type==0) // static
    {
        double compute_count = (kernel_params.n_stile_size * kernel_params.feature_stile_size * amm_params.num_codebook) / (pim_params.parallelism * kernel_params.feature_mtile_size);
        double compute_latency = compute_count * calc_lut_reduce_latency(kernel_params) / (350 * 1e6);

        double input_tile_per_tasklet = kernel_params.n_mtile_size * kernel_params.cb_mtile_size * 2 / pim_params.parallelism;
        double input_count = 0;
        if((kernel_params.loop_order==1) || (kernel_params.loop_order==4))
            input_count = (kernel_params.n_stile_size * amm_params.num_codebook) / (kernel_params.n_mtile_size * kernel_params.cb_mtile_size);
        else
            input_count = (kernel_params.feature_stile_size * kernel_params.n_stile_size * amm_params.num_codebook) / (kernel_params.feature_mtile_size * kernel_params.n_mtile_size * kernel_params.cb_mtile_size);
        double input_read_latency = calc_inout_latency(input_tile_per_tasklet);
        double input_transfer_latency = input_count * input_read_latency / (350 * 1e6);

        double output_tile_per_tasklet = kernel_params.n_mtile_size * kernel_params.feature_mtile_size * 4 / pim_params.parallelism;
        double output_count=0, output_read_latency=0, output_write_latency=0;
        if(kernel_params.loop_order==0 || kernel_params.loop_order==2)
        {
            output_count = (kernel_params.feature_stile_size * kernel_params.n_stile_size) / (kernel_params.feature_mtile_size * kernel_params.n_mtile_size);
            output_read_latency = calc_inout_latency(output_tile_per_tasklet, true, false);
            output_write_latency = calc_inout_latency(output_tile_per_tasklet, false, true);
        }    
        else
        {
            output_count = (kernel_params.feature_stile_size * kernel_params.n_stile_size * amm_params.num_codebook) / (kernel_params.feature_mtile_size * kernel_params.n_mtile_size * kernel_params.cb_mtile_size);
            output_read_latency = calc_inout_latency(output_tile_per_tasklet);
            output_write_latency = calc_inout_latency(output_tile_per_tasklet, false, true);
        }
        double output_transfer_latency = output_count * (output_read_latency + output_write_latency) / (350 * 1e6);

        double lut_tile_per_tasklet = amm_params.num_codebook * amm_params.num_centroid * kernel_params.feature_stile_size * 1 / pim_params.parallelism;
        double lut_read_latency = calc_lut_read_latency(lut_tile_per_tasklet, kernel_params.lut_load_type);
        double lut_transfer_latency = lut_read_latency / (350 * 1e6);

        total_latency = compute_latency + input_transfer_latency + output_transfer_latency + lut_transfer_latency;
    }
    else if(kernel_params.lut_load_type==1) // fine grain
    {
        double compute_count = (kernel_params.n_stile_size * kernel_params.feature_stile_size * amm_params.num_codebook) / (pim_params.parallelism * kernel_params.feature_load_tile_size);
        double compute_latency = compute_count * calc_lut_reduce_latency(kernel_params) / (350 * 1e6);

        double input_tile_per_tasklet = kernel_params.n_mtile_size * kernel_params.cb_mtile_size * 2 / pim_params.parallelism;
        double input_count = 0;
        if(kernel_params.loop_order==1 || kernel_params.loop_order==4)
            input_count = (kernel_params.n_stile_size * amm_params.num_codebook) / (kernel_params.n_mtile_size * kernel_params.cb_mtile_size);
        else
            input_count = (kernel_params.feature_stile_size * kernel_params.n_stile_size * amm_params.num_codebook) / (kernel_params.feature_mtile_size * kernel_params.n_mtile_size * kernel_params.cb_mtile_size);
        double input_read_latency = calc_inout_latency(input_tile_per_tasklet);
        double input_transfer_latency = input_count * input_read_latency / (350 * 1e6);

        double output_tile_per_tasklet = kernel_params.n_mtile_size * kernel_params.feature_mtile_size * 4 / pim_params.parallelism;
        double output_count=0, output_read_latency=0, output_write_latency=0;
        if(kernel_params.loop_order==0 || kernel_params.loop_order==2)
        {
            output_count = (kernel_params.feature_stile_size * kernel_params.n_stile_size) / (kernel_params.feature_mtile_size * kernel_params.n_mtile_size);
            output_read_latency = calc_inout_latency(output_tile_per_tasklet, true, false);
            output_write_latency = calc_inout_latency(output_tile_per_tasklet, false, true);
        }
        else
        {
            output_count = (kernel_params.feature_stile_size * kernel_params.n_stile_size * amm_params.num_codebook) / (kernel_params.feature_mtile_size * kernel_params.n_mtile_size * kernel_params.cb_mtile_size);
            output_read_latency = calc_inout_latency(output_tile_per_tasklet);
            output_write_latency = calc_inout_latency(output_tile_per_tasklet, false, true);
        }
        double output_transfer_latency = output_count * (output_read_latency + output_write_latency) / (350 * 1e6);

        double lut_tile_per_tasklet = kernel_params.feature_load_tile_size * 1;
        // lut latency is contained in compute latency
        double lut_transfer_latency = 0;

        total_latency = compute_latency + input_transfer_latency + output_transfer_latency + lut_transfer_latency;
    }
    else if(kernel_params.lut_load_type==2) // coarse grain
    {
        double compute_count = (kernel_params.n_stile_size * kernel_params.feature_stile_size * amm_params.num_codebook) / (pim_params.parallelism * kernel_params.feature_load_tile_size * kernel_params.cb_load_tile_size);
        double compute_latency = compute_count * calc_lut_reduce_latency(kernel_params) / (350 * 1e6);

        double input_tile_per_tasklet = kernel_params.n_mtile_size * kernel_params.cb_mtile_size * 2 / pim_params.parallelism;
        double input_count = 0;
        if(kernel_params.loop_order==1 || kernel_params.loop_order==4)
            input_count = (kernel_params.n_stile_size * amm_params.num_codebook) / (kernel_params.n_mtile_size * kernel_params.cb_mtile_size);
        else
            input_count = (kernel_params.feature_stile_size * kernel_params.n_stile_size * amm_params.num_codebook) / (kernel_params.feature_mtile_size * kernel_params.n_mtile_size * kernel_params.cb_mtile_size);
        double input_read_latency = calc_inout_latency(input_tile_per_tasklet);
        double input_transfer_latency = input_count * input_read_latency / (350 * 1e6);

        double output_tile_per_tasklet = kernel_params.n_mtile_size * kernel_params.feature_mtile_size * 4 / pim_params.parallelism;
        double output_count=0, output_read_latency=0, output_write_latency=0;
        if(kernel_params.loop_order==0 || kernel_params.loop_order==2)
        {
            output_count = (kernel_params.feature_stile_size * kernel_params.n_stile_size) / (kernel_params.feature_mtile_size * kernel_params.n_mtile_size);
            output_read_latency = calc_inout_latency(output_tile_per_tasklet, true, false);
            output_write_latency = calc_inout_latency(output_tile_per_tasklet, true, true);
        }
        else
        {
            output_count = (kernel_params.feature_stile_size * kernel_params.n_stile_size * amm_params.num_codebook) / (kernel_params.feature_mtile_size * kernel_params.n_mtile_size * kernel_params.cb_mtile_size);
            output_read_latency = calc_inout_latency(output_tile_per_tasklet);
            output_write_latency = calc_inout_latency(output_tile_per_tasklet, true, true);
        }
        double output_transfer_latency = output_count * (output_read_latency + output_write_latency) / (350 * 1e6);

        double lut_tile_per_tasklet = kernel_params.cb_load_tile_size * kernel_params.feature_load_tile_size * amm_params.num_centroid * 1 / pim_params.parallelism;
        double lut_count = (kernel_params.n_stile_size * kernel_params.feature_stile_size * amm_params.num_codebook) / (kernel_params.n_mtile_size * kernel_params.feature_load_tile_size * kernel_params.cb_load_tile_size);
        double lut_read_latency = calc_lut_read_latency(lut_tile_per_tasklet, kernel_params.lut_load_type);
        double lut_transfer_latency = lut_count * lut_read_latency / (350 * 1e6);

        total_latency = compute_latency + input_transfer_latency + output_transfer_latency + lut_transfer_latency;
    }
    else
        total_latency = 1919810000;

    return total_latency;
}

/* Latency Estimator End */

void gen_mtile_product(vector<int>& n_mtile_list, vector<int>& feature_mtile_list, vector<int>& cb_mtile_list, vector<vector<int>>& mtile_product, int parallelism)
{
    mtile_product.clear();
    for(int n_mtile: n_mtile_list)
    {
        if(n_mtile % parallelism != 0)
            continue;
        for(int feature_mtile: feature_mtile_list)
        {
            for(int cb_mtile: cb_mtile_list)
            {
                vector<int> tmp_product = {n_mtile, feature_mtile, cb_mtile};
                mtile_product.push_back(tmp_product);
            }
        }
    }
}

double micro_kernel_searcher(AMMParams& amm_params, PIMParams& pim_params, KernelParams& sub_lut_tiling, KernelParams& best_kernel, bool verbose=false)
{
    double min_computation_latency = 191981000;
    
    vector<int> n_mtile_list, feature_mtile_list, cb_mtile_list;
    get_factors(sub_lut_tiling.n_stile_size, n_mtile_list);
    get_factors(sub_lut_tiling.feature_stile_size, feature_mtile_list);
    get_factors(amm_params.num_codebook, cb_mtile_list);
    if(verbose)
        printf("n_mtile factor num %d, feature_mtile factor num %d, cb_mtile factor num %d\n",
                n_mtile_list.size(), feature_mtile_list.size(), cb_mtile_list.size());

    vector<vector<int>> mtile_product;  
    gen_mtile_product(n_mtile_list, feature_mtile_list, cb_mtile_list, mtile_product, pim_params.parallelism);
    vector<int> lut_load_types = {0, 1, 2};
    vector<int> loop_order_types = {0, 1, 2, 3, 4, 5};

    vector<double> per_thread_min_computation_latency(THREAD_NUM, 191981000);
    vector<KernelParams> per_thread_best_kernel(THREAD_NUM, KernelParams());
    #pragma omp parallel for num_threads(THREAD_NUM)
    for(auto tmp_product: mtile_product)
    {
        int thread_id = omp_get_thread_num();

        int n_mtile = tmp_product[0];
        int feature_mtile = tmp_product[1];
        int cb_mtile = tmp_product[2];
        for(int lut_load_type: lut_load_types)
        {
            if(lut_load_type==0) // static
            {
                double on_chip_size = n_mtile * cb_mtile * 2
                                      + n_mtile * feature_mtile * 4
                                      + amm_params.num_codebook * amm_params.num_centroid * sub_lut_tiling.feature_stile_size * 1;  
                int input_tile_per_tasklet = n_mtile * cb_mtile * 2 / pim_params.parallelism;
                int output_tile_per_tasklet = n_mtile * feature_mtile * 4 / pim_params.parallelism;
                int lut_tile_per_tasklet = amm_params.num_codebook * amm_params.num_centroid * sub_lut_tiling.feature_stile_size * 1 / pim_params.parallelism;

                if(on_chip_size > (48*1024-656))
                    continue;
                if((input_tile_per_tasklet<8) || (output_tile_per_tasklet<8) || (lut_tile_per_tasklet<8))
                    continue;
                if(((input_tile_per_tasklet%8)!=0) || ((output_tile_per_tasklet%8)!=0) || ((lut_tile_per_tasklet%8)!=0))
                    continue;

                for(int loop_order: loop_order_types)
                {
                    KernelParams tmp_kernel_design;
                    memcpy(&tmp_kernel_design, &sub_lut_tiling, sizeof(KernelParams));
                    tmp_kernel_design.loop_order = loop_order;
                    tmp_kernel_design.lut_load_type = lut_load_type;
                    tmp_kernel_design.n_mtile_size = n_mtile;
                    tmp_kernel_design.feature_mtile_size = feature_mtile;
                    tmp_kernel_design.cb_mtile_size = cb_mtile;

                    double tmp_micro_kernel_latency = computation_calculator(amm_params, pim_params, tmp_kernel_design);

                    if(tmp_micro_kernel_latency<per_thread_min_computation_latency[thread_id])
                    {
                        per_thread_min_computation_latency[thread_id] = tmp_micro_kernel_latency;
                        memcpy(&per_thread_best_kernel[thread_id], &tmp_kernel_design, sizeof(KernelParams));
                    }
                }
            }
            else if(lut_load_type==1) // fine grain
            {
                vector<int> feature_load_tile_list;
                get_factors(feature_mtile, feature_load_tile_list);
                for(int feature_load_tile: feature_load_tile_list)
                {
                    double on_chip_size = n_mtile * cb_mtile * 2
                                          + n_mtile * feature_mtile * 4
                                          + pim_params.parallelism * feature_load_tile * 1;
                    int input_tile_per_tasklet = n_mtile * cb_mtile * 2 / pim_params.parallelism;
                    int output_tile_per_tasklet = n_mtile * feature_mtile * 4 / pim_params.parallelism;
                    int lut_tile_per_tasklet = feature_load_tile * 1;

                    if(on_chip_size > (48*1024-656))
                        continue;
                    if(input_tile_per_tasklet<8 || output_tile_per_tasklet<8 || lut_tile_per_tasklet<8)
                        continue;
                    if(((input_tile_per_tasklet%8)!=0) || ((output_tile_per_tasklet%8)!=0) || ((lut_tile_per_tasklet%8)!=0))
                        continue;
                    
                    for(int loop_order: loop_order_types)
                    {
                        KernelParams tmp_kernel_design;
                        memcpy(&tmp_kernel_design, &sub_lut_tiling, sizeof(KernelParams));
                        tmp_kernel_design.loop_order = loop_order;
                        tmp_kernel_design.lut_load_type = lut_load_type;
                        tmp_kernel_design.n_mtile_size = n_mtile;
                        tmp_kernel_design.feature_mtile_size = feature_mtile;
                        tmp_kernel_design.cb_mtile_size = cb_mtile;
                        tmp_kernel_design.feature_load_tile_size = feature_load_tile;

                        double tmp_micro_kernel_latency = computation_calculator(amm_params, pim_params, tmp_kernel_design);

                        if(tmp_micro_kernel_latency<per_thread_min_computation_latency[thread_id])
                        {
                            per_thread_min_computation_latency[thread_id] = tmp_micro_kernel_latency;
                            memcpy(&per_thread_best_kernel[thread_id], &tmp_kernel_design, sizeof(KernelParams));
                        }
                    }
                }
            }
            else if(lut_load_type==2) // coarse grain
            {
                if(((feature_mtile%3)==0) || ((cb_mtile%3)==0))
                    continue;
                vector<int> feature_load_tile_list, cb_load_tile_list;
                get_factors(feature_mtile, feature_load_tile_list);
                get_factors(cb_mtile, cb_load_tile_list);
                for(int feature_load_tile: feature_load_tile_list)
                {
                    for(int cb_load_tile: cb_load_tile_list)
                    {
                        if(((feature_load_tile%3)==0) || ((cb_load_tile%3)==0))
                            continue;
                        
                        double on_chip_size = n_mtile * cb_mtile * 2
                                              + n_mtile * feature_mtile * 4
                                              + cb_load_tile * feature_load_tile * amm_params.num_centroid * 1;
                        int input_tile_per_tasklet = n_mtile * cb_mtile * 2 / pim_params.parallelism;
                        int output_tile_per_tasklet = n_mtile * feature_mtile * 4 / pim_params.parallelism;
                        int lut_tile_per_tasklet = cb_load_tile * feature_load_tile * amm_params.num_centroid * 1 / pim_params.parallelism;

                        if(on_chip_size > (48*1024-656))
                            continue;
                        if(input_tile_per_tasklet<8 || output_tile_per_tasklet<8 || lut_tile_per_tasklet<8)
                            continue;
                        if(((input_tile_per_tasklet%8)!= 0) || ((output_tile_per_tasklet%8)!=0) || ((lut_tile_per_tasklet%8)!=0))
                            continue;
                        
                        for(int loop_order: loop_order_types)
                        {
                            KernelParams tmp_kernel_design;
                            memcpy(&tmp_kernel_design, &sub_lut_tiling, sizeof(KernelParams));
                            tmp_kernel_design.loop_order = loop_order;
                            tmp_kernel_design.lut_load_type = lut_load_type;
                            tmp_kernel_design.n_mtile_size = n_mtile;
                            tmp_kernel_design.feature_mtile_size = feature_mtile;
                            tmp_kernel_design.cb_mtile_size = cb_mtile;
                            tmp_kernel_design.feature_load_tile_size = feature_load_tile;
                            tmp_kernel_design.cb_load_tile_size = cb_load_tile;

                            double tmp_micro_kernel_latency = computation_calculator(amm_params, pim_params, tmp_kernel_design);

                            if(tmp_micro_kernel_latency<per_thread_min_computation_latency[thread_id])
                            {
                                per_thread_min_computation_latency[thread_id] = tmp_micro_kernel_latency;
                                memcpy(&per_thread_best_kernel[thread_id], &tmp_kernel_design, sizeof(KernelParams));
                            }
                        }
                    }
                }
            }
        }
    }

    for(int i=0; i<THREAD_NUM; ++i)
    {
        if(min_computation_latency > per_thread_min_computation_latency[i])
        {
            min_computation_latency = per_thread_min_computation_latency[i];
            memcpy(&best_kernel, &per_thread_best_kernel[i], sizeof(KernelParams));
        }
    }
    
    if(verbose)
    {
        printf("current sub lut tiling\'s best micro kernel\'s settings are:\n");
        printf("loop_order %d, lut_load_type %d\n", best_kernel.loop_order, best_kernel.lut_load_type);
        printf("n_mtile %d, feature_mtile %d, cb_mtile %d\n", best_kernel.n_mtile_size, best_kernel.feature_mtile_size, best_kernel.cb_mtile_size);
        if(best_kernel.lut_load_type == 1)
            printf("feature_load_tile %d\n", best_kernel.feature_load_tile_size);
        else if(best_kernel.lut_load_type == 2)
            printf("feature_load_tile %d, cb_load_tile %d\n", best_kernel.feature_load_tile_size, best_kernel.cb_load_tile_size);
        printf("------------------------------\n");
    }

    return min_computation_latency;
}


void tune_single_kernel(std::string input_path, std::string output_path, bool verbose=false)
{
    AMMParams amm_params;
    PIMParams pim_params;
    int num_threads;
    init_single_kernel_params(input_path, amm_params, pim_params, num_threads);

    vector<KernelParams> sub_lut_tiling_space;
    generate_sub_lut_tiling_space(amm_params, pim_params, sub_lut_tiling_space);

    double min_cost = 191981000;
    KernelParams best_kernel;
    for(KernelParams sub_lut_tiling : sub_lut_tiling_space)
    {
        double total_data_size = sub_lut_tiling.n_stile_size * amm_params.num_codebook * 2
                               + amm_params.num_codebook * amm_params.num_centroid * sub_lut_tiling.feature_stile_size * 1
                               + sub_lut_tiling.n_stile_size * sub_lut_tiling.feature_stile_size * 4;
        total_data_size = total_data_size / (1024 * 1024);
        if(total_data_size > 64)
        {
            if(verbose)
                printf("total data size %.6f MB, is larger than PE\'s max memory capacity (64 MB)\n", total_data_size);
            continue;
        }

        KernelParams tmp_best_kernel;
        memcpy(&tmp_best_kernel, &sub_lut_tiling, sizeof(KernelParams));
        double communication_latency = communication_calculator(amm_params, pim_params, tmp_best_kernel, verbose);
        double min_computation_latency = micro_kernel_searcher(amm_params, pim_params, sub_lut_tiling, tmp_best_kernel, verbose);
        
        if(min_computation_latency+communication_latency < min_cost)
        {
            min_cost = min_computation_latency+communication_latency;
            memcpy(&best_kernel, &tmp_best_kernel, sizeof(KernelParams));
        }
    }

    if(verbose)
    {
        printf("best kernel settings are:\n");
        printf("n_stile %d, feature_stile %d, input_parallelism %d, lut_parallelism %d\n", 
                best_kernel.n_stile_size, best_kernel.feature_stile_size, best_kernel.input_parallelism, best_kernel.lut_parallelism);
        printf("loop order %d, lut load type %d\n", best_kernel.loop_order, best_kernel.lut_load_type);
        printf("n_mtile %d, feature_mtile %d, cb_mtile %d\n", 
                best_kernel.n_mtile_size, best_kernel.feature_mtile_size, best_kernel.cb_mtile_size);
        if(best_kernel.lut_load_type==1)
            printf("feature_load_tile %d\n", best_kernel.feature_load_tile_size);
        else if(best_kernel.lut_load_type==2)
            printf("feature_load_tile %d, cb_load_tile %d\n", best_kernel.feature_load_tile_size, best_kernel.cb_load_tile_size);
    }
    dump_single_kernel_params(output_path, amm_params, pim_params, num_threads, best_kernel);
}


void init_single_model_params(string input_path, vector<AMMParams>& amm_param_list, PIMParams& pim_params, NetworkParams& network_params, int& num_threads, vector<string>& kernel_binary_list)
{
    YAML::Node config;

    // read yaml file
    try{
        config = YAML::LoadFile(input_path);
    }catch(YAML::BadFile &e){
        std::cerr<<"Config File Not Open"<<std::endl;
    }

    // parse yaml configs
    try
    {
        pim_params.pe_num = config["system_params"]["dpu_num"].as<int>();
        pim_params.parallelism = config["system_params"]["nr_tasklets"].as<int>();
        num_threads = config["system_params"]["num_threads"].as<int>();
        kernel_binary_list[0] = config["system_params"]["qkv_lut_pim_binary"].as<string>();
        kernel_binary_list[1] = config["system_params"]["o_lut_pim_binary"].as<string>();
        kernel_binary_list[2] = config["system_params"]["ffn1_lut_pim_binary"].as<string>();
        kernel_binary_list[3] = config["system_params"]["ffn2_lut_pim_binary"].as<string>();

        network_params.seq_len = config["network_params"]["seq_len"].as<int>();
        network_params.batch_size = config["network_params"]["batch_size"].as<int>();
        network_params.head_num = config["network_params"]["head_num"].as<int>();
        network_params.head_dim = config["network_params"]["head_dim"].as<int>();
        network_params.token_dim = config["network_params"]["token_dim"].as<int>();
        network_params.ffn_hidden_dim = config["network_params"]["ffn_hidden_dim"].as<int>();
        network_params.layer_num = config["network_params"]["layer_num"].as<int>();

        amm_param_list[0].n = network_params.seq_len*network_params.batch_size;
        amm_param_list[0].input_feature_len = network_params.token_dim;  
        amm_param_list[0].output_feature_len = network_params.token_dim*3;
        amm_param_list[0].num_codebook = config["kernel_params"]["qkv_num_codebook"].as<int>();        
        amm_param_list[0].num_centroid = config["kernel_params"]["qkv_num_centroid"].as<int>();

        amm_param_list[1].n = network_params.seq_len*network_params.batch_size;
        amm_param_list[1].input_feature_len = network_params.token_dim;  
        amm_param_list[1].output_feature_len = network_params.token_dim;
        amm_param_list[1].num_codebook = config["kernel_params"]["o_num_codebook"].as<int>();        
        amm_param_list[1].num_centroid = config["kernel_params"]["o_num_centroid"].as<int>();

        amm_param_list[2].n = network_params.seq_len*network_params.batch_size;
        amm_param_list[2].input_feature_len = network_params.token_dim;  
        amm_param_list[2].output_feature_len = network_params.ffn_hidden_dim;
        amm_param_list[2].num_codebook = config["kernel_params"]["ffn1_num_codebook"].as<int>();        
        amm_param_list[2].num_centroid = config["kernel_params"]["ffn1_num_centroid"].as<int>();

        amm_param_list[3].n = network_params.seq_len*network_params.batch_size;
        amm_param_list[3].input_feature_len = network_params.token_dim;  
        amm_param_list[3].output_feature_len = network_params.ffn_hidden_dim;
        amm_param_list[3].num_codebook = config["kernel_params"]["ffn2_num_codebook"].as<int>();        
        amm_param_list[3].num_centroid = config["kernel_params"]["ffn2_num_centroid"].as<int>();

    }catch(YAML::TypedBadConversion<std::string> &e){
        std::cerr<<"Error in parsing configs"<<std::endl;
    }
}

void dump_single_model_params(string output_path, vector<AMMParams>& amm_param_list, PIMParams& pim_params, NetworkParams& network_params, int& num_threads, vector<string>& kernel_binary_list, vector<KernelParams>& best_kernel_list)
{
    ofstream output_file(output_path);
    output_file << "network_params:\n";
    output_file << "  seq_len: " << network_params.seq_len << "\n";
    output_file << "  batch_size: " << network_params.batch_size << "\n";
    output_file << "  head_num: " << network_params.head_num << "\n";
    output_file << "  head_dim: " << network_params.head_dim << "\n";
    output_file << "  token_dim: " << network_params.token_dim << "\n";
    output_file << "  ffn_hidden_dim: " << network_params.ffn_hidden_dim << "\n";
    output_file << "  layer_num: " << network_params.layer_num << "\n";
    output_file << "\n";
    output_file << "system_params:\n";
    output_file << "  dpu_num: " << pim_params.pe_num << "\n";
    output_file << "  nr_tasklets: " << pim_params.parallelism << "\n";
    output_file << "  qkv_lut_pim_binary: " << kernel_binary_list[0] << "\n";
    output_file << "  o_lut_pim_binary: " << kernel_binary_list[1] << "\n";
    output_file << "  ffn1_lut_pim_binary: " << kernel_binary_list[2] << "\n";
    output_file << "  ffn2_lut_pim_binary: " << kernel_binary_list[3] << "\n";
    output_file << "  num_threads: " << num_threads << "\n";
    output_file << "\n";
    output_file << "kernel_params:\n";
    output_file << "  qkv_scale: " << 0.1 << "\n";
    output_file << "  qkv_bias: " << 0.2 << "\n";
    output_file << "  qkv_num_codebook: " << amm_param_list[0].num_codebook << "\n";
    output_file << "  qkv_num_centroid: " << amm_param_list[0].num_centroid << "\n";
    output_file << "  qkv_input_parallelism: " << best_kernel_list[0].input_parallelism << "\n";
    output_file << "  qkv_lut_parallelism: " << best_kernel_list[0].lut_parallelism << "\n";
    output_file << "  qkv_loop_order: " << best_kernel_list[0].loop_order << "\n";
    output_file << "  qkv_lut_load_type: " << best_kernel_list[0].lut_load_type << "\n";
    output_file << "  qkv_n_mtile_size: " << best_kernel_list[0].n_mtile_size << "\n";
    output_file << "  qkv_feature_mtile_size: " << best_kernel_list[0].feature_mtile_size << "\n";
    output_file << "  qkv_cb_mtile_size: " << best_kernel_list[0].cb_mtile_size << "\n";
    output_file << "  qkv_feature_load_tile_size: " << best_kernel_list[0].feature_load_tile_size << "\n";
    output_file << "  qkv_cb_load_tile_size: " << best_kernel_list[0].cb_load_tile_size << "\n";
    output_file << "\n";
    output_file << "  o_scale: " << 0.1 << "\n";
    output_file << "  o_bias: " << 0.2 << "\n";
    output_file << "  o_num_codebook: " << amm_param_list[1].num_codebook << "\n";
    output_file << "  o_num_centroid: " << amm_param_list[1].num_centroid << "\n";
    output_file << "  o_input_parallelism: " << best_kernel_list[1].input_parallelism << "\n";
    output_file << "  o_lut_parallelism: " << best_kernel_list[1].lut_parallelism << "\n";
    output_file << "  o_loop_order: " << best_kernel_list[1].loop_order << "\n";
    output_file << "  o_lut_load_type: " << best_kernel_list[1].lut_load_type << "\n";
    output_file << "  o_n_mtile_size: " << best_kernel_list[1].n_mtile_size << "\n";
    output_file << "  o_feature_mtile_size: " << best_kernel_list[1].feature_mtile_size << "\n";
    output_file << "  o_cb_mtile_size: " << best_kernel_list[1].cb_mtile_size << "\n";
    output_file << "  o_feature_load_tile_size: " << best_kernel_list[1].feature_load_tile_size << "\n";
    output_file << "  o_cb_load_tile_size: " << best_kernel_list[1].cb_load_tile_size << "\n";
    output_file << "\n";
    output_file << "  ffn1_scale: " << 0.1 << "\n";
    output_file << "  ffn1_bias: " << 0.2 << "\n";
    output_file << "  ffn1_num_codebook: " << amm_param_list[2].num_codebook << "\n";
    output_file << "  ffn1_num_centroid: " << amm_param_list[2].num_centroid << "\n";
    output_file << "  ffn1_input_parallelism: " << best_kernel_list[2].input_parallelism << "\n";
    output_file << "  ffn1_lut_parallelism: " << best_kernel_list[2].lut_parallelism << "\n";
    output_file << "  ffn1_loop_order: " << best_kernel_list[2].loop_order << "\n";
    output_file << "  ffn1_lut_load_type: " << best_kernel_list[2].lut_load_type << "\n";
    output_file << "  ffn1_n_mtile_size: " << best_kernel_list[2].n_mtile_size << "\n";
    output_file << "  ffn1_feature_mtile_size: " << best_kernel_list[2].feature_mtile_size << "\n";
    output_file << "  ffn1_cb_mtile_size: " << best_kernel_list[2].cb_mtile_size << "\n";
    output_file << "  ffn1_feature_load_tile_size: " << best_kernel_list[2].feature_load_tile_size << "\n";
    output_file << "  ffn1_cb_load_tile_size: " << best_kernel_list[2].cb_load_tile_size << "\n";
    output_file << "\n";
    output_file << "  ffn2_scale: " << 0.1 << "\n";
    output_file << "  ffn2_bias: " << 0.2 << "\n";
    output_file << "  ffn2_num_codebook: " << amm_param_list[3].num_codebook << "\n";
    output_file << "  ffn2_num_centroid: " << amm_param_list[3].num_centroid << "\n";
    output_file << "  ffn2_input_parallelism: " << best_kernel_list[3].input_parallelism << "\n";
    output_file << "  ffn2_lut_parallelism: " << best_kernel_list[3].lut_parallelism << "\n";
    output_file << "  ffn2_loop_order: " << best_kernel_list[3].loop_order << "\n";
    output_file << "  ffn2_lut_load_type: " << best_kernel_list[3].lut_load_type << "\n";
    output_file << "  ffn2_n_mtile_size: " << best_kernel_list[3].n_mtile_size << "\n";
    output_file << "  ffn2_feature_mtile_size: " << best_kernel_list[3].feature_mtile_size << "\n";
    output_file << "  ffn2_cb_mtile_size: " << best_kernel_list[3].cb_mtile_size << "\n";
    output_file << "  ffn2_feature_load_tile_size: " << best_kernel_list[3].feature_load_tile_size << "\n";
    output_file << "  ffn2_cb_load_tile_size: " << best_kernel_list[3].cb_load_tile_size << "\n";
    output_file << "\n";
}

void tune_single_model(std::string input_path, std::string output_path, bool verbose=false)
{
    vector<AMMParams> amm_param_list(4);
    PIMParams pim_params;
    NetworkParams network_params;
    int num_threads;
    vector<string> kernel_binary_list(4);
    init_single_model_params(input_path, amm_param_list, pim_params, network_params, num_threads, kernel_binary_list);

    vector<KernelParams> best_kernel_list(4);
    for(int i=0; i<4; ++i)
    {
        AMMParams& amm_params = amm_param_list[i];
        vector<KernelParams> sub_lut_tiling_space;
        generate_sub_lut_tiling_space(amm_params, pim_params, sub_lut_tiling_space);

        double min_cost = 191981000;
        KernelParams& best_kernel = best_kernel_list[i];
        for(KernelParams sub_lut_tiling : sub_lut_tiling_space)
        {
            double total_data_size = sub_lut_tiling.n_stile_size * amm_params.num_codebook * 2
                                + amm_params.num_codebook * amm_params.num_centroid * sub_lut_tiling.feature_stile_size * 1
                                + sub_lut_tiling.n_stile_size * sub_lut_tiling.feature_stile_size * 4;
            total_data_size = total_data_size / (1024 * 1024);
            if(total_data_size > 64)
            {
                if(verbose)
                    printf("total data size %.6f MB, is larger than PE\'s max memory capacity (64 MB)\n", total_data_size);
                continue;
            }

            KernelParams tmp_best_kernel;
            memcpy(&tmp_best_kernel, &sub_lut_tiling, sizeof(KernelParams));
            double communication_latency = communication_calculator(amm_params, pim_params, tmp_best_kernel, verbose);
            double min_computation_latency = micro_kernel_searcher(amm_params, pim_params, sub_lut_tiling, tmp_best_kernel, verbose);
            
            if(min_computation_latency+communication_latency < min_cost)
            {
                min_cost = min_computation_latency+communication_latency;
                memcpy(&best_kernel, &tmp_best_kernel, sizeof(KernelParams));
            }
        }
    }

    dump_single_model_params(output_path, amm_param_list, pim_params, network_params, num_threads, kernel_binary_list, best_kernel_list);
}


int main(int argc, char** argv)
{
    if(argc != 4)
    {
        printf("usage: %s type (0: single kernel, 1: single model) input_config_path output_config_path\n", argv[0]);
        exit(-1);
    }
    string type(argv[1]);
    string input_config_path(argv[2]);
    string output_config_path(argv[3]);

    printf("start tuning\n");
    if(type=="0")
    {
        tune_single_kernel(input_config_path, output_config_path);
    }    
    else if(type=="1")
    {
        tune_single_model(input_config_path, output_config_path);
    }
    else
    {
        printf("warning: unsupported tuning type (0: single kernel, 1: single model)\n");
    }
    printf("finish tuning\n");

    return 0;
}
