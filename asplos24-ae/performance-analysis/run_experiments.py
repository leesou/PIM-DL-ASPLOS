import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_only', action='store_true', default=False)
    parser.add_argument('--single_point', action='store_true', default=False)
    parser.add_argument('--experiment_type', type=str, default='latency')
    parser.add_argument('--model_type', type=str, default='bert-base')

    args = parser.parse_args()
    return args

def run_single_point(args):
    if args.experiment_type == 'latency':
        os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/performance-analysis/configs/%s/v4-ct16.yaml --need_compile --need_breakdown --breakdown_type=0 > ../asplos24-ae/performance-analysis/log/%s/latency_breakdown.txt 2>&1' % (args.model_type, args.model_type))
        os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/performance-analysis/configs/%s/v4-ct16.yaml --need_compile --need_breakdown --breakdown_type=1 > ../asplos24-ae/performance-analysis/log/%s/lut_breakdown.txt 2>&1' % (args.model_type, args.model_type))
    elif args.experiment_name == 'layer-wise':
        os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/performance-analysis/configs/%s/v4-ct16.yaml --need_compile --need_breakdown --breakdown_type=2 > ../asplos24-ae/performance-analysis/log/%s/layerwise_breakdown.txt 2>&1' % (args.model_type, args.model_type))

def run_latency_breakdown(plot_only):
    latency_breakdown = {
        'lut' : {},
        'ccs' : {},
        'other' : {}
    }
    experiment_names = ['bert-base', 'bert-large', 'vit-huge']
    for experiment_name in experiment_names:
        if not plot_only:
            os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/performance-analysis/configs/%s/v4-ct16.yaml --need_compile --need_breakdown --breakdown_type=0 > ../asplos24-ae/performance-analysis/log/%s/latency_breakdown.txt 2>&1' % (experiment_name, experiment_name))
            print("model %s, 1 of 2 jobs finished" % experiment_name)
            os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/performance-analysis/configs/%s/v4-ct16.yaml --need_compile --need_breakdown --breakdown_type=1 > ../asplos24-ae/performance-analysis/log/%s/lut_breakdown.txt 2>&1' % (experiment_name, experiment_name))
            print("model %s, 2 of 2 jobs finished" % experiment_name)

        inference_time = 0.
        lut_nn_time = 0.
        non_lut_nn_time = 0.
        with open('./log/%s/lut_breakdown.txt' % experiment_name, 'r') as f:
            for line in f.readlines():
                if 'amm time' in line:
                    non_lut_nn_time = float(line.split(' ')[-1])
                    lut_nn_time = float(line.split(' ')[2].split(',')[0])
                    inference_time = lut_nn_time + non_lut_nn_time

        ccs_time = 0.
        lut_time = 0.
        other_time = 0.
        with open('./log/%s/latency_breakdown.txt' % experiment_name, 'r') as f:
            lines = list(f.readlines())
            for i in range(len(lines)):
                if 'index calculation' in lines[i]:
                    ccs_time = float(lines[i].split(' ')[-1])
                elif 'other' in lines[i] and 'iteration' in lines[i+1]:
                    other_time = float(lines[i].split(' ')[2].split(',')[0])
                    lut_time = (float(lines[i].split(' ')[-1]) + float(lines[i].split(' ')[-5].split(',')[0]))

        latency_breakdown['lut'][experiment_name] = (lut_nn_time * lut_time / (ccs_time + lut_time + other_time)) / inference_time
        latency_breakdown['ccs'][experiment_name] = (lut_nn_time * ccs_time / (ccs_time + lut_time + other_time)) / inference_time
        latency_breakdown['other'][experiment_name] = (inference_time - lut_time - ccs_time) / inference_time
    
    print("latency breakdown starts plotting")
    lut_latency = list(latency_breakdown['lut'].values())
    ccs_latency = list(latency_breakdown['ccs'].values())
    other_latency = list(latency_breakdown['other'].values())

    fig, ax = plt.subplots()
    ax.bar(experiment_names, lut_latency, label='LUT')
    ax.bar(experiment_names, ccs_latency, label='CCS', bottom=lut_latency)
    ax.bar(experiment_names, other_latency, label='Other', bottom=np.add(lut_latency, ccs_latency))
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.legend()
    plt.savefig('./plot/inference-latency-breakdown.png')
    plt.close()
    print("latency breakdown finishes plotting")


def run_layer_wise_comparison(plot_only):
    cpu_baseline = {
        'bert-base' : {'QKV':0.6138157, 'O':0.302533167, 'FFN1':0.740642767, 'FFN2':1.2866067},
        'bert-large' : {'QKV':0.955449033, 'O':0.427947233, 'FFN1':1.4071156, 'FFN2':2.219334933},
        'vit-huge' : {'QKV':1.504691833, 'O':0.615488133, 'FFN1':1.8119571, 'FFN2':3.262872067}
    }

    for experiment_name in cpu_baseline.keys():
        lut_nn_performance = {}
        if not plot_only:
            os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/performance-analysis/configs/%s/v4-ct16.yaml --need_compile --need_breakdown --breakdown_type=2 > ../asplos24-ae/performance-analysis/log/%s/layerwise_breakdown.txt 2>&1' % (experiment_name, experiment_name))
            print("model %s 1 of 1 jobs finished" % experiment_name)
        with open('./log/%s/layerwise_breakdown.txt' % experiment_name, 'r') as f:
            lines = f.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if 'qkv projection' in line:
                    next_line = lines[i+1]
                    line = line.split(' ')
                    next_line = next_line.split(' ')
                    lut_nn_performance['QKV'] = float(line[3].split(',')[0]) / (float(next_line[1].split(',')[0]) + 1)
                    lut_nn_performance['O'] = float(line[7].split(',')[0]) / (float(next_line[1].split(',')[0]) + 1)
                    lut_nn_performance['FFN1'] = float(line[10].split(',')[0]) / (float(next_line[1].split(',')[0]) + 1)
                    lut_nn_performance['FFN2'] = float(line[13].split(',')[0]) / (float(next_line[1].split(',')[0]) + 1)
        
        print("model %s starts plotting" % experiment_name)
        layer_name = list(lut_nn_performance.keys())
        speedup = [cpu_baseline[experiment_name][name] / lut_nn_performance[name] for name in layer_name]
        fig, ax = plt.subplots()
        ax.bar(layer_name, speedup)
        ax.set_ylim(0, 3)
        ax.set_yticks([0, 1, 2, 3])
        plt.savefig('./plot/layerwise-breakdown-%s.png' % experiment_name)
        plt.close()
        print("model %s finishes plotting" % experiment_name)


def main():
    os.system('mkdir -p log/')
    os.system('mkdir -p log/bert-base')
    os.system('mkdir -p log/bert-large')
    os.system('mkdir -p log/vit-huge')
    os.system('mkdir -p plot/')

    args = parse_args()

    if not args.single_point:
        print("start running latency breakdown")
        run_latency_breakdown(args.plot_only)
        print("finish running latency breakdown")
        print("----------")
        print("start running layer-wise comparison")
        run_layer_wise_comparison(args.plot_only)
        print("finish running comparison")
    else:
        print("start running single point")
        run_single_point(args)
        print("finish running single point")


if __name__ == '__main__':
    main()
