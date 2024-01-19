import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


cpu_server_latency = {
    'bert-base' : {'fp32':6.64171, 'int8':4.31126},
    'bert-large' : {'fp32': 12.3212, 'int8':6.20743},
    'vit-huge' : {'fp32': 14.5531, 'int8':7.6552}
}

pim_gemm_latency = {
    'bert-base' : 38.4659089,
    'bert-large' : 68.03529887,
    'vit-huge' : 105.8759874
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_only', action='store_true', default=False)
    parser.add_argument('--single_point', action='store_true', default=False)
    parser.add_argument('--model_type', type=str, default='bert-base')
    parser.add_argument('--vec_len', type=int, default=2)
    parser.add_argument('--centroid_num', type=int, default=16)

    args = parser.parse_args()
    return args

def run_single_point(args):
    os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/end-to-end/configs/%s/v%d-ct%d.yaml --need_compile > ../asplos24-ae/end-to-end/log/%s/v%d-ct%d.txt 2>&1' 
              % (args.model_type, args.vec_len, args.centroid_num, args.model_type, args.vec_len, args.centroid_num))

def run_single_experiment(experiment_name, plot_only):
    if not plot_only:
        os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/end-to-end/configs/%s/v2-ct16.yaml --need_compile > ../asplos24-ae/end-to-end/log/%s/v2-ct16.txt 2>&1' % (experiment_name, experiment_name))
        print("model %s, 1 of 2 jobs finished" % experiment_name)
        os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/end-to-end/configs/%s/v4-ct16.yaml --need_compile > ../asplos24-ae/end-to-end/log/%s/v4-ct16.txt 2>&1' % (experiment_name, experiment_name))
        print("model %s, 2 of 2 jobs finished" % experiment_name)

    print("model %s starts plotting" % experiment_name)
    latency = {'cpu-fp32':cpu_server_latency[experiment_name]['fp32'], 'cpu-int8':cpu_server_latency[experiment_name]['int8'],}
    for config_name in ['v2-ct16', 'v4-ct16']:
        with open('./log/%s/%s.txt' % (experiment_name, config_name), 'r') as f:
            for line in f.readlines():
                if 'inference time' in line:
                    latency[config_name] = float(line.split(' ')[-1])
    latency['pim'] = pim_gemm_latency[experiment_name]
    real_latency = list(latency.values())
    norm_latency = [latency['cpu-fp32']/val for val in real_latency]
    labels = list(latency.keys())
    bar_yticks = [0, 1.5, 3, 4.5, 6]
    scatter_yticks = [0, 5, 10, 15, 20]

    fig, ax1 = plt.subplots()
    ax1.bar(labels, norm_latency)
    ax1.set_yticks(bar_yticks)
    ax2 = ax1.twinx()
    ax2.plot(labels, real_latency, marker='o', color='orange')
    ax2.set_yticks(scatter_yticks)
    ax2.set_ylim(0, 20)
    ax2.text(labels[-1], 20, str(real_latency[-1]), ha='center', va='top')
    plt.savefig('./plot/%s/throughput.png' % experiment_name)
    plt.close()

    power = [596.93, 596.93, 419.19, 419.19, 366.35]
    energy = [power[i]*real_latency[i] for i in range(len(power))]
    norm_energy = [energy[0]/energy[i] for i in range(len(energy))]

    fig, ax = plt.subplots()
    ax.bar(labels, norm_energy)
    ax.set_yticks(bar_yticks)
    plt.savefig('./plot/%s/energy.png' % experiment_name)
    plt.close()
    print("model %s finishes plotting" % experiment_name)


def main():
    os.system('mkdir -p log/')
    os.system('mkdir -p log/bert-base')
    os.system('mkdir -p log/bert-large')
    os.system('mkdir -p log/vit-huge')
    os.system('mkdir -p plot/')
    os.system('mkdir -p plot/bert-base')
    os.system('mkdir -p plot/bert-large')
    os.system('mkdir -p plot/vit-huge')

    args = parse_args()

    if not args.single_point:
        for experiment_name in ['bert-base', 'bert-large', 'vit-huge']:
            print("start running %s's experiments" % experiment_name)
            run_single_experiment(experiment_name, args.plot_only)
    else:
        print("start running single point")
        run_single_point(args)
        print("finish running single point")


if __name__ == '__main__':
    main()
