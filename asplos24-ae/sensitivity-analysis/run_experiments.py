import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_only', action='store_true', default=False)
    parser.add_argument('--single_point', action='store_true', default=False)
    parser.add_argument('--experiment_type', type=str, default='sub-vector')
    parser.add_argument('--model_type', type=str, default='bert-base')
    parser.add_argument('--exploration_value', type=int, default=0)

    args = parser.parse_args()
    return args

def run_single_point(args):
    if args.experiment_type == 'sub-vector':
        os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/sensitivity-analysis/configs/%s/sub-vector-length/v%d.yaml --need_compile > ../asplos24-ae/sensitivity-analysis/log/%s/sub-vector-length/v%d.txt 2>&1' % (args.model_type, args.exploration_value, args.model_type, args.exploration_value))
    elif args.experiment_type == 'centroid':
        os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/sensitivity-analysis/configs/%s/centroid-number/ct%d.yaml --need_compile > ../asplos24-ae/sensitivity-analysis/log/%s/centroid-number/ct%d.txt 2>&1' % (args.model_type, args.exploration_value, args.model_type, args.exploration_value))
    elif args.experiment_type == 'batch-size':
        os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/sensitivity-analysis/configs/%s/batch-size/bs%d.yaml --need_compile > ../asplos24-ae/sensitivity-analysis/log/%s/batch-size/bs%d.txt 2>&1' % (args.model_type, args.exploration_value, args.model_type, args.exploration_value))
    elif args.experiment_type == 'hidden-dim':
        os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/sensitivity-analysis/configs/hidden-dim/dim%d.yaml --need_compile > ../asplos24-ae/sensitivity-analysis/log/hidden-dim/dim%d.txt 2>&1' % (args.exploration_value, args.exploration_value))

def sub_vec_len_exploration(plot_only):
    cpu_baseline = {
        'bert-base' : 4.31126,
        'bert-large' : 6.20743,
        'vit-huge' : 7.59894
    }
    experiment_names = list(cpu_baseline.keys())
    results = {}
    finished_jod_count = 0
    for experiment_name in experiment_names:
        results[experiment_name] = {}
        for v in [2, 4, 8, 16, 32]:
            if not plot_only:
                os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/sensitivity-analysis/configs/%s/sub-vector-length/v%d.yaml --need_compile > ../asplos24-ae/sensitivity-analysis/log/%s/sub-vector-length/v%d.txt 2>&1' % (experiment_name, v, experiment_name, v))
            with open('./log/%s/sub-vector-length/v%d.txt' % (experiment_name, v)) as f:
                for line in f.readlines():
                    if 'inference time' in line:
                        results[experiment_name][v] = float(line.split(' ')[-1])
            finished_jod_count += 1
            print("sub-vector length exploration, %d of 15 jobs finished" % finished_jod_count)
    
    print("sub-vector length exploration starts plotting")
    x = [1, 2, 3, 4, 5]
    vs = [2, 4, 8, 16, 32]
    speedup = {
        'bert-base' : [cpu_baseline['bert-base'] / results['bert-base'][v] for v in vs],
        'bert-large' : [cpu_baseline['bert-large'] / results['bert-large'][v] for v in vs],
        'vit-huge' : [cpu_baseline['vit-huge'] / results['vit-huge'][v] for v in vs]
    }

    fig, ax = plt.subplots()
    ax.plot(x, speedup['bert-base'], color='blue', marker='o', label='bert-base')
    ax.plot(x, speedup['bert-large'], color='orange', marker='^', label='bert-large')
    ax.plot(x, speedup['vit-huge'], color='grey', marker='D', label='vit-huge')
    ax.set_xticks(x)
    ax.set_xticklabels(vs)
    ax.set_ylim(1, 3)
    ax.set_yticks([1, 1.5, 2, 2.5, 3])
    ax.legend()
    plt.savefig('./plot/sub-vector-length.png')
    plt.close()
    print("sub-vector length exploration finishes plotting")


def centroid_number_exploration(plot_only):
    cpu_baseline = {
        'bert-base' : 4.31126,
        'bert-large' : 6.20743,
        'vit-huge' : 7.59894
    }
    experiment_names = list(cpu_baseline.keys())
    results = {}
    finished_jod_count = 0
    for experiment_name in experiment_names:
        results[experiment_name] = {}
        for ct in [128, 64, 32, 16, 8]:
            if not plot_only:
                os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/sensitivity-analysis/configs/%s/centroid-number/ct%d.yaml --need_compile > ../asplos24-ae/sensitivity-analysis/log/%s/centroid-number/ct%d.txt 2>&1' % (experiment_name, ct, experiment_name, ct))
            with open('./log/%s/centroid-number/ct%d.txt' % (experiment_name, ct)) as f:
                for line in f.readlines():
                    if 'inference time' in line:
                        results[experiment_name][ct] = float(line.split(' ')[-1])
            finished_jod_count += 1
            print("centroid number exploration, %d of 15 jobs finished" % finished_jod_count)
    
    print("centroid number exploration starts plotting")
    x = [1, 2, 3, 4, 5]
    cts = [128, 64, 32, 16, 8]
    speedup = {
        'bert-base' : [cpu_baseline['bert-base'] / results['bert-base'][ct] for ct in cts],
        'bert-large' : [cpu_baseline['bert-large'] / results['bert-large'][ct] for ct in cts],
        'vit-huge' : [cpu_baseline['vit-huge'] / results['vit-huge'][ct] for ct in cts]
    }

    fig, ax = plt.subplots()
    ax.plot(x, speedup['bert-base'], color='blue', marker='o', label='bert-base')
    ax.plot(x, speedup['bert-large'], color='orange', marker='^', label='bert-large')
    ax.plot(x, speedup['vit-huge'], color='grey', marker='D', label='vit-huge')
    ax.set_xticks(x)
    ax.set_xticklabels(cts)
    ax.set_ylim(0.5, 2.5)
    ax.set_yticks([0.5, 1, 1.5, 2, 2.5])
    ax.legend()
    plt.savefig('./plot/centroid-number.png')
    plt.close()
    print("centroid number exploration finishes plotting")


def batch_size_exploration(plot_only):
    cpu_baseline = {
        'bert-base' : {8:0.325862, 16:0.897627, 32:1.73693, 64:4.31126, 128:6.98285},
        'bert-large' : {8:0.537471, 16:1.04923, 32:2.32561, 64:6.20743, 128:11.2442},
        'vit-huge' : {8:0.750662, 16:1.54711, 32:3.04215, 64:7.59894, 128:12.3811}
    }
    experiment_names = list(cpu_baseline.keys())
    results = {}
    finished_jod_count = 0
    for experiment_name in experiment_names:
        results[experiment_name] = {}
        for bs in [8, 16, 32, 64, 128]:
            if not plot_only:
                os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/sensitivity-analysis/configs/%s/batch-size/bs%d.yaml --need_compile > ../asplos24-ae/sensitivity-analysis/log/%s/batch-size/bs%d.txt 2>&1' % (experiment_name, bs, experiment_name, bs))
            with open('./log/%s/batch-size/bs%d.txt' % (experiment_name, bs)) as f:
                for line in f.readlines():
                    if 'inference time' in line:
                        results[experiment_name][bs] = float(line.split(' ')[-1])
            finished_jod_count += 1
            print("batch size exploration, %d of 15 jobs finished" % finished_jod_count)
    
    print("batch size exploration starts plotting")
    x = [1, 2, 3, 4, 5]
    bss = [8, 16, 32, 64, 128]
    speedup = {
        'bert-base' : [cpu_baseline['bert-base'][bs] / results['bert-base'][bs] for bs in bss],
        'bert-large' : [cpu_baseline['bert-large'][bs] / results['bert-large'][bs] for bs in bss],
        'vit-huge' : [cpu_baseline['vit-huge'][bs] / results['vit-huge'][bs] for bs in bss]
    }

    fig, ax = plt.subplots()
    ax.plot(x, speedup['bert-base'], color='blue', marker='o', label='bert-base')
    ax.plot(x, speedup['bert-large'], color='orange', marker='^', label='bert-large')
    ax.plot(x, speedup['vit-huge'], color='grey', marker='D', label='vit-huge')
    ax.set_xticks(x)
    ax.set_xticklabels(bss)
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.legend()
    plt.savefig('./plot/batch-size.png')
    plt.close()
    print("batch size exploration finishes plotting")


def hidden_dim_exploration(plot_only):
    cpu_baseline = {1024:6.20743, 2048:12.1038, 2560:15.8093, 4096:36.6115, 5120:50.9591}

    results = {}
    finished_jod_count = 0
    for dim in [1024, 2048, 2560, 4096, 5120]:
        if not plot_only:
            os.system('cd ../../inference-engine; python run_layer.py --transformer_config_file=../asplos24-ae/sensitivity-analysis/configs/hidden-dim/dim%d.yaml --need_compile > ../asplos24-ae/sensitivity-analysis/log/hidden-dim/dim%d.txt 2>&1' % (dim, dim))
        with open('./log/hidden-dim/dim%d.txt' % (dim)) as f:
            for line in f.readlines():
                if 'inference time' in line:
                    results[dim] = float(line.split(' ')[-1])
        finished_jod_count += 1
        print("hidden dim exploration, %d of 5 jobs finished" % finished_jod_count)
    
    print("hidden dim exploration starts plotting")
    x = [1, 2, 3, 4, 5]
    dims = [1024, 2048, 2560, 4096, 5120]
    speedup = [cpu_baseline[dim] / results[dim] for dim in dims]

    fig, ax = plt.subplots()
    ax.bar(x, speedup)
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylim(0, 4)
    ax.set_yticks([0, 1, 2, 3, 4])
    plt.savefig('./plot/hidden-dim.png')
    plt.close()
    print("hidden dim exploration finishes plotting")


def main():
    os.system('mkdir -p log/')
    os.system('mkdir -p log/bert-base')
    os.system('mkdir -p log/bert-base/batch-size')
    os.system('mkdir -p log/bert-base/centroid-number')
    os.system('mkdir -p log/bert-base/sub-vector-length')
    os.system('mkdir -p log/bert-large')
    os.system('mkdir -p log/bert-large/batch-size')
    os.system('mkdir -p log/bert-large/centroid-number')
    os.system('mkdir -p log/bert-large/sub-vector-length')
    os.system('mkdir -p log/vit-huge')
    os.system('mkdir -p log/vit-huge/batch-size')
    os.system('mkdir -p log/vit-huge/centroid-number')
    os.system('mkdir -p log/vit-huge/sub-vector-length')
    os.system('mkdir -p log/hidden-dim')
    os.system('mkdir -p plot/')

    args = parse_args()

    if not args.single_point:
        print("start running sub-vector length exploration")
        sub_vec_len_exploration(args.plot_only)
        print("finish running sub-vector length exploration")
        print("----------")
        print("start running centroid number exploration")
        centroid_number_exploration(args.plot_only)
        print("finish running centroid number exploration")
        print("----------")
        print("start running batch size exploration")
        batch_size_exploration(args.plot_only)
        print("finish running batch size exploration")
        print("----------")
        print("start running hidden dim exploration")
        hidden_dim_exploration(args.plot_only)
        print("finish running hidden dim exploration")
    else:
        print("start running single point")
        run_single_point(args)
        print("finish running single point")


if __name__ == '__main__':
    main()
