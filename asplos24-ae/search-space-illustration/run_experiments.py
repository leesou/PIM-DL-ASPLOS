import numpy as np
import matplotlib.pyplot as plt
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_only', action='store_true', default=False)
    parser.add_argument('--single_point', action='store_true', default=False)
    parser.add_argument('--experiment_type', type=str, default='sub-vector')
    parser.add_argument('--exploration_value', type=str, default="1-1-1-1-1")

    args = parser.parse_args()
    return args

def run_single_point(args):
    if args.experiment_type == 'coarse-grain':
        illegal_shapes = [
            ((256,4), (32,128,128)), ((256,4), (64,128,32)), ((16,64), (64,128,32)), ((256,4), (64,64,64)), ((128,8), (64,64,64))
        ]
        params = [int(num) for num in args.exploration_value.split('-')]
        if ((params[0], params[1]), (params[2], params[3], params[4])) not in illegal_shapes:
            os.system('cd ../../inference-engine; \
                    python run_amm.py --amm_config_file=../asplos24-ae/search-space-illustration/configs/coarse_grain/%d_%d__%d_%d_%d.yaml \
                    > ../asplos24-ae/search-space-illustration/log/coarse_grain/%d_%d__%d_%d_%d.txt 2>&1' 
                    % (params[0], params[1], params[2], params[3], params[4], params[0], params[1], params[2], params[3], params[4]))
        else:
            print('meet illegal shapes, skip execution')
    elif args.experiment_type == 'fine-grain':
        illegal_shapes = [
            ((64,), (64,128,32)), ((128,), (32,256,64)), ((128,), (64,64,64)), ((128,), (64,128,32))
        ]
        params = [int(num) for num in args.exploration_value.split('-')]
        if ((params[0]), (params[1], params[2], params[3])) not in illegal_shapes:
            os.system('cd ../../inference-engine; \
                       python run_amm.py --amm_config_file=../asplos24-ae/search-space-illustration/configs/fine_grain/%d__%d_%d_%d.yaml \
                       > ../asplos24-ae/search-space-illustration/log/fine_grain/%d__%d_%d_%d.txt 2>&1' 
                       % (params[0], params[1], params[2], params[3], params[0], params[1], params[2], params[3]))
        else:
            print('meet illegal shapes, skip execution')
    elif args.experiment_type == 'static':
        illegal_shapes = [
            ((1,), (16,128)), ((4,), (128,16))
        ]
        params = [int(num) for num in args.exploration_value.split('-')]
        if ((params[0]), (params[1], params[2])) not in illegal_shapes:
            os.system('cd ../../inference-engine; \
                       python run_amm.py --amm_config_file=../asplos24-ae/search-space-illustration/configs/static/%d__%d_%d.yaml \
                       > ../asplos24-ae/search-space-illustration/log/static/%d__%d_%d.txt 2>&1' 
                       % (params[0], params[1], params[2], params[0], params[1], params[2]))
        else:
            print('meet illegal shapes, skip execution')
    elif args.experiment_type == 'global':
        params = [int(num) for num in args.exploration_value.split('-')]
        loop_order = ['nfc', 'ncf', 'fnc', 'fcn', 'cnf', 'cfn']
        os.system('cd ../../inference-engine; \
                   python run_amm.py --amm_config_file=../asplos24-ae/search-space-illustration/configs/global/%s__%d_%d.yaml \
                   > ../asplos24-ae/search-space-illustration/log/global/%s__%d_%d.txt 2>&1' 
                   % (loop_order[params[0]], params[1], params[2], loop_order[params[0]], params[1], params[2]))

def coarse_grain(plot_only):
    baseline_latency = 1.4071156
    shapes = [
        ((256,4), (16,256,256)), ((128,8), (16,256,256)), ((64,16), (16,256,256)), ((32,32), (16,256,256)), ((16,64), (16,256,256)),
        ((256,4), (32,256,64)), ((128,8), (32,256,64)), ((64,16), (32,256,64)), ((32,32), (32,256,64)), ((16,64), (32,256,64)),
        ((256,4), (32,128,128)), ((128,8), (32,128,128)), ((64,16), (32,128,128)), ((32,32), (32,128,128)), ((16,64), (32,128,128)),
        ((256,4), (64,128,32)), ((128,8), (64,128,32)), ((64,16), (64,128,32)), ((32,32), (64,128,32)), ((16,64), (64,128,32)),
        ((256,4), (64,64,64)), ((128,8), (64,64,64)), ((64,16), (64,64,64)), ((32,32), (64,64,64)), ((16,64), (64,64,64))
    ]
    illegal_shapes = [
        ((256,4), (32,128,128)), ((256,4), (64,128,32)), ((16,64), (64,128,32)), ((256,4), (64,64,64)), ((128,8), (64,64,64))
    ]

    x_data = [5 for _ in range(5)] + [4 for _ in range(5)] + [3 for _ in range(4)] + [2 for _ in range(3)] + [1 for _ in range(3)]
    y_data = [i for i in range(1, 6)] + [i for i in range(1, 6)] + [i for i in range(2, 6)] + [i for i in range(2, 5)] + [i for i in range(3, 6)]
    z_data = []
    
    x_line_data = [5 for _ in range(5)] + [4 for _ in range(5)] + [3 for _ in range(5)] + [2 for _ in range(5)] + [1 for _ in range(5)]
    y_line_data = [i for i in range(1, 6)] * 5
    z_line_data = []

    finished_job_count = 0
    for shape in shapes:
        if shape in illegal_shapes:
            z_line_data.append(0)
            finished_job_count += 1
            print("coarse-grain, %d of %d jobs finished" % (finished_job_count, len(shapes)))
            continue
        
        if not plot_only:
            os.system('cd ../../inference-engine; \
                    python run_amm.py --amm_config_file=../asplos24-ae/search-space-illustration/configs/coarse_grain/%d_%d__%d_%d_%d.yaml \
                    > ../asplos24-ae/search-space-illustration/log/coarse_grain/%d_%d__%d_%d_%d.txt 2>&1' 
                    % (shape[0][0], shape[0][1], shape[1][0], shape[1][1], shape[1][2], shape[0][0], shape[0][1], shape[1][0], shape[1][1], shape[1][2]))
        with open('./log/coarse_grain/%d_%d__%d_%d_%d.txt' % (shape[0][0], shape[0][1], shape[1][0], shape[1][1], shape[1][2]), 'r') as f:
            for line in f.readlines():
                if 'AMM time' in line:
                    latency = float(line.split(' ')[-1])
        speedup = baseline_latency / latency
        z_data.append(speedup)
        z_line_data.append(speedup)
        finished_job_count += 1
        print("coarse-grain, %d of %d jobs finished" % (finished_job_count, len(shapes)))

    print("coarse-grain starts plotting")
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim(0.5, 2.5)
    ax.set_zticks([0.5, 1, 1.5, 2, 2.5])
    ax.scatter(x_data, y_data, z_data, c='black', marker='o', s=10)

    for i in range(len(x_line_data) - 1):
        if ((i+1)%5)!=0  and z_line_data[i]!=0 and z_line_data[i+1]!=0:
            ax.plot([x_line_data[i], x_line_data[i + 1]], [y_line_data[i], y_line_data[i + 1]], [z_line_data[i], z_line_data[i + 1]], c='black', alpha=0.8, linewidth=1.2)
        if (i+5)<len(x_line_data) and z_line_data[i]!=0 and z_line_data[i+5]!=0:
            ax.plot([x_line_data[i], x_line_data[i + 5]], [y_line_data[i], y_line_data[i + 5]], [z_line_data[i], z_line_data[i + 5]], c='black', alpha=0.8, linewidth=1.2)
    ax.plot([x_line_data[14], x_line_data[24]], [y_line_data[14], y_line_data[24]], [z_line_data[14], z_line_data[24]], c='black', alpha=0.8, linewidth=1.2)

    x_illegal_points = [1, 2, 3, 1, 2]
    y_illegal_points = [1, 1, 1, 2, 5]
    z_illegal_points = [0.5, 0.5, 0.5, 0.5, 0.5]
    ax.scatter(x_illegal_points, y_illegal_points, z_illegal_points, c='red', marker='x', s=100, label='illegal points')

    ax.scatter(2, 2, z_line_data[16], c='blue', marker='*', s=200, label='best mapping in PIM-DL auto tuner')
    ax.scatter(4, 3, z_line_data[7], c='orange', marker='*', s=200, label='best mapping in real performance')
    ax.scatter(5, 5, z_line_data[4], c='green', marker='^', s=200, label='worst mapping in real performance')

    ax.set_xlabel('M-tile (N, CB, F)', labelpad=35)
    ax.set_ylabel('Load-tile (CB, F)')
    ax.set_zlabel('Norm. Speedup')

    new_x_ticks = [1, 2, 3, 4, 5]
    new_x_labels = ['(64, 64, 64)', '(64, 128, 32)', '(32, 128, 128)', '(32, 256, 64)', '(16, 256, 256)']

    new_y_ticks = [1, 2, 3, 4, 5]
    new_y_labels = ['(256, 4)', '(128, 8)', '(64, 16)', '(32, 32)', '(16, 64)']

    ax.view_init(elev=25, azim=15)
    plt.xticks(new_x_ticks, new_x_labels)
    plt.tick_params(axis='x', labelsize=10, pad=20, rotation=-10, 
                    labeltop=True, labelbottom=False, labelleft=False, labelright=False)
    plt.yticks(new_y_ticks, new_y_labels)
    plt.tick_params(axis='y', labelsize=10, pad=0, rotation=-10)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('./plot/coarse_grain.png')
    print("coarse-grain finishes plotting")


def fine_grain(plot_only):
    baseline_latency = 1.4071156
    shapes = [
        ((8,), (16,256,256)), ((16,), (16,256,256)), ((32,), (16,256,256)), ((64,), (16,256,256)), ((128,), (16,256,256)),
        ((8,), (32,128,128)), ((16,), (32,128,128)), ((32,), (32,128,128)), ((64,), (32,128,128)), ((128,), (32,128,128)),
        ((8,), (32,256,64)), ((16,), (32,256,64)), ((32,), (32,256,64)), ((64,), (32,256,64)), ((128,), (32,256,64)),
        ((8,), (64,64,64)), ((16,), (64,64,64)), ((32,), (64,64,64)), ((64,), (64,64,64)), ((128,), (64,64,64)),
        ((8,), (64,128,32)), ((16,), (64,128,32)), ((32,), (64,128,32)), ((64,), (64,128,32)), ((128,), (64,128,32))
    ]
    illegal_shapes = [
        ((64,), (64,128,32)), ((128,), (32,256,64)), ((128,), (64,64,64)), ((128,), (64,128,32))
    ]

    x_data = [5 for _ in range(5)] + [4 for _ in range(5)] + [3 for _ in range(4)] + [2 for _ in range(4)] + [1 for _ in range(3)]
    y_data = [i for i in range(1, 6)] + [i for i in range(1, 6)] + [i for i in range(1, 5)] + [i for i in range(1, 5)] + [i for i in range(1, 4)]
    z_data = []
    
    x_line_data = [5 for _ in range(5)] + [4 for _ in range(5)] + [3 for _ in range(5)] + [2 for _ in range(5)] + [1 for _ in range(5)]
    y_line_data = [i for i in range(1, 6)] * 5
    z_line_data = []

    finished_job_count = 0
    for shape in shapes:
        if shape in illegal_shapes:
            z_line_data.append(0)
            finished_job_count += 1
            print("fine-grain, %d of %d jobs finished" % (finished_job_count, len(shapes)))
            continue
        
        if not plot_only:
            os.system('cd ../../inference-engine; \
                    python run_amm.py --amm_config_file=../asplos24-ae/search-space-illustration/configs/fine_grain/%d__%d_%d_%d.yaml \
                    > ../asplos24-ae/search-space-illustration/log/fine_grain/%d__%d_%d_%d.txt 2>&1' 
                    % (shape[0][0], shape[1][0], shape[1][1], shape[1][2], shape[0][0], shape[1][0], shape[1][1], shape[1][2]))
        with open('./log/fine_grain/%d__%d_%d_%d.txt' % (shape[0][0], shape[1][0], shape[1][1], shape[1][2]), 'r') as f:
            for line in f.readlines():
                if 'AMM time' in line:
                    latency = float(line.split(' ')[-1])
        speedup = baseline_latency / latency
        z_data.append(speedup)
        z_line_data.append(speedup)
        finished_job_count += 1
        print("fine-grain, %d of %d jobs finished" % (finished_job_count, len(shapes)))

    print("fine-grain starts plotting")
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim(0.5, 2.5)
    ax.set_zticks([0.5, 1, 1.5, 2, 2.5])
    ax.scatter(x_data, y_data, z_data, c='black', marker='o', s=10)

    for i in range(len(x_line_data) - 1):
        if ((i+1)%5)!=0  and z_line_data[i]!=0 and z_line_data[i+1]!=0:
            ax.plot([x_line_data[i], x_line_data[i + 1]], [y_line_data[i], y_line_data[i + 1]], [z_line_data[i], z_line_data[i + 1]], c='black', alpha=0.8, linewidth=1.2)
        if (i+5)<len(x_line_data) and z_line_data[i]!=0 and z_line_data[i+5]!=0:
            ax.plot([x_line_data[i], x_line_data[i + 5]], [y_line_data[i], y_line_data[i + 5]], [z_line_data[i], z_line_data[i + 5]], c='black', alpha=0.8, linewidth=1.2)

    x_illegal_points = [1, 1, 2, 3]
    y_illegal_points = [4, 5, 5, 5]
    z_illegal_points = [0.5, 0.5, 0.5, 0.5]
    ax.scatter(x_illegal_points, y_illegal_points, z_illegal_points, c='red', marker='x', s=100, label='illegal points')

    ax.scatter(5, 1, z_line_data[0], c='blue', marker='*', s=200, label='best mapping in PIM-DL auto tuner')
    ax.scatter(4, 2, z_line_data[6], c='orange', marker='*', s=200, label='best mapping in real performance')
    ax.scatter(4, 4, z_line_data[8], c='green', marker='^', s=200, label='worst mapping in real performance')

    ax.set_xlabel('M-tile (N, CB, F)', labelpad=35)
    ax.set_ylabel('Load-tile (F)')
    ax.set_zlabel('Norm. Speedup')

    new_x_ticks = [1, 2, 3, 4, 5]
    new_x_labels = ['(64, 128, 32)', '(64, 64, 64)', '(32, 256, 64)', '(32, 128, 128)', '(16, 256, 256)']

    new_y_ticks = [1, 2, 3, 4, 5]
    new_y_labels = ['8', '16', '32', '64', '128']

    ax.view_init(elev=25, azim=15)
    plt.xticks(new_x_ticks, new_x_labels)
    plt.tick_params(axis='x', labelsize=10, pad=20, rotation=-10, 
                    labeltop=True, labelbottom=False, labelleft=False, labelright=False)
    plt.yticks(new_y_ticks, new_y_labels)
    plt.tick_params(axis='y', labelsize=10, pad=0, rotation=-10)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('./plot/fine_grain.png')
    print("fine-grain finishes plotting")


def static(plot_only):
    baseline_latency = 1.4071156
    shapes = [
        ((1,), (16,128)), ((2,), (16,128)), ((4,), (16,128)), ((8,), (16,128)),
        ((1,), (32,64)), ((2,), (32,64)), ((4,), (32,64)), ((8,), (32,64)),
        ((1,), (64,32)), ((2,), (64,32)), ((4,), (64,32)), ((8,), (64,32)),
        ((1,), (128,16)), ((2,), (128,16)), ((4,), (128,16)), ((8,), (128,16)),
        ((1,), (256,8)), ((2,), (256,8)), ((4,), (256,8)), ((8,), (256,8))
    ]
    illegal_shapes = [
        ((1,), (16,128)), ((4,), (128,16))
    ]

    x_data = [5 for _ in range(3)] + [4 for _ in range(4)] + [3 for _ in range(4)] + [2 for _ in range(3)] + [1 for _ in range(4)]
    y_data = [i for i in range(2, 5)] + [i for i in range(1, 5)] * 2 + [1, 2, 4] + [i for i in range(1, 5)]
    z_data = []

    x_line_data = [5 for _ in range(4)] + [4 for _ in range(4)] + [3 for _ in range(4)] + [2 for _ in range(4)] + [1 for _ in range(4)]
    y_line_data = [i for i in range(1, 5)] * 5
    z_line_data = []

    finished_job_count = 0
    for shape in shapes:
        if shape in illegal_shapes:
            z_line_data.append(0)
            finished_job_count += 1
            print("static, %d of %d jobs finished" % (finished_job_count, len(shapes)))
            continue

        if not plot_only:
            os.system('cd ../../inference-engine; \
                    python run_amm.py --amm_config_file=../asplos24-ae/search-space-illustration/configs/static/%d__%d_%d.yaml \
                    > ../asplos24-ae/search-space-illustration/log/static/%d__%d_%d.txt 2>&1' 
                    % (shape[0][0], shape[1][0], shape[1][1], shape[0][0], shape[1][0], shape[1][1]))
        with open('./log/static/%d__%d_%d.txt' % (shape[0][0], shape[1][0], shape[1][1]), 'r') as f:
            for line in f.readlines():
                if 'AMM time' in line:
                    latency = float(line.split(' ')[-1])
        speedup = baseline_latency / latency
        z_data.append(speedup)
        z_line_data.append(speedup)
        finished_job_count += 1
        print("static, %d of %d jobs finished" % (finished_job_count, len(shapes)))

    print("static starts plotting")
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim(0.5, 2.5)
    ax.set_zticks([0.5, 1, 1.5, 2, 2.5])
    ax.scatter(x_data, y_data, z_data, c='black', marker='o', s=10)

    for i in range(len(x_line_data) - 1):
        if ((i+1)%4)!=0  and z_line_data[i]!=0 and z_line_data[i+1]!=0:
            ax.plot([x_line_data[i], x_line_data[i + 1]], [y_line_data[i], y_line_data[i + 1]], [z_line_data[i], z_line_data[i + 1]], c='black', alpha=0.8, linewidth=1.2)
        if (i+4)<len(x_line_data) and z_line_data[i]!=0 and z_line_data[i+4]!=0:
            ax.plot([x_line_data[i], x_line_data[i + 4]], [y_line_data[i], y_line_data[i + 4]], [z_line_data[i], z_line_data[i + 4]], c='black', alpha=0.8, linewidth=1.2)
    ax.plot([x_line_data[13], x_line_data[15]], [y_line_data[13], y_line_data[15]], [z_line_data[13], z_line_data[15]], c='black', alpha=0.8, linewidth=1.2)

    x_illegal_points = [5, 2]
    y_illegal_points = [1, 3]
    z_illegal_points = [0.5, 0.5]
    ax.scatter(x_illegal_points, y_illegal_points, z_illegal_points, c='red', marker='x', s=100, label='illegal points')

    ax.scatter(1, 4, z_line_data[19], c='blue', marker='*', s=200, label='best mapping in PIM-DL auto tuner')
    ax.scatter(5, 4, z_line_data[3], c='orange', marker='*', s=200, label='best mapping in real performance')
    ax.scatter(4, 1, z_line_data[4], c='green', marker='^', s=200, label='worst mapping in real performance')

    ax.set_xlabel('M-tile (N, CB)', labelpad=35)
    ax.set_ylabel('M-tile (F)')
    ax.set_zlabel('Norm. Speedup')

    new_x_ticks = [1, 2, 3, 4, 5]
    new_x_labels = ['(256, 8)', '(128, 16)', '(64, 32)', '(32, 64)', '(16, 128)']

    new_y_ticks = [1, 2, 3, 4]
    new_y_labels = ['1', '2', '4', '8']

    ax.view_init(elev=25, azim=15)
    plt.xticks(new_x_ticks, new_x_labels)
    plt.tick_params(axis='x', labelsize=10, pad=20, rotation=-10, 
                    labeltop=True, labelbottom=False, labelleft=False, labelright=False)
    plt.yticks(new_y_ticks, new_y_labels)
    plt.tick_params(axis='y', labelsize=10, pad=0, rotation=-10)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('./plot/static.png')
    print("static finishes plotting")


def global_optimal(plot_only):
    baseline_latency = 1.4071156
    shapes = [
        ('nfc', (32,4096)), ('ncf', (32,4096)), ('fnc', (32,4096)), ('fcn', (32,4096)), ('cnf', (32,4096)), ('cfn', (32,4096)),
        ('nfc', (128,1024)), ('ncf', (128,1024)), ('fnc', (128,1024)), ('fcn', (128,1024)), ('cnf', (128,1024)), ('cfn', (128,1024)),
        ('nfc', (512,256)), ('ncf', (512,256)), ('fnc', (512,256)), ('fcn', (512,256)), ('cnf', (512,256)), ('cfn', (512,256)),
        ('nfc', (2048,64)), ('ncf', (2048,64)), ('fnc', (2048,64)), ('fcn', (2048,64)), ('cnf', (2048,64)), ('cfn', (2048,64)),
        ('nfc', (8192,16)), ('ncf', (8192,16)), ('fnc', (8192,16)), ('fcn', (8192,16)), ('cnf', (8192,16)), ('cfn', (8192,16))
    ]

    y_data = [5 for _ in range(6)] + [4 for _ in range(6)] + [3 for _ in range(6)] + [2 for _ in range(6)] + [1 for _ in range(6)]
    x_data = [i for i in range(1, 7)] * 5
    z_data = []
    
    y_line_data = [5 for _ in range(6)] + [4 for _ in range(6)] + [3 for _ in range(6)] + [2 for _ in range(6)] + [1 for _ in range(6)]
    x_line_data = [i for i in range(1, 7)] * 5
    z_line_data = []

    finished_job_count = 0
    for shape in shapes:
        if not plot_only:
            os.system('cd ../../inference-engine; \
                    python run_amm.py --amm_config_file=../asplos24-ae/search-space-illustration/configs/global/%s__%d_%d.yaml \
                    > ../asplos24-ae/search-space-illustration/log/global/%s__%d_%d.txt 2>&1' 
                    % (shape[0], shape[1][0], shape[1][1], shape[0], shape[1][0], shape[1][1]))
        with open('./log/global/%s__%d_%d.txt' % (shape[0], shape[1][0], shape[1][1]), 'r') as f:
            for line in f.readlines():
                if 'AMM time' in line:
                    latency = float(line.split(' ')[-1])
        speedup = baseline_latency / latency
        z_data.append(speedup)
        z_line_data.append(speedup)
        finished_job_count += 1
        print("global optimal, %d of %d jobs finished" % (finished_job_count, len(shapes)))

    print("global optimal starts plotting")
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(projection='3d')
    ax.set_zlim(0.5, 2.5)
    ax.set_zticks([0.5, 1, 1.5, 2, 2.5])
    ax.scatter(x_data, y_data, z_data, c='black', marker='o', s=10)

    for i in range(len(x_line_data) - 1):
        if ((i+1)%6)!=0  and z_line_data[i]!=0 and z_line_data[i+1]!=0:
            ax.plot([x_line_data[i], x_line_data[i + 1]], [y_line_data[i], y_line_data[i + 1]], [z_line_data[i], z_line_data[i + 1]], c='black', alpha=0.8, linewidth=1.2)
        if (i+6)<len(x_line_data) and z_line_data[i]!=0 and z_line_data[i+6]!=0:
            ax.plot([x_line_data[i], x_line_data[i + 6]], [y_line_data[i], y_line_data[i + 6]], [z_line_data[i], z_line_data[i + 6]], c='black', alpha=0.8, linewidth=1.2)

    ax.scatter(1, 3, z_line_data[12], c='blue', marker='*', s=200, label='best mapping in PIM-DL auto tuner')
    ax.scatter(3, 3, z_line_data[14], c='orange', marker='*', s=200, label='best mapping in real performance')
    ax.scatter(1, 5, z_line_data[0], c='green', marker='^', s=200, label='worst mapping in real performance')

    ax.set_xlabel('Loop Order', labelpad=5)
    ax.set_ylabel('S-tile (N, F)', labelpad=10)
    ax.set_zlabel('Norm. Speedup')

    new_x_ticks = [1, 2, 3, 4, 5, 6]
    new_x_labels = ['nfc', 'ncf', 'fnc', 'fcn', 'cnf', 'cfn']

    new_y_ticks = [1, 2, 3, 4, 5]
    new_y_labels = ['(8192, 16)', '(2048, 64)', '(512, 256)', '(128, 1024)', '(32, 4096)']

    ax.view_init(elev=25, azim=15)
    plt.xticks(new_x_ticks, new_x_labels)
    plt.tick_params(axis='x', labelsize=10, pad=5, rotation=-10, 
                    labeltop=True, labelbottom=False, labelleft=False, labelright=False)
    plt.yticks(new_y_ticks, new_y_labels)
    plt.tick_params(axis='y', labelsize=10, pad=0, rotation=10)
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('./plot/global.png')
    print("global optimal finishes plotting")


def main():
    os.system('mkdir -p log/')
    os.system('mkdir -p log/coarse_grain/')
    os.system('mkdir -p log/fine_grain/')
    os.system('mkdir -p log/static/')
    os.system('mkdir -p log/global/')
    os.system('mkdir -p plot/')

    args = parse_args()

    if not args.single_point:
        print("start running coarse-grain")
        coarse_grain(args.plot_only)
        print("finish running coarse-grain")
        print("----------")
        print("start running fine-grain")
        fine_grain(args.plot_only)
        print("finish running fine-grain")
        print("----------")
        print("start running static")
        static(args.plot_only)
        print("finish running static")
        print("----------")
        print("start running global optimal")
        global_optimal(args.plot_only)
        print("finish running global optimal")
    else:
        print("start running single point")
        run_single_point(args)
        print("finish running single point")


if __name__ == '__main__':
    main()
