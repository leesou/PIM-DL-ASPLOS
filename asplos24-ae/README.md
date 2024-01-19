# Instructions for ASPLOS-2024 Artifact Evaluation

## 1. Getting Started

Our implementation is based on the UPMEM PIM-DIMM product. Considering the AE reviewers might not have PIM-DIMMs on-hand, we provide AE reviewers the access to our PIM server and the experiments can be conducted on our machine. Please check the ```AE Appendix``` in our submission for details to log in our server.

After logging in, we provide a pre-built project in the ```~/PIM-DL-ASPLOS``` folder. Reviewers just need to run the following commands to set up the evaluation environment:

``` bash
cd PIM-DL-ASPLOS
cd asplos24-ae
conda activate pim-dl-ae
```

The following instructions introduces the hardware requirements and the building process for artifact evaluation. AE reviewers can skip to [2. Run Experiments](#2-run-experiments-3-hours) to conduct experiments.

#### Hardware Requirements

Our system configurations are listed as follows:

> * OS: Ubuntu 18.04
> * CPU: Intel Xeon 4210 (dual-socket)
> * Memory: 128GB DDR4 memory
> * UPMEM PIM-DIMM: 8 DIMMs, 64GB DDR4 memory, DPU runing at 350MHz
> * UPMEM SDK: Version 2021.3.0

#### Build Stuff (< 10 minutes)

Please run the following commands to set up the Python environment:

```bash
conda create -n pim-dl-ae -y
conda activate pim-dl-ae
conda install python -y
pip install numpy matplotlib PyYAML
```

Then, please run the following bash script to compile the inference engine. This command assumes your bash is in the ```asplos24-ae``` directory:

```bash
# in the asplos24-ae folder
bash compile-inference-engine.sh
```

## 2. Run Experiments (~3 hours)

**For AE reviewers**: Note that due to the resource limit, **_only one experiment job can be executed at a time_**. Before running the following scripts, please make sure the server is free for use (e.g., using ```htop``` to check the running process or ```dpu-diag``` to check the occupacy of PIM-DIMMs). We also provide file lock mechanism to inform whether the resources are occupied. If you get the following outputs when running these scripts, please wait for the machine being free.

``` bash
Lock file already exists. Exiting...
```

#### Main Results All-in-one (~2.5 hours)

We provide an all-in-one script to run all experiments for artifact evaluation:

``` bash
# in the asplos24-ae folder
bash run-all.sh
```

After finishing this script, all the plotted results are saved in the ```results``` folder. You can check [3. Validate Results](#3-validate-results) for result validation. 

The experiments will take about 2.5 hours to finish. You can run the experiments in a ```tmux``` environment (type ```tmux new -s <session-name>``` to create and enter the shell, and run ```tmux attach -t <session-name>``` to re-enter the shell. The processes running in a tmux shell will not be terminated even if you close the remote ssh).

The following description introduces how to run each experiment separately, which can be skipped if you have run the all-in-one script.

#### End-to-end Performance (~10 minutes)

If you have not run the all-in-one script, you can run the following script to get the end-to-end results (throughput and energy efficiency):

```bash
# in the asplos24-ae folder
bash run-end-to-end.sh
```

After finishing this script, all the plotted results are saved in the ```results/end-to-end``` folder. You can check [3. Validate Results](#3-validate-results) for result validation. 

Considering the baselines are run on different server, and the energy profiling requires ```sudo``` authority, we use the previously profiled latency & power data for the ease of artifact evaluation, which can also avoid performance number mismatching caused by hardware differences and other unstable factors. We provide baseline codes in the ```end-to-end/baselines``` folder.

#### Performance Analysis (~15 minutes)

If you have not run the all-in-one script, you can run the following script to get the performance analysis results (latency breakdown and layer-wise performance comparison):

```bash
# in the asplos24-ae folder
bash run-performance-analysis.sh
```

After finishing this script, all the plotted results are saved in the ```results/performance-analysis``` folder. You can check [3. Validate Results](#3-validate-results) for result validation.

#### Sensitivity Analysis (~75 minutes)

If you have not run the all-in-one script, you can run the following script to get the sensitivity analysis results (changing sub-vector length, centroid number, batch size, and hidden dim):

```bash
# in the asplos24-ae folder
bash run-sensitivity-analysis.sh
```

After finishing this script, all the plotted results are saved in the ```results/sensitivity-analysis``` folder. You can check [3. Validate Results](#3-validate-results) for result validation.

#### Mapping Space Visualization (~50 minutes)

If you have not run the all-in-one script, you can run the following script to get the mapping space visualization results (coars-grain LUT load, fine-grain LUT load, static LUT load, global optimal):

```bash
# in the asplos24-ae folder
bash run-search-space-illustration.sh
```

After finishing this script, all the plotted results are saved in the ```results/search-space-illustration``` folder. You can check [3. Validate Results](#3-validate-results) for result validation.

#### Known Issues

In rare cases, test points might meet hardware error caused by UPMEM PIM-DIMMs. If you find the results vary greatly or some points cannot report results, you can re-run the corresponding experiment separately or use the following commands to run single experiment point or plot the figure without running experiments:

For end-to-end performance:

``` bash
# in the asplos24-ae/end-to-end folder
python run_experiments.py 
       [--plot_only] [--single_point] 
       [--model_type] [--vec_len] [--centroid_num]
```

- ```--plot_only```: If this option is on, the script will only plot figures using pre-dumped logs. Please open this option only when **_all experiments have finished_**.

- ```--single_point```: If this option is on, the script will only run the experiment point specified as below. The following options are valid only when this option is on.

- ```--model_type```: The model you want to run for single-point experiment. You can specify ```bert-base```, ```bert-large```, or ```vit-huge``` for this experiment.

- ```--vec_len```: The sub-vector length you want to run for single-point experiment. You can specify ```2```, or ```4``` for this experiment.

- ```--centroid_num```: The centroid number you want to run for single-point experiment. You can specify ```16``` for this experiment.

For performance analysis:

``` bash
# in the asplos24-ae/performance-analysis folder
python run_experiments.py
       [--plot_only] [--single_point] 
       [--experiment_type] [--model_type]
```

- ```--plot_only```: If this option is on, the script will only plot figures using pre-dumped logs. Please open this option only when **_all experiments have finished_**.

- ```--single_point```: If this option is on, the script will only run the experiment point specified as below. The following options are valid only when this option is on.

- ```--experiment_type```: The experiment type you want to run. You can specify ```latency``` for points in latency breakdown and ```layer-wise``` for points in layer-wise performance comparison.

- ```--model_type```: The model you want to run for single-point experiment. You can specify ```bert-base```, ```bert-large```, or ```vit-huge``` for this experiment.

For sensitivity analysis:

``` bash
# in the asplos24-ae/sensitivity-analysis folder
python run_experiments.py
       [--plot_only] [--single_point] 
       [--experiment_type] [--model_type] [--exploration_value]
```

- ```--plot_only```: If this option is on, the script will only plot figures using pre-dumped logs. Please open this option only when **_all experiments have finished_**.

- ```--single_point```: If this option is on, the script will only run the experiment point specified as below. The following options are valid only when this option is on.

- ```--experiment_type```: The experiment type you want to run. You can specify ```sub-vector``` for points in sub-vector length exploration, ```centroid``` for points in centroid number exploration, ```batch-size``` for points in batch size exploration, and ```hidden-dim``` for points in hidden-dim number exploration.

- ```--model_type```: The model you want to run for single-point experiment. You can specify ```bert-base```, ```bert-large```, or ```vit-huge``` for this experiment. This option is not used for hidden-dim number exploration.

- ```--exploration_value```: The value you set for changeable parameters. You can specify one of ```{2,4,6,8,16,32}``` for sub-vector length exploration, one of ```{8,16,32,64,128}``` for centroid number exploration, one of ```{8,16,32,64,128}``` for batch size exploration, and one of ```{1024,2048,2560,4096,5120}``` for hidden-dim number exploration.

For mapping space visualization:

``` bash
# in the asplos24-ae/search-space-illustration folder
python run_experiments.py
       [--plot_only] [--single_point] 
       [--experiment_type] [--exploration_value]
```

- ```--plot_only```: If this option is on, the script will only plot figures using pre-dumped logs. Please open this option only when **_all experiments have finished_**.

- ```--single_point```: If this option is on, the script will only run the experiment point specified as below. The following options are valid only when this option is on.

- ```--experiment_type```: The experiment type you want to run. You can specify ```coarse-grain``` for points in coarse-grain LUT load visualization, ```fine-grain``` for points in fine-grain LUT load visualization, ```static``` for points in static LUT load visualization, and ```global``` for points in hidden-dim number exploration.

- ```--exploration_value```: The value you set for changeable parameters. It should be a string of integers where all integers are separated by ```-``` delimiter. For coarse-grain LUT load, the format is ```{Load-tile CB}-{Load-tile F}-{M-tile N}-{M-tile CB}-{M-tile F}```. For fine-grain LUT load, the format is ```{Load-tile F}-{M-tile N}-{M-tile CB}-{M-tile F}```. For static LUT load, the format is ```{M-tile F}-{M-tile N}-{M-tile CB}```. For global, the format is ```{Loop Order}-{S-tile N}-{S-tile F}```, where the loop order follows: ```0->nfc```, ```1->ncf```, ```2->fnc```, ```3->fcn```, ```4->cnf```, ```5->cfn```. Please check the axises in our paper for detailed value range.

## 3. Validate Results

#### End-to-end Performance

The sub-figures for each model are stored in the ```results/end-to-end/{model-name}``` folder. In each folder, the ```throughput.png``` corresponds to the sub-figure of each model in throughput comparison (Fig. 9-(a)), and the ```energy.png``` corresponds to the sub-figure of each model in energy efficiency comparison (Fig. 9-(b)).

#### Performance Analysis

All (sub-)figures for performance analysis are stored in the ```results/performance-analysis/``` folder. The ```inference-latency-breakdown.png``` corresponds to the figure of inference latency breakdown (Fig. 10-(a)). The sub-figure of each model in layer-wise comparison (Fig. 10-(b)) is plotted in ```layer-wise-breakdown-{model-name}.png```.

#### Sensitivity Analysis

All (sub-)figures for sensitivity analysis are stored in the ```results/sensitivity-analysis``` folder. The ```sub-vector-length.png``` correponds to sub-vector length exploration (Fig. 11-(a)). The ```centroid-number.png``` correponds to centroid number exploration (Fig. 11-(b)). The ```batch-size.png``` correponds to batch-size exploration (Fig. 11-(c)). The ```hidden-dim.png``` correponds to hidden dim exploration (Fig. 11-(d)). 

#### Mapping Space Illustration

All (sub-)figures for mapping space visualization are stored in the ```results/search-space-illustration``` folder. The ```coarse_grain.png``` corresponds to the coarse-grain LUT load space illustration (Fig. 12-(a)). The ```fine_grain.png``` corresponds to the fine-grain LUT load space illustration (Fig. 12-(b)). The ```static.png``` corresponds to the static LUT load space illustration (Fig. 12-(c)). The ```global.png``` corresponds to the global optimal space illustration (Fig. 12-(d)).

It is possible that the reproduced results are slightly different from that in the paper due to the runtime perturbation. For reference, we also provide pre-run results in the ```results-ref``` folder, which has the same organization as the ```results``` folder.