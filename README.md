# Artifact of PIM-DL (ASPLOS '24)

PIM-DL is a deep-learning framework built on commodity DRAM-PIM products using LUT-based neural networks. This prototype is based on the [UPMEM PIM-DIMM product](https://www.upmem.com/).

## Overview

[1. Instructions for Artifact Evaluation](#1-instructions-for-artifact-evaluation)  
[2. Installation](#2-installation)  
[3. Model Calibration](#3-model-calibration)    
[4. Auto-Tuner](#4-auto-tuner)  
[5. Inference Engine](#5-inference-engine)

## 1. Instructions for Artifact Evaluation

We have provided many click-to-run scripts in the `asplos24-ae` folder for conducting the majority of experiments mentioned in the paper, including the end-to-end performance (Fig. 10), performance analysis (Fig. 11), sensitivity analysis (Fig. 12), mapping space visualization (Fig. 13). Please refer to the [README.md](./asplos24-ae/README.md) in the `asplos24-ae` folder for detailed instructions. 

The following sections will introduce how to reuse this repository beyond the artifact evaluation.

## 2. Installation

If you want to conduct model calibration, please run the following commands to install the packages required by the Python scripts:

``` bash
conda create -n pim-dl -y
conda activate pim-dl
conda install python=3.11 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install accelerate transformers dataset gitpython tensorboardX datasets tensorboard scikit-learn evaluate safetensors numpy matplotlib
```

To conduct model inference on UPMEM PIM-DIMMs, please first follow [UPMEM's SDK manual](https://sdk.upmem.com/) to set up the server and install the toolchain. Then, please run the following commands to clone the submodules:

``` bash
git submodule update --init --recursive
```

Then, please install the following Python packages:

``` bash
pip install PyYAML
```

Finally, please run the following commands to compile the auto-tuner and the inference engine:

``` bash
cd inference-engine
mkdir -p build
cd build
cmake ..
make
```

## 3. Model Calibration

The source codes for model calibration are in the `model-calibration` folder, which can convert the pre-trained transformer models to LUT-NN-based models. Before running conversion code, please first install the LUT-NN module using the following commands:

``` bash
cd model-calibration
cd lutneuro
pip install -e .
```

Currently, we have evaluated LUT-NN conversion on Bert models for NLP tasks and ViT models for CV tasks, but the scripts can be used to explore other transformer-based models or other datasets (tasks). Here we provide the instructions on Bert/ViT models.

#### Converting Bert Models

To convert Bert models into LUT-NN, please firstly run the following commands to pre-process the bookcorpus dataset:

``` bash
cd model-calibration
python bookcorpus_generator.py [--store_tokenized_datasets]
```

- ```--store_tokenized_datasets TOKENIZED_DIR```: Set the output directory to save tokenized datasets. The default directory is ```./bert_bookcorpus.token.128/```.

Then, please run these commands to convert the model on the bookcorpus dataset:

``` bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # adjust according to the GPUs you have
accelerate launch examples/run_lutrize_mlm_no_trainer.py 
           [--model_type] [--dataset_name] [--dataset_config_name] [--model_name_or_path]
           [--per_device_train_batch_size] [--per_device_eval_batch_size] 
           [--log_steps] [--with_tracking] [--report_to] [--output_dir]
           [--learning_rate] [--reconstruct_rate] [--weight_transpose]
           [--max_seq_length] [--num_train_epochs] [--checkpointing_steps]
           [--centroid_requires_grad] [--weight_requires_grad]
           [--vec_len] [--ncentroid] [--nsharecodebook] [--distance_p]
           [--load_tokenized_datasets]
```
- ```--model_type MODEL_TYPE```: Model type to use. Currently we use ```bert``` as the model type.

- ```--dataset_name DATASET_NAME```: The name of the dataset to use. Currently we use ```bookcorpus``` dataset.

- ```--dataset_config_name CONFIG_NAME```: The configuration name of the dataset to use. The bookcorpus dataset is a ```plain_text``` dataset.

- ```--model_name_or_path MODEL```: Path to bert model from huggingface. Currently we use ```bert-base-uncased``` or ```bert-large-uncased``` models.

- ```--per_device_train_batch_size TRAIN_BS```: Batch size (per device) for the training dataloader. The batch size is ```8``` by default.

- ```--per_device_eval_batch_size EVAL_BS```: Batch size (per device) for the evaluation dataloader. The batch size is ```8``` by default.

- ```--log_steps STEP```: Logging interval. The interval is ```500``` by default.

- ```--with_tracking```: If open this option, the logging trackers will be enabled.

- ```--report_to INTEGRATION```: The integration to report the results and logs to. Supported platforms are ```tensorboard```, ```wandb```, ```comet_ml``` and ```clearml```. Use ```all``` (default) to report to all integrations. This option is only applicable when ```--with_tracking``` is passed.

- ```--output_dir OUTPUT_DIR```: The location to store the final model.

- ```--learning_rate LR```: Initial learning rate (after the potential warmup period) to use.

- ```--reconstruct_rate RR```: Reconstruct rate to balance the model loss and reconstruction loss.

- ```--weight_transpose```: If this option is opened (default), the weight is transposed.

- ```--max_seq_length SEQ_LEN```: The maximum total input sequence length after tokenization. The preprocessed sequence length of bookcorpus dataset is ```128```.

- ```--num_train_epochs EPOCH```: Total number of training epochs to perform.

- ```--checkpointing_steps STEP```: The interval to save the various states, or ```epoch``` for each epoch.

- ```--centroid_requires_grad```: If this option is open, we require grad for centroids.

- ```--weight_requires_grad```: If this option is open, we require grad for weight.

- ```--vec_len VEC_LEN```: The length of the sub-vectors.

- ```--ncentroid N_CENTROID```: The number of centroids.

- ```--nsharecodebook N_SHARED_CODEBOOK```: The number of shared codebooks. Currently we do not share codebooks among different layers (i.e., ```--nsharecodebook``` is ```1```).

- ```--distance_p P```: The norm for distance calculation.

- ```--load_tokenized_datasets DATASET_DIR```: Path to a tokenized dataset pickle file to load instead of tokenizing the dataset from scratch. This argument should be in consistent with the directory used for pre-processing datasets.

To fine-tune and evaluate the converted models on downstream tasks in GLUE dataset, please run the following commands:

``` bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # adjust according to the GPUs you have
accelerate launch examples/run_lutrize_glue_no_trainer.py 
           [--task_name cola] [--model_name_or_path] [--resume_from_pretrain] 
           [--per_device_train_batch_size] [--per_device_eval_batch_size] 
           [--log_steps] [--with_tracking] [--report_to] [--output_dir]
           [--learning_rate] [--reconstruct_rate] [--weight_transpose] 
           [--max_length] [--num_train_epochs]
           [--centroid_requires_grad] [--weight_requires_grad]
           [--vec_len] [--ncentroid] [--nsharecodebook] [--distance_p] 
```

- ```--task_name TASK_NAME```: The name of the glue task to train on: ```mnli```, ```qqp```, ```qnli```, ```sst2```, ```cola```, ```stsb```, ```mrpc```, ```rte```. 

- ```--resume_from_pretrain MODEL_WEIGHT```: The calibrated model weight you want to use. It should be a ```safetensors``` file.

The other arguments have the same semantics as listed above.

#### Converting ViT Models

To convert ViT models into LUT-NN and evaluate the converted models, please run the following commands:

``` bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # adjust according to the GPUs you have
accelerate launch examples/run_luterize_image_classification_no_trainer.py
           [--model_name_or_path] [--dataset_name] 
           [--per_device_train_batch_size] [--per_device_eval_batch_size]
           [--log_steps] [--with_tracking] [--report_to] [--output_dir]
           [--learning_rate] [--reconstruct_rate] [--weight_transpose] 
           [--num_train_epochs] [--checkpointing_steps]
           [--centroid_requires_grad] [--weight_requires_grad]
           [--vec_len] [--ncentroid] [--nsharecodebook] [--distance_p]
           [--resume_from_pretrain]
```

- ```--model_name_or_path MODEL```: Path to ViT model from huggingface, e.g., ```google/vit-base-patch16-224-in21k```, ```google/vit-huge-patch14-224-in21k```, etc.

- ```--dataset_name DATASET_NAME```: The name of the dataset to use, e.g., ```cifar100```, ```cifar10```.

- ```--resume_from_pretrain```: The calibrated model weight you want to use. It should be a ```safetensors``` file. Set this option when you want to evaluate a calibrated model.

The other arguments have the same semantics as listed above.

#### Weight Conversion

Since UPMEM PIM-DIMM's floating point computation capacity is extremely low, we conduct INT8 quantization on the LUTs to deploy the models on UPMEM PIM-DIMM. To conduct weight conversion, please run the following commands:

``` bash
python pim_weight_converter.py
       [--weight_input_file] [--weight_output_file]
       [--features] [--ncentroids] [--vec_len] [--nsharecodebook]
```

- ```--weight_input_file INPUT```: The orginal model weight file you want to convert. It should be a  ```safetensors``` file.

- ```--weight_output_file OUTPUT```: The output file which saves the converted model weight. The converted weight is stored in Numpy's ```npz``` format.

- ```--features FEATURE```: The hidden dim of your model, e.g., ```768``` for ```bert-base```.

- ```--ncentroids N_CENTROID```: The number of centroids. It should be in consistent with your model config.

- ```--vec_len VEC_LEN```: The length of the sub-vectors. It should be in consistent with your model config.

- ```--nsharecodebook N_SHARED_CODEBOOK```: The number of shared codebooks. It should be in consistent with your model config.

Currently we only dump the weights of transformer modules after LUT-NN conversion into the output file. To dump other modules' weights, you can modify this script for further parsing the safetensor file.

#### Example Commands

We provide example bash scripts to run these model calibration scripts, which are saved in the ```model-calibration``` folder. You can modify these bash scripts to fit in with your requirements. Besides, you can also refer to the ```parse_args``` functions in these scripts to check all arguments you can adjust.

## 4. Auto-Tuner

The source codes for model calibration are in the `inference-engine/src/tuner` folder. The auto-tuner is stored in ```build/bin/tuner``` after it is compiled as described in [3. Installation](#3-installation).

#### Tuning Single LUT Kernel

Please run the following command to tune single LUT kernel:

``` bash
./build/bin/tuner 0 input_config_path output_config_path
```

In this command, ```input_config_path``` is the input config file containing the workload and hardware settings. ```output_config_path``` is the output config file with kernel parameters added. All config files are organized in ```yaml``` format. We provide sample input (```single_kernel_input_example.yaml```) & output (```single_kernel_output_example.yaml```) files in the ```inference-engine/configs``` folder for reference.

#### Tuning Transformer Model

Please run the following command to tune all LUT kernels in a transformer model:

``` bash
./build/bin/tuner 1 input_config_path output_config_path
```

In this command, ```input_config_path``` is the input config file containing the workload and hardware settings. ```output_config_path``` is the output config file with kernel parameters added. All config files are organized in ```yaml``` format. We provide sample input (```transformer_input_example.yaml```) & output (```transformer_output_example.yaml```) files in the ```inference-engine/configs``` folder for reference.

#### Adjusting Tuner Parameters to Other Servers

Currently the latency estimators in the auto-tuner is based on our UPMEM server (dual-socket Xeon 4210 CPUs, 128 GB DDR4 memory, and 8 PIM-DIMMs), which will vary under different server configurations. You can follow the latency profiling implementation in [PriM Benckmarks](https://github.com/CMU-SAFARI/prim-benchmarks) to re-profile each part's latency and update these estimators accordingly.

## 5. Inference Engine

The source codes for model calibration are in the `inference-engine/src/dpu` and `inference-engine/src/host` folders, which contain the PIM-side and host-side codes, respectively. Before running the following commands, please make sure you have finished installation as described in [3. Installation](#3-installation).

We provide three Python wrappers to run LUT-NN in different granularity. 

#### Run Single LUT Kernel

Please run the following command to run single LUT kernel:

``` bash
python run_amm.py [--amm_config_file] [--need_compile]
```

- ```--amm_config_file CONFIG```: The config file (e.g., ```single_kernel_output_example.yaml```) of LUT kernel.

- ```--need_compile```: If this argument is set, the PIM kernel of current LUT kernel will be compiled. (This is because UPMEM requires re-compile PIM kernel when the LUT kernel parameters are changed.)

#### Run Single Transformer Layer

Please run the following command to run single transformer layer:

``` bash
python run_layer.py [--transformer_config_file] [--need_compile] 
                    [--need_measure_energy] [--need_breakdown] [--breakdown_type]
```

- ```--transformer_config_file CONFIG```: The config file (e.g., ```transformer_output_example.yaml```) of the transformer model.

- ```--need_compile```: If this argument is set, the PIM kernels of current transformer model will be compiled.

- ```--need_measure_energy```: If this option is on, energy consumption will be reported after finishing execution. Note that this option requires ```sudo``` authority.

- ```--need_breakdown```: If this option is on, latency breakdown will be reported after finishing execution.

- ```--breakdown_type TYPE```: Set the latency breakdown type: ```0```: Breakdown latency among different operators. ```1```: Breakdown latency of the LUT kernel. ```2```: Breakdown latency of different LUT kernels in this transformer layer. This argument is only valid when ```--need_breakdown``` is on. 

#### Run Transformer Model

Please run the following command to run a transformer model:

``` bash
python run_model.py [--transformer_config_file] [--model_weight_file] [--need_compile]
```

- ```--transformer_config_file CONFIG```: The config file (e.g., ```transformer_output_example.yaml```) of the transformer model.

- ```--model_weight_file WEIGHT```: The weight file (dumped by ```pim_weight_converter.py```) of the transformer model.

- ```--need_compile```: If this argument is set, the PIM kernels of current transformer model will be compiled.

Currently we only implement the transformer blocks in the model. You can refer to other GGML-based transformer implementation (e.g., [bert.cpp](https://github.com/skeskinen/bert.cpp)) to add the token embedding layers and LM head layers.
