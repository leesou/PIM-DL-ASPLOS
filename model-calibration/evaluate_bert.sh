#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # adjust according to the GPUs you have
accelerate launch examples/run_lutrize_glue_no_trainer.py \
--task_name cola \
--model_name_or_path bert-base-uncased \
# adjust according to your model storage
--resume_from_pretrain $PATH_TO_PRETRAINED_MODEL/model.safetensors \ 
--output_dir serialization_dir \
--weight_transpose \
--nsharecodebook 1 \
--learning_rate 5e-5 \
--reconstruct_rate 4e-2 \
--vec_len 2 \
--ncentroid 16 \
--weight_requires_grad \
--per_device_eval_batch_size 4 \
--per_device_train_batch_size 4 \
--max_length 128 \
--with_tracking \
--log_steps 5 \
--distance_p 2.0 \
--num_train_epochs 20