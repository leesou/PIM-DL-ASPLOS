#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # adjust according to the GPUs you have
accelerate launch examples/run_lutrize_mlm_no_trainer.py \
--model_type bert \
--dataset_name bookcorpus \
--dataset_config_name plain_text \
--model_name_or_path bert-base-uncased \
--per_device_train_batch_size 6 \
--per_device_eval_batch_size 6 \
--log_steps 5 \
--with_tracking \
--report_to tensorboard \
--output_dir serialization_dir \
--learning_rate 1e-3 \
--reconstruct_rate 1e-3 \
--max_seq_length 128 \
--num_train_epochs 20 \
--checkpointing_steps 2000 \
--centroid_requires_grad \
--vec_len 2 \
--ncentroid 16 \
--nsharecodebook 1 \
--distance_p 2.0 \
--load_tokenized_datasets bert_bookcorpus.token.128
