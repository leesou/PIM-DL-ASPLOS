#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # adjust according to the GPUs you have
accelerate launch examples/run_luterize_image_classification_no_trainer.py \
--model_name_or_path google/vit-base-patch16-224-in21k \
--dataset_name cifar100 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--log_steps 5 \
--with_tracking \
--report_to tensorboard \
--output_dir serialization_dir \
--learning_rate 5e-5 \
--reconstruct_rate 1e-3 \
--num_train_epochs 20 \
--checkpointing_steps 2000 \
--centroid_requires_grad \
--vec_len 2 \
--ncentroid 32 \
--nsharecodebook 1 \
--distance_p 2.0
