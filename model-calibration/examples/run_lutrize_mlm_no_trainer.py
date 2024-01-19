#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
import safetensors
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

import pickle
from LUTNeuro import LUTerize
from LUTNeuro.LUTLinear_t import LUTLinear_t

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=False,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument("--log_steps", type=int, default=500, help="Accelerator logging interval.")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--load_tokenized_datasets",
        type=str,
        default=None,
        help="Path to a tokenized dataset pickle file to load instead of tokenizing the dataset from scratch.",
    )
    parser.add_argument(
        "--store_tokenized_datasets",
        type=str,
        default=None,
        help="Path to a tokenized dataset pickle file to store.",
    )
    parser.add_argument(
        "--ncentroid",
        type=int,
        default=16,
        help="The number of centroids"
    )

    parser.add_argument(
        "--vec_len",
        type=int,
        default=2,
        help="The length of the product quantization vector"
    )
    parser.add_argument(
        "--nsharecodebook",
        type=int,
        default=1,
        help="The number of shared codebooks"
    )

    parser.add_argument(
        "--activation-dir",
        type=str,
        default="data/activation/",
        help="dir where the activations are stored"
    )
    parser.add_argument(
        "--centroid-dir",
        type=str,
        default="data/centroid/",
        help="dir where the centroids are stored"
    )
    parser.add_argument(
        "--nsample",
        type=int,
        default=1024,
        help="The number of samples to be used for product quantization"
    )
    parser.add_argument(
        "--disable_finetune",
        action="store_true",
        default=False,
        help="Whether or not to disable finetuning.",
    )
    parser.add_argument(
        "--weight_transpose",
        action="store_true",
        default=True,
        help="Whether or not to transpose the weights.",
    )
    parser.add_argument(
        "--init_centroids",
        action="store_true",
        default=True,
        help="Whether or not to initialize centroids from numpy",
    )
    parser.add_argument(
        '--reconstruct_rate',
        type=float,
        default=0.01,
        help='reconstruct rate'
    )
    parser.add_argument(
        '--centroid_requires_grad',
        action="store_true",
        default=False,
        help="Whether or not to require grad for centroids",
    )
    parser.add_argument(
        '--weight_requires_grad',
        action="store_true",
        default=False,
        help="Whether or not to require grad for weight",
    )
    parser.add_argument(
        "--resume_from_centroids",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--resume_from_pretrain",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--distance_p",
        type=str,
        default="2.0",
        help="The distance p for product quantization",
    )
    args = parser.parse_args()

    # update output_dir
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = (
        f"nsharecb={args.nsharecodebook}_"
        f"veclen={args.vec_len}_"
        f"ncentroids={args.ncentroid}_"
        f"wgrad_{args.weight_requires_grad}_"
        f"cgrad_{args.centroid_requires_grad}_"
        f"batch={args.per_device_train_batch_size}_"
        f"lr={args.learning_rate}_"
        f"recon={args.reconstruct_rate}_"
        f"p={args.distance_p}_"
        f"{timestamp}"
    )
    args.output_dir = os.path.join(args.output_dir, log_filename)

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`train_file` should be a csv, json or txt file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or txt file.")

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.load_tokenized_datasets is None:
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
        else:
            data_files = {}
            if args.train_file is not None:
                data_files["train"] = args.train_file
            if args.validation_file is not None:
                data_files["validation"] = args.validation_file
            extension = args.train_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
            raw_datasets = load_dataset(extension, data_files=data_files)
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{args.validation_split_percentage}%]",
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{args.validation_split_percentage}%:]",
                )
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    lut_model = LUTerize.LUTerize(model=model,
                                  dataloader=None,
                                  tokenizer=None,
                                  activation_dir=None,
                                  centroid_dir=None,
                                  output_dir=None,
                                  ncentroid=args.ncentroid,
                                  nsharecodebook=args.nsharecodebook,
                                  vec_len=args.vec_len,
                                  nsample=None,
                                  logger=logger,
                                  lut_layers=[i for i in range(100)],
                                  weight_transpose=True,
                                  init_centroids=False,
                                  fp16=False,
                                  distance_p=args.distance_p,)

    lut_model.luterize_model()
    model = lut_model.model

    if args.resume_from_pretrain is not None:
        logger.info(f"Loading pretrained lut from checkpoint {args.resume_from_pretrain}")
        if args.resume_from_pretrain.endswith(".bin"):
            state_dict = torch.load(args.resume_from_pretrain, map_location="cpu")
        elif args.resume_from_pretrain.endswith(".safetensors"):
            with open(args.resume_from_pretrain, "rb") as f:
                state_dict = safetensors.torch.load(f.read())
                state_dict = {k: v.to('cpu') for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    # set required grad to False for each module
    def set_requires_grad(nested_model, requires_grad=False):
        for param in nested_model.parameters():
            param.requires_grad = requires_grad
    set_requires_grad(model, args.weight_requires_grad)

    def set_embedding_requires_grad(nested_model, requires_grad=False):
        for name, param in nested_model.named_parameters():
            if 'embedding' in name.lower():
                param.requires_grad = requires_grad

    set_embedding_requires_grad(model, False)

    # set required grad to True for centroid in each LUTLinear module
    lut_linear_modules = [module for module in model.modules() if isinstance(module, LUTLinear_t)]
    for module in lut_linear_modules:
        module.centroids.weight.requires_grad = args.centroid_requires_grad

    # load pretrained lut from checkpoint
    if args.resume_from_centroids is not None:
        logger.info(f"Loading pretrained lut from checkpoint {args.resume_from_centroids}")
        model.load_state_dict(torch.load(args.resume_from_centroids))

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))


    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    if args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=True,
            )

        with accelerator.main_process_first():
            if args.load_tokenized_datasets is None:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=[text_column_name],
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on dataset line_by_line",
                )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with accelerator.main_process_first():
            if args.load_tokenized_datasets is None:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not args.overwrite_cache,
                    desc="Running tokenizer on every text in dataset",
                )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
            if total_length >= max_seq_length:
                total_length = (total_length // max_seq_length) * max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        with accelerator.main_process_first():
            if args.load_tokenized_datasets is not None:
                logger.info(f"Loading tokenized datasets from {args.load_tokenized_datasets}")
                tokenized_datasets = datasets.load_from_disk(args.load_tokenized_datasets)
            else:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=not args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {max_seq_length}",
                )
                if args.store_tokenized_datasets is not None:
                    logger.info(f"Saving tokenized datasets at {args.store_tokenized_datasets}")
                    tokenized_datasets.save_to_disk(args.store_tokenized_datasets)
                    exit()

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Conditional for small test subsets
    if len(train_dataset) > 3:
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    accelerator.wait_for_everyone()

    print(model)

    unwrapped_model = accelerator.unwrap_model(model)

    print(unwrapped_model)
    # unwrapped_model._set_model_trainable(True)
    # unwrapped_model._set_lut_trainable(False)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("mlm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    # calculate the total number of tokens
    total_tokens = 0

    # train the model
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss_log = 0
            model_loss_log = 0
            lut_loss_log = 0

        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1
                    continue

            num_tokens = torch.sum(batch["input_ids"] != 0).item()
            total_tokens += num_tokens

            with accelerator.accumulate(model):
                outputs = model(**batch)
                # model loss
                model_loss = outputs.loss
                # We keep track of the loss at each epoch
                # add reconstruction loss
                # unwrapped_model = accelerator.unwrap_model(model)
                # lut_loss = unwrapped_model.reconstruction_loss(args.reconstruct_rate)

                 # add reconstruction loss
                lut_loss = 0
                for name, module in model.named_modules():
                    if isinstance(module, LUTLinear_t):
                        lut_loss += module.lut_loss

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    model_loss_log += model_loss.detach().clone()
                    lut_loss_log += lut_loss.detach().clone()
                    total_loss_log = model_loss_log + (lut_loss_log * args.reconstruct_rate)

                loss = model_loss + (lut_loss * args.reconstruct_rate)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

                # log step
                if completed_steps % args.log_steps == 0:
                    if args.with_tracking and accelerator.is_main_process:
                        # log total loss
                        accelerator.log(
                            {"loss_step/total_loss": total_loss_log.item() / args.log_steps},
                            step=completed_steps,
                        )
                        accelerator.log(
                            {"loss_step/lut_loss": lut_loss_log.item() / args.log_steps},
                            step=completed_steps,
                        )
                        accelerator.log(
                            {"loss_step/model_loss": model_loss_log.item() / args.log_steps},
                            step=completed_steps,
                        )

                        # log total loss by token
                        accelerator.log(
                            {"loss_token/total_loss": total_loss_log.item() / args.log_steps},
                            step=total_tokens,
                        )
                        accelerator.log(
                            {"loss_token/lut_loss": lut_loss_log.item() / args.log_steps},
                            step=total_tokens,
                        )
                        accelerator.log(
                            {"loss_token/model_loss": model_loss_log.item() / args.log_steps},
                            step=total_tokens,
                        )

                        total_loss_log = 0
                        lut_loss_log = 0
                        model_loss_log = 0
                        # log learning rate
                        accelerator.log({"lr": optimizer.param_groups[0]["lr"]}, step=completed_steps)

                        # Assuming 'model' is your model and 'accelerator' is your Accelerator instance
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                # calculate and log statistics of gradients, weights and biases
                                if param.grad is not None:
                                    # print('add grad', name, param.grad.shape)
                                    mean_grad = param.grad.data.mean().item()
                                    if len(param.grad.data.shape) > 1:
                                        std_grad = param.grad.data.std().item()
                                        accelerator.log({f"grad/{name}_mean": mean_grad, f"grad/{name}_std": std_grad}, step=completed_steps)
                                    else:
                                        accelerator.log({f"grad/{name}_mean": mean_grad, f"grad/{name}_std": std_grad}, step=completed_steps)
                                if "weight" in name and "centroid" not in name:
                                    mean_weight = param.data.mean().item()
                                    std_weight = param.data.std().item()
                                    accelerator.log({f"weight/{name}_mean": mean_weight, f"weight/{name}_std": std_weight}, step=completed_steps)

                                if "bias" in name:
                                    mean_bias = param.data.mean().item()
                                    std_bias = param.data.std().item()
                                    accelerator.log({f"bias/{name}_mean": mean_bias, f"bias/{name}_std": std_bias}, step=completed_steps)

                                if "centroid" in name:
                                    mean_centroid = param.data.mean().item()
                                    std_centroid = param.data.std().item()
                                    accelerator.log({f"centroid/{name}_mean": mean_centroid, f"centroid/{name}_std": std_centroid}, step=completed_steps)

                                    if args.report_to.lower() == "all" or args.report_to.lower() == "tensorboard":
                                        tensorboard_tracker = accelerator.get_tracker("tensorboard")
                                        tensorboard_tracker.writer.add_histogram(f"centroid_hist/{name}", param.data, global_step=completed_steps)

                    else:
                        accelerator.log({"step": completed_steps}, step=completed_steps)

                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity}")

        if args.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss_log / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
