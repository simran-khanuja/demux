from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import math
import os
import random
import glob
import shutil
import numpy as np
import wandb
import sys

import datasets 
import torch
import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from functools import partial
from datasets import disable_caching

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
)
from datasets import concatenate_datasets

from al_strategies import get_train_dataset

from helper.data_utils import load_train_datasets, load_test_datasets, preprocess_datasets
from helper.train_utils import eval_dev_test, eval_model_question_answering, eval_model_sequence_classification, eval_model_token_classification, eval_model_mt, find_optimal_k

import pandas as pd
import shutil
import yaml

logger = get_logger(__name__)
disable_caching()

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on Al strategies.")

    # Pass "false" if you simply want to finetune model on source languages/dataset
    parser.add_argument(
        "--do_active_learning",
        type=str, default="true",
        help="Whether to do AL or simply train."
    )

    # Dataset arguments
    parser.add_argument(
        "--source_languages",
        type=str,
        help="Source languages (to train on)",
    )
    parser.add_argument(
        "--target_languages",
        type=str,
        help="Target languages (to evaluate on)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--target_dataset_name",
        type=str,
        default=None,
        help="(Optional) The name of the test dataset to use, if different from the dataset_name passed above.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        help="Optional. One of token, sequence, qa, mt.",
    )
    parser.add_argument(
        "--dataset_config_file",
        type=str,
        default="scripts/train/dataset-configs.yaml",
        help="Dataset yaml config file containing info on labels, metrics, etc.",
    )
    parser.add_argument(
        "--source_dataset_path",
        type=str,
        help="Optional custom path to source dataset.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--target_dataset_path",
        type=str,
        help="Optional custom path to target dataset.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--per_language_subset_size",
        type=int,
        default=None,
        help=(
            "Per language subset size, if passed, will only use a subset of the data for each language."
        ),
        required=False,
    )
    parser.add_argument(
        "--save_dataset_path",
        type=str,
        help="Path where to save dataset selected by strategies.",
        required=False,
    )

    # For MT only
    parser.add_argument(
        "--decoder_target_language",
        type=str,
        default="en_XX",
        help="Decoder target language for MT, if task is MT.",
    )
    parser.add_argument(
        "--get_all_en_configs",
        type=bool,
        default=False,
        help="Whether to get all English configs for OPUS-100.",
    )

    # Model arguments
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
        "--do_lower_case",
        type=bool,
        default=False,
        help="Do lower case when tokenizing.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to cache directory.",
    )

    # For MT only
    parser.add_argument(
        "--predict_with_generate",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )

    # Inference arguments
    parser.add_argument(
        "--embedding_model",
        type=str,
        help="Model to obtain embeddings from"
    )
    parser.add_argument(
        "--embedding_method",
        type=str,
        default="cls",
        help="Embedding method, one of [cls, last_layer_mean].",
    )
    parser.add_argument(
        "--save_embeddings",
        type=bool,
        default=True,
        help="Whether to save [CLS] embeddings for each example in the test set."
    )
    parser.add_argument(
        "--save_embeddings_path",
        type=str,
        default=None,
        help="Where to store embeddings."
    )
    parser.add_argument(
        "--token_task_margin",
        type=str,
        default="min",
        help=(
            "How to calculate uncertainty for token level tasks. Mean takes the mean margin across all tokens. Min takes the min margin across all tokens."
        ),
    )
    parser.add_argument(
        "--qa_uncertainty_method",
        type=str,
        default="logits",
        help=(
            "How to calculate uncertainty for token level tasks. Mean takes the mean margin across all tokens. Min takes the min margin across all tokens."
        ),
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=1024,
        help="Batch size for inference.",
    )

    # Wandb arguments
    parser.add_argument(
        "--with_tracking",
        type=bool, default=True,
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default=None,
        help="Project name for wandb.",
    )
    
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Run name for wandb.",
    )
    parser.add_argument(
        "--target_config_name",
        type=str,
        help="Target config name. Only used in directory path to save embeddings/models/data",
        default="target_lp",
        required=False,
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    # Training arguments
    parser.add_argument(
        "--do_train",
        type=bool,
        default=True,
        help="Whether to run training"
    )
    parser.add_argument(
        "--do_predict",
        type=bool,
        default=True,
        help="Whether to run prediction on the test set. (Training will not be executed.)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help=(
            "(Only for MT) The maximum total input sequence length after "
            "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help=(
            "(Only for MT) The maximum total sequence length for target text after "
            "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool, default=True,
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
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
    parser.add_argument(
        "--early_stopping",
        type=bool, default=False,
        help="Whether to early stop.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of checkpoints for which the eval accuracy should be dropping before stoppint.",
    )
    parser.add_argument(
        "--early_stopping_language",
        type=str,
        default="en",
        help="Language to be used for early stopping.",
    )
    parser.add_argument(
        "--max_to_keep",
        type=float,
        default=5,
        help="Max checkpoints to keep.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform."
    )
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
        "--pred_output_dir",
        type=str,
        default=None,
        help="Where to store the predictions on the test set."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        type=bool, default=False,
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Steps after which you should log eval.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        help="Number of steps after which to eval."
    )
    parser.add_argument(
        "--save_predictions",
        type=bool, default=True,
        help="Whether to save predictions on the test set."
    )
    parser.add_argument(
        "--delete_model_output",
        type=bool, default=False,
        help="Set to true when doing hparam search and you don't want to waste space."
    )
    parser.add_argument(
        "--silent",
        type=bool, default=False,
        help="Set to true if you want silent logging"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--debug",
        type=bool, default=False,
        help="Activate debug mode and run training only with a subset of data.",
    )
    
    # Active learning arguments
    parser.add_argument(
        "--budget",
        type=str,
        default="10000",
        help=(
            "Budget"
        ),
    )
    parser.add_argument(
        "--max_train_emb_len",
        type=int,
        default=400000,
        help=(
            "Max size of train embedding matrix to fit in GPU memory"
        ),
    )
    parser.add_argument(
        "--max_target_emb_len",
        type=int,
        default=5000,
        help=(
            "Max size of train embed matrix for GPU memory"
        ),
    )
    parser.add_argument(
        "--total_rounds",
        type=int,
        default=5,
        help=(
            "Total number of active learning rounds."
        ),
    )
    parser.add_argument(
        "--k_value",
        type=int,
        default=None,
        help=(
            "Total number of K nearest neighbours for KNN-UNCERTAINTY."
        ),
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="average_dist",
        help=(
            "Strategy for AL"
        )
    )
    parser.add_argument(
        "--multiple_budgets_one_round",
        type=bool, default=False,
        help="multiple budgets one round?",
    )

    # Only used with random if user wants to specify the exact lang-wise budget. Used for the LITMUS baseline
    parser.add_argument(
        "--per_language_allocation_file",
        type=str,
        default=None,
        help=(
            "Comma separated list of language-wise budget allocations. Used only with random strategy."
        ),
    )

    args = parser.parse_args()

    # Check if source and target paths are None
    if args.source_dataset_path == "None":
        args.source_dataset_path = None
    if args.target_dataset_path == "None":
        args.target_dataset_path = None

    # Override source and target languages if source and target dataset paths are provided
    if args.source_dataset_path is not None:
        args.source_languages = None
    if args.target_dataset_path is not None:
        args.target_languages = None

    # If predict: check whether test file present
    if args.do_predict:
        if args.target_languages is None and args.target_dataset_path is None:
            raise ValueError("Need to specify target languages for predict mode.")
    
    if args.pred_output_dir is None and args.save_predictions:
        raise ValueError("Need prediction output dir for save predictions mode.")
    
    if not args.wandb_name:
        args.wandb_name = args.strategy
    
    # Get list of source languages for gold strategy
    if args.strategy.startswith("gold"):
        source_languages_list = args.strategy.split("_")[1:]
        args.source_languages = ",".join(source_languages_list)
    
    # Check whether to do active learning
    if args.do_active_learning == "true":
        args.do_active_learning = True
    else:
        args.do_active_learning = False

    return args


def main():
    args = parse_args()
      
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    config = {
        "num_iterations": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.per_device_train_batch_size,
        "seed": args.seed,
    }
    accelerator = (
        Accelerator(log_with=args.report_to, logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    level = logging.INFO if accelerator.is_main_process else logging.ERROR

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.with_tracking:
        args.wandb_name = args.wandb_name + f"_{args.learning_rate}" + f"_{args.num_train_epochs}" + f"_{args.seed}"
        accelerator.init_trackers(project_name=args.project_name, config=config, init_kwargs={"wandb": {"name": args.wandb_name}})

    if not args.silent:
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
    else:
        transformers.logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()
        datasets.utils.logging.disable_propagation()
        datasets.disable_progress_bar()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    # Read in dataset config yaml file
    with open(args.dataset_config_file, "r") as f:
        dataset_config = yaml.safe_load(f)[args.dataset_name]
    
    # Get metric, columns to remove, task_type and label_names
    metric = evaluate.load(dataset_config["metric"])
    args.remove_columns = None
    if "remove_columns" in dataset_config:
        args.remove_columns = dataset_config["remove_columns"]
    args.task_type = dataset_config["task_type"]
    label_names = None
    if args.task_type not in ["qa", "mt"]:
        label_names = dataset_config["label_names"]
        num_labels = len(label_names)
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {label: i for i, label in enumerate(label_names)}
   
    # Initialize config and tokenizer
    if args.task_type not in ["qa", "mt"]:
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            cache_dir=args.cache_dir,
        )
    else:
        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,	
            cache_dir=args.cache_dir
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir,
        use_fast=not args.use_slow_tokenizer,
    )
    
    # Initialize model 
    if args.task_type in ["token"]:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
        )
    elif args.task_type in ["sequence"]:	
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
        )
    elif args.task_type in ["qa"]:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
        )
    elif args.task_type in ["mt"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
        )

        # Set model.config.decoder_start_token_id to decoder_target_language
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.decoder_target_language]

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        if args.task_type in ["mt"]:
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=-100,
                pad_to_multiple_of=8 if args.fp16 else None,
            )
        else:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    if args.do_train:
        # Get train, validation and unlabelled target datasets from specified source and target languages
        raw_train_datasets, processed_train_datasets, raw_validation_datasets, \
            processed_validation_datasets, raw_target_datasets, processed_target_datasets, en_raw_train_dataset, en_processed_train_dataset \
                = load_train_datasets(args, accelerator, tokenizer, args.remove_columns)
        
    if args.do_predict:
        # Get test dataset from specified target languages
        raw_test_datasets, processed_test_datasets = load_test_datasets(args, accelerator, tokenizer, args.remove_columns)
    
    
    # Main training loop
    def train_loop(train_dataset_AL, output_dir_AL, pred_output_dir_AL, iteration):
        
        # Initialize model 
        if args.task_type in ["token"]:
            model = AutoModelForTokenClassification.from_pretrained(
                args.model_name_or_path,
                config=config,
                cache_dir=args.cache_dir,
            )
        elif args.task_type in ["sequence"]:	
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path,
                config=config,
                cache_dir=args.cache_dir,
            )
        elif args.task_type in ["qa"]:
            model = AutoModelForQuestionAnswering.from_pretrained(
                args.model_name_or_path,
                config=config,
                cache_dir=args.cache_dir,
            )
        elif args.task_type in ["mt"]:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                config=config,
                cache_dir=args.cache_dir,
            )
        if args.remove_columns is None:
            args.remove_columns = train_dataset_AL.column_names
        
        with accelerator.main_process_first():
            processed_train_dataset_AL_iteration = train_dataset_AL.map(
                partial(preprocess_datasets,
                args=args,
                remove_columns=args.remove_columns,
                tokenizer=tokenizer,
                dataset=args.dataset_name,
                padding="max_length" if args.pad_to_max_length else False,
                max_seq_length=args.max_seq_length,
                train=True),
                batched=True,
                remove_columns=args.remove_columns,
            )

        # Log a few random samples from the training set:
        for index in random.sample(range(len(processed_train_dataset_AL_iteration)), 3):
            logger.info(f"Sample {index} of the training set: {processed_train_dataset_AL_iteration[index]}.")

        train_dataloader = DataLoader(
            processed_train_dataset_AL_iteration,
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size,
            shuffle=True
        )

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

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

        if args.max_train_steps is None:
            max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            num_train_epochs = args.num_train_epochs
        else:
            num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
            max_train_steps = args.max_train_steps

        num_warmup_steps = args.warmup_ratio * max_train_steps

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        # Calculate the number of checkpoints to save and the checkpoint interval.
        if args.checkpointing_steps is None:
            checkpointing_steps = max(max_train_steps // 5, 500)
        else:
            checkpointing_steps = args.checkpointing_steps

        # Train!
        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset_AL)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        logger.info(f"  Checkpointing steps = {checkpointing_steps}")

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)

        completed_steps = 0
        starting_epoch = 0
        max_val_acc = 0
        best_ckpt_path = os.path.join(output_dir_AL,  "best_checkpoint")
        accuracy_drop_count = 0
        best_ckpt = None
        best_ckpt_step = 0
        if iteration:
            iteration_suffix = "_iter{}".format(iteration)
        else:
            iteration_suffix = ""
    
        logger.info("***** Running evaluation at step {} *****".format(completed_steps))
        dev_test_accuracies = eval_dev_test(model, tokenizer, args.task_type, args.dataset_name, processed_validation_datasets, processed_test_datasets,
        raw_validation_datasets, raw_test_datasets, metric, completed_steps, iteration, 0, args, data_collator, accelerator, label_names)

        # Save accuracies at step 0
        with open(os.path.join(output_dir_AL, "initial_accuracies.json"), "w") as f:
            json.dump(dev_test_accuracies, f)
        # Save to csv as well
        df = pd.DataFrame(dev_test_accuracies, index=[0])
        df.to_csv(os.path.join(output_dir_AL, "initial_accuracies.csv"), index=False)
            
        for _ in range(starting_epoch, num_train_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                if args.task_type in ["qa"]:
                    outputs = model(input_ids=batch["input_ids"], 
                                    attention_mask=batch["attention_mask"],
                                    start_positions=batch["start_positions"],
                                    end_positions=batch["end_positions"])
                else:
                    outputs = model(input_ids=batch["input_ids"], 
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"])
                loss = outputs.loss

                # We keep track of the loss at each epoch
                total_loss += loss.detach().float()
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
            
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                
                # Log train metrics
                if args.with_tracking:
                    accelerator.log(
                        {
                            f"train_loss{iteration_suffix}": total_loss / (step + 1),
                        },
                        step=completed_steps,
                    )

                if completed_steps % int(checkpointing_steps) == 0:
                    ckpt_dir = f"step_{completed_steps}.ckpt"
                    if output_dir_AL is not None:
                        output_dir = os.path.join(output_dir_AL, ckpt_dir)
                    # Save model checkpoint
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
                    logger.info(f"Saving checkpoint to {output_dir}")
                    # Check for early stopping by keeping the max of validation accuracy
                    dev_test_accuracies = eval_dev_test(model, tokenizer, args.task_type, args.dataset_name, processed_validation_datasets, processed_test_datasets,
                        raw_validation_datasets, raw_test_datasets, metric, completed_steps, iteration, max_train_steps, args, data_collator, accelerator, label_names)
                    if args.early_stopping:
                        if dev_test_accuracies[args.early_stopping_language+"_val"] > max_val_acc:
                            max_val_acc = dev_test_accuracies[args.early_stopping_language+"_val"]
                            best_ckpt = model
                            best_ckpt_step = completed_steps
                        else:
                            # check for patience
                            if accuracy_drop_count >= args.early_stopping_patience:
                                logger.info(f"Validation accuracy dropped for {args.early_stopping_patience} rounds of {checkpointing_steps} steps. Stopping training.")
                                break
                            else:
                                accuracy_drop_count += 1
                                logger.info(f"Validation accuracy dropped. Count: {accuracy_drop_count}*{checkpointing_steps}")
                    # Delete all checkpoints except last "max_to_keep" checkpoints based on timestamp. Each checkpoint is a directory.
                    if args.max_to_keep > 0:
                        checkpoint_dirs = glob.glob(os.path.join(output_dir_AL, "step_*.ckpt"))
                        checkpoint_dirs.sort(key=os.path.getmtime)
                        if len(checkpoint_dirs) > args.max_to_keep:
                            shutil.rmtree(checkpoint_dirs[-1])
                            logger.info(f"Deleted checkpoint {checkpoint_dirs[-1]}")

                if completed_steps >= max_train_steps:
                    break

        # Save the final model
        if output_dir_AL is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                output_dir_AL, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if best_ckpt:
                unwrapped_best_ckpt = accelerator.unwrap_model(best_ckpt)
                unwrapped_best_ckpt.save_pretrained(best_ckpt_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                logger.info(f"Saved best checkpoint to {best_ckpt_path}")
                with open(os.path.join(best_ckpt_path, "best_ckpt_step.json"), "w") as f:
                    json.dump({"best_ckpt_step": best_ckpt_step}, f)
                    
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir_AL)
                if best_ckpt:
                    tokenizer.save_pretrained(best_ckpt_path)
            
            logger.info("***** Running evaluation at step {} *****".format(completed_steps))
            dev_test_accuracies = eval_dev_test(model, tokenizer, args.task_type, args.dataset_name, processed_validation_datasets, processed_test_datasets,
                        raw_validation_datasets, raw_test_datasets, metric, completed_steps, iteration, max_train_steps, args, data_collator, accelerator, label_names)
            with open(os.path.join(output_dir_AL, "all_results.json"), "w") as f:
                json.dump(dev_test_accuracies, f)
            # Save validation accuracies dict to csv
            df = pd.DataFrame(dev_test_accuracies, index=[0])
            df.to_csv(os.path.join(output_dir_AL, "all_results.csv"), index=False)

            # Remove checkpointing directories
            checkpoint_dirs = glob.glob(os.path.join(output_dir_AL, "step_*.ckpt"))
            for ckpt_dir in checkpoint_dirs:
                shutil.rmtree(ckpt_dir)
                logger.info(f"Deleted checkpoint {ckpt_dir}")
    
        if args.do_predict:
            for language in processed_test_datasets:
                # TODO: Can this be moved to loading / preprocessing?
                processed_test_dataset = processed_test_datasets[language]
                if args.task_type in ["qa"] and "example_id" in processed_test_dataset.features:
                    processed_test_dataset= processed_test_dataset.remove_columns(["example_id", "offset_mapping"])
                 
                test_dataloader = DataLoader(processed_test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

                metric_to_track = "f1"
                if args.task_type in ["token"]:
                    eval_metric, _ = eval_model_token_classification(model, test_dataloader, metric, label_names, accelerator)
                elif args.task_type in ["sequence"]:
                    eval_metric, _ = eval_model_sequence_classification(model, test_dataloader, metric, accelerator)
                elif args.task_type in ["qa"]:
                    eval_metric, _ = eval_model_question_answering(model, test_dataloader, metric,  processed_test_datasets[language], raw_test_datasets[language], accelerator)
                elif args.task_type in ["mt"]:
                    eval_metric = eval_model_mt(args, tokenizer, model, test_dataloader, metric, accelerator)
                    metric_to_track = "score"
                logger.info(f"test {metric_to_track} for {language}: {eval_metric[metric_to_track]}")
            

                if args.task_type not in ["mt"]:
                    # Write predictions to output file
                    logger.info(f"Writing predictions for {language} to {pred_output_dir_AL}...")
                    if pred_output_dir_AL is not None:
                        os.makedirs(pred_output_dir_AL, exist_ok=True)
                        pred_output_file = os.path.join(pred_output_dir_AL, "predictions_" + language + ".txt")
                        with open(pred_output_file, "w") as f:
                            for pred in eval_metric["predictions"]:
                                f.write(str(pred) + "\n")
    
        accelerator.free_memory()
        return 
    
    if args.do_train:
        # Get all budget values if multiple are passed
        if "," in args.budget:
            budget_list = [int(b) for b in args.budget.split(",")]
        else:
            budget_list = [int(args.budget)]
        
        # Get raw_train_dataset and processed_train_dataset from dictionaries of raw and processed datasets
        raw_train_dataset = concatenate_datasets(list(raw_train_datasets.values()))
        raw_target_dataset = concatenate_datasets(list(raw_target_datasets.values()))
        processed_train_dataset = concatenate_datasets(list(processed_train_datasets.values()))
        processed_target_dataset = concatenate_datasets(list(processed_target_datasets.values()))
        
        # Initialize variables
        train_embeddings = None
        train_logits = None
        train_uncertainty = None
        target_embeddings = None
        target_logits = None
        target_uncertainty = None
        original_save_dataset_path = args.save_dataset_path
        original_pred_output_dir = args.pred_output_dir
        original_output_dir = args.output_dir
        original_save_embeddings_path = args.save_embeddings_path
    
        for budget in budget_list:
            args.save_dataset_path = original_save_dataset_path
            args.pred_output_dir = original_pred_output_dir
            args.output_dir = original_output_dir
            args.save_embeddings_path = original_save_embeddings_path
            logger.info(f"Starting training with budget {budget}...")
            if args.do_active_learning is True:
                per_iteration_budget = budget // args.total_rounds
                num_iterations = 1
                output_dir_AL = None
                selected_indices = []
                while num_iterations <= args.total_rounds:
                    logger.info(f"Starting active learning iteration {num_iterations}...")
                    if num_iterations == 1:
                        # Suffix creation for saving files
                        if "uncertainty" in args.strategy and args.task_type in ["token"]:
                            strategy_folder_name = args.strategy + "_margin_" + str(args.token_task_margin)
                        else:
                            strategy_folder_name = args.strategy

                        # Handle the repository creation
                        hparam_suffix = f"/{args.embedding_model}" + f"/{args.dataset_name}" + f"/{args.target_config_name}" + f"/{budget}"+ f"/{args.learning_rate}" + f"_{args.num_train_epochs}" + f"_seed_{args.seed}" + f"/{strategy_folder_name}"
                        # Create repository for saving files
                        if accelerator.is_main_process:
                            args.save_dataset_path += hparam_suffix
                            args.output_dir += hparam_suffix
                            args.pred_output_dir += hparam_suffix
                            args.save_embeddings_path += hparam_suffix
                            if not os.path.exists(args.save_dataset_path):
                                os.makedirs(args.save_dataset_path, exist_ok=True)
                            if not os.path.exists(args.output_dir):
                                os.makedirs(args.output_dir, exist_ok=True)
                            if not os.path.exists(args.pred_output_dir):
                                os.makedirs(args.pred_output_dir, exist_ok=True)
                            if not os.path.exists(args.save_embeddings_path):
                                os.makedirs(args.save_embeddings_path, exist_ok=True)
                            logger.info(f"Saving dataset to {args.save_dataset_path}")
                            logger.info(f"Saving model to {args.output_dir}")
                            logger.info(f"Saving predictions to {args.pred_output_dir}")
                            logger.info(f"Saving embeddings to {args.save_embeddings_path}")
                        accelerator.wait_for_everyone()

                    # Load model from model path
                    model_path=output_dir_AL if num_iterations > 1 else args.model_name_or_path
                    logger.info("Loading model from %s", model_path)
                    if args.task_type in ["token"]:
                        model = AutoModelForTokenClassification.from_pretrained(
                            model_path,
                            config=config,
                            cache_dir=args.cache_dir
                        )
                    elif args.task_type in ["qa"]:	
                        model = AutoModelForQuestionAnswering.from_pretrained(
                            args.model_name_or_path,
                            config=config,
                            cache_dir=args.cache_dir,
                        )
                    elif args.task_type in ["sequence"]:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            model_path,
                            config=config,
                            cache_dir=args.cache_dir
                        )
                
                    # Raw train datasets for each language are only needed for the egalitarian strategy
                    if args.multiple_budgets_one_round:
                        train_dataset_AL, selected_indices, train_embeddings, train_logits, train_uncertainty, target_embeddings, target_logits, target_uncertainty = get_train_dataset(
                                                        args,
                                                        args.task_type,
                                                        raw_train_datasets,
                                                        raw_target_datasets,
                                                        processed_train_dataset,
                                                        processed_target_dataset,
                                                        selected_indices,
                                                        model=model,
                                                        budget=num_iterations*per_iteration_budget,
                                                        strategy=args.strategy,
                                                        iteration=num_iterations,
                                                        accelerator=accelerator,
                                                        train_embeddings=train_embeddings,
                                                        train_logits=train_logits,
                                                        train_uncertainty=train_uncertainty,
                                                        target_embeddings=target_embeddings,
                                                        target_logits=target_logits,
                                                        target_uncertainty=target_uncertainty,
                                                        return_embeddings=True)
                    else:
                        train_dataset_AL, selected_indices, _, _, _, _, _, _ = get_train_dataset(
                                                        args,
                                                        args.task_type,
                                                        raw_train_datasets,
                                                        raw_target_datasets,
                                                        processed_train_dataset,
                                                        processed_target_dataset,
                                                        selected_indices,
                                                        model=model,
                                                        budget=num_iterations*per_iteration_budget,
                                                        strategy=args.strategy,
                                                        iteration=num_iterations,
                                                        accelerator=accelerator,
                                                        train_embeddings=train_embeddings,
                                                        train_logits=train_logits,
                                                        train_uncertainty=train_uncertainty,
                                                        target_embeddings=target_embeddings,
                                                        target_logits=target_logits,
                                                        target_uncertainty=target_uncertainty,
                                                        return_embeddings=False)

                    logger.info(f"Selected {len(selected_indices)} samples for training.")
                    output_dir_AL = os.path.join(args.output_dir, f"iter_{num_iterations}")
                    pred_output_dir_AL = os.path.join(args.pred_output_dir, f"iter_{num_iterations}")
                    # Make sure output directories exist
                    if not os.path.exists(output_dir_AL):
                        os.makedirs(output_dir_AL)
                    if not os.path.exists(pred_output_dir_AL):
                        os.makedirs(pred_output_dir_AL)
                    if args.multiple_budgets_one_round:
                        _ = train_loop(train_dataset_AL, output_dir_AL, pred_output_dir_AL, budget)
                    else:
                        _ = train_loop(train_dataset_AL, output_dir_AL, pred_output_dir_AL, num_iterations)
                    logger.info(f"Finished active learning iteration {num_iterations}.")
                    num_iterations += 1
                
                # Delete model output directories if flag is set
                if args.delete_model_output:
                    logger.info("Deleting model output directories...")
                    for i in range(1, args.total_rounds + 1):
                        shutil.rmtree(os.path.join(args.output_dir, f"iter_{i}"))
                        logger.info(f"Deleted model output directory for iteration {i}")
        
            else:
                if accelerator.is_main_process:
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir, exist_ok=True)
                    if not os.path.exists(args.pred_output_dir):
                        os.makedirs(args.pred_output_dir, exist_ok=True)
                    if not os.path.exists(args.save_dataset_path):
                        os.makedirs(args.save_dataset_path, exist_ok=True)
                    logger.info(f"Saving model to {args.output_dir}")
                    logger.info(f"Saving predictions to {args.pred_output_dir}")
                accelerator.wait_for_everyone()
                # Save raw train dataset to csv
                if accelerator.is_main_process:
                    logger.info(f"Saving raw train dataset to {args.save_dataset_path}")
                    logger.info(f"Raw train dataset length: {len(raw_train_dataset)}")
                    raw_train_dataset.to_csv(os.path.join(args.save_dataset_path, "train.csv"), index=False)
                _ = train_loop(raw_train_dataset, args.output_dir, args.pred_output_dir, None)
        
        if args.with_tracking:
            accelerator.end_training()
    
    if not args.do_train and args.do_predict:
        for language in processed_test_datasets:
            # TODO: Can this be moved to loading / preprocessing?
            processed_test_dataset = processed_test_datasets[language]
            if args.task_type in ["qa"] and "example_id" in processed_test_dataset.features:
                processed_test_dataset= processed_test_dataset.remove_columns(["example_id", "offset_mapping"])
                
            test_dataloader = DataLoader(processed_test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

            metric_to_track = "f1"
            if args.task_type in ["token"]:
                eval_metric, _ = eval_model_token_classification(model, test_dataloader, metric, label_names, accelerator)
            elif args.task_type in ["sequence"]:
                eval_metric, _ = eval_model_sequence_classification(model, test_dataloader, metric, accelerator)
            elif args.task_type in ["qa"]:
                eval_metric, _ = eval_model_question_answering(model, test_dataloader, metric,  processed_test_datasets[language], raw_test_datasets[language], accelerator)
            elif args.task_type in ["mt"]:
                eval_metric = eval_model_mt(args, tokenizer, model, test_dataloader, metric, accelerator)
                metric_to_track = "score"
            logger.info(f"test {metric_to_track} for {language}: {eval_metric[metric_to_track]}")
        

            if args.task_type not in ["mt"]:
                # Write predictions to output file
                logger.info(f"Writing predictions for {language} to {pred_output_dir_AL}...")
                if pred_output_dir_AL is not None:
                    os.makedirs(pred_output_dir_AL, exist_ok=True)
                    pred_output_file = os.path.join(pred_output_dir_AL, "predictions_" + language + ".txt")
                    with open(pred_output_file, "w") as f:
                        for pred in eval_metric["predictions"]:
                            f.write(str(pred) + "\n")
        
##########################################################################################################################
##########################################################################################################################

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # raise  # re-raise the last exception
        # Exit with 1 to indicate failure
        sys.exit(1)
        

