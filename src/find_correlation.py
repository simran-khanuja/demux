import argparse
import transformers
import collections
import datasets
import os
import torch
from functools import partial
from datasets import load_dataset, concatenate_datasets, load_from_disk
import matplotlib.pyplot as plt
import logging
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    DataCollatorWithPadding,
)
from torch.nn import CrossEntropyLoss
import scipy
import yaml

from helper.data_utils import preprocess_datasets, deduplicate_dataset, get_tydiqa_dataset

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Find uncertainty correlation between source and target points")

    # Dataset arguments
    parser.add_argument(
        "--source_languages",
        type=str,
        help="Source languages that the model is fine-tuned on",
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
        "--task_type",
        type=str,
        default=None,
        help="The type of task to use (e.g. sequence classification, token classification, etc.).",
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
            "Per language subset size if passed. Otherwise, use the full train dataset."
        ),
    )
    # Optionally, provide path to train dataset
    parser.add_argument(
        "--selected_train_dataset_path",
        type=str,
        help="Path where selected train dataset is saved.",
        required=False,
    )

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
        required=False,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
        required=False,
    )

    # Inference arguments
    parser.add_argument(
        "--embedding_method",
        type=str,
        default="cls",
        help="Embedding used to represent the sequence, one of [cls, last_layer_mean].",
        choices=["cls", "last_layer_mean", "last_layer_first_subword_mean"],
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=1024,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--token_task_margin",
        type=str,
        default="min",
        help=(
            "How to calculate uncertainty for token level tasks. \
            Mean takes the mean margin across all tokens. Min takes the min margin across all tokens."
        ),
        choices=["mean", "min", "max", "mnlp"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to store the final model.",
        required=True,
    )
    parser.add_argument(
        "--knn_list",
        type=str,
        default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192",
        help="Comma separated list of number of neighbors to use.",
        required=False,
    )

    # Other arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to cache directory.",
    )

    args = parser.parse_args()

     # Override source and target languages if source and target dataset paths are provided
    if args.source_dataset_path is not None:
        args.source_languages = None
    if args.target_dataset_path is not None:
        args.target_languages = None

    return args

def calculate_embeddings(inputs, device, model, dataset, embedding_method):
    encoding = inputs
    encoding = {k:v.to(device) for k, v in encoding.items()}
    logits = None
    start_logits = None
    end_logits = None
    if dataset in ["udpos", "PAN-X"]:
        outputs = model(**encoding, output_hidden_states=True)
        if embedding_method == "last_layer_first_subword_mean":
            mask = encoding['labels']!=-100
            mask = mask.unsqueeze(-1).expand(outputs.hidden_states[-1].size()).float()
            denominator = torch.clamp(mask.sum(dim=1), min=1e-9)
            masked_hidden_states = outputs.hidden_states[-1] * mask
            embeddings = torch.sum(masked_hidden_states, dim=1) / denominator
        elif embedding_method == "cls":
            embeddings = outputs.hidden_states[-1][:,0,:]
        elif embedding_method == "last_layer_mean":
            mask = encoding['attention_mask']
            mask = mask.unsqueeze(-1).expand(outputs.hidden_states[-1].size()).float()
            denominator = torch.clamp(mask.sum(dim=1), min=1e-9)
            masked_hidden_states = outputs.hidden_states[-1] * mask
            embeddings = torch.sum(masked_hidden_states, dim=1) / denominator
        logits = outputs.logits # shape: (batch_size, sequence_length, num_labels)
    else:
        outputs = model(**encoding, output_hidden_states=True)
        if embedding_method == "cls":
            embeddings = outputs.hidden_states[-1][:,0,:]
        elif embedding_method == "last_layer_mean":
            mask = encoding['attention_mask']
            mask = mask.unsqueeze(-1).expand(outputs.hidden_states[-1].size()).float()
            denominator = torch.clamp(mask.sum(dim=1), min=1e-9)
            masked_hidden_states = outputs.hidden_states[-1] * mask
            embeddings = torch.sum(masked_hidden_states, dim=1) / denominator
        if dataset in ["tydiqa"]:
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            logits = [start_logits,end_logits]
        else:
            logits = outputs.logits # shape: (batch_size, num_labels)
    return embeddings, logits


def get_embeddings_and_uncertainty(
        dataloader,
        model,
        accelerator, 
        args, 
        embeddings, 
        logits, 
        loss, 
        uncertainty_margin
    ):
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()
    loss_fct = CrossEntropyLoss(reduction='none')

    if args.task_type in ['qa']:
        start_logits, end_logits = logits

    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            batch_size = batch['input_ids'].shape[0]
            batch_embeddings, batch_logits = calculate_embeddings(batch, accelerator.device, model, args.dataset_name, args.embedding_method)
            embeddings = torch.cat((embeddings, batch_embeddings))
            batch_size = batch_embeddings.shape[0]

             # Concatenate logits for each batch (except MT for which this is very large)
            if args.task_type in ['qa']:
                start_logits = torch.cat((start_logits, batch_logits[0]))
                end_logits = torch.cat((end_logits, batch_logits[1]))
            elif args.task_type in ['token', 'sequence']:
                logits = torch.cat((logits, batch_logits))

            # Find uncertainty margin by subtracting the log softmax of most probable class from the log softmax of the second most probable class
            if args.task_type in ['token']:
                # Calculate the cross entropy loss for each example and pass mask to ignore padding tokens
                batch_loss_per_token = loss_fct(batch_logits.view(-1, model.config.num_labels), batch['labels'].view(-1)).view(batch_size, args.max_seq_length)
                mask = batch_loss_per_token!=0
                batch_loss = torch.sum(batch_loss_per_token, dim=1) / mask.sum(dim=1)
                loss = torch.cat((loss, batch_loss))
                # Calculate source uncertainty margin
                batch_log_softmax = torch.nn.functional.log_softmax(batch_logits, dim=2)
                mask = batch["labels"] != -100
                batch_log_softmax = batch_log_softmax*mask.unsqueeze(2)
                batch_log_softmax_sorted, _ = torch.sort(batch_log_softmax, dim=2, descending=True)
                # For each example, find the mean of the difference between the log softmax of the most probable class and the log softmax of the second most probable class
                uncertainty_margin_sample = batch_log_softmax_sorted[:,:,0] - batch_log_softmax_sorted[:,:,1]
                if args.token_task_margin == "min":
                    margin_mask = uncertainty_margin_sample != 0
                    masked_uncertainty_margin_sample = uncertainty_margin_sample.clone()
                    masked_uncertainty_margin_sample[~margin_mask] = float('inf')
                    uncertainty_margin_sample_min = torch.min(masked_uncertainty_margin_sample, dim=1)[0]
                    uncertainty_margin = torch.cat((uncertainty_margin, uncertainty_margin_sample_min))
                elif args.token_task_margin == "mean":
                    uncertainty_margin_sample_mean = torch.sum(uncertainty_margin_sample, dim=1) / torch.sum(mask, dim=1)
                    uncertainty_margin = torch.cat((uncertainty_margin, uncertainty_margin_sample_mean))
                elif args.token_task_margin == "max":
                    uncertainty_margin_sample_max = torch.max(uncertainty_margin_sample, dim=1)[0]
                    uncertainty_margin = torch.cat((uncertainty_margin, uncertainty_margin_sample_max))
                elif args.token_task_margin == "mnlp":
                    uncertainty_margin_sample_mnlp = torch.sum(batch_log_softmax_sorted[:,:,0], dim=1) / torch.sum(mask, dim=1)
                    uncertainty_margin = torch.cat((uncertainty_margin, uncertainty_margin_sample_mnlp))
            
            elif args.task_type in ['sequence']:
                batch_log_softmax = torch.nn.functional.log_softmax(batch_logits, dim=1)
                batch_log_softmax_sorted, _ = torch.sort(batch_log_softmax, dim=1, descending=True)
                uncertainty_margin = torch.cat((uncertainty_margin, batch_log_softmax_sorted[:,0] - batch_log_softmax_sorted[:,1]))
                batch_loss = loss_fct(batch_logits.view(-1, model.config.num_labels), batch['labels'].view(-1))
                loss = torch.cat((loss, batch_loss))
            
            elif args.task_type in ['qa']:
                start_logits =  torch.nn.functional.log_softmax(batch_logits[0], dim=1)
                end_logits = torch.nn.functional.log_softmax(batch_logits[1], dim=1)
                if args.qa_uncertainty_method == "logits":
                    start_logits_max = torch.max(start_logits, dim=1)
                    end_logits_max = torch.max(end_logits, dim=1)
                    batch_uncertainty = start_logits_max.values + end_logits_max.values
                elif args.qa_uncertainty_method == "margin":
                    start_logits_sorted, _ = torch.sort(start_logits, dim=1, descending=True)
                    end_logits_sorted, _ = torch.sort(end_logits, dim=1, descending=True)
                    batch_uncertainty = (start_logits_sorted[:,0] - start_logits_sorted[:,1]) + (end_logits_sorted[:,0] - end_logits_sorted[:,1])
                uncertainty_margin = torch.cat((uncertainty_margin, batch_uncertainty))
                batch_loss = loss_fct(batch_logits[0].view(-1, args.max_seq_length), batch['start_positions'].view(-1)) + loss_fct(batch_logits[1].view(-1, args.max_seq_length), batch['end_positions'].view(-1))
                loss = torch.cat((loss, batch_loss))

            if step%100==0:
                logger.info(f'completed batch {step}')
    return embeddings, loss, uncertainty_margin


def main():
    args = parse_args()

    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    level = logging.INFO if accelerator.is_main_process else logging.ERROR

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )
    logger.info(accelerator.state, main_process_only=False)

    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    datasets.utils.logging.disable_propagation()
    datasets.disable_progress_bar()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
      if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Read in dataset config yaml file
    with open(args.dataset_config_file, "r") as f:
        dataset_config = yaml.safe_load(f)[args.dataset_name]
    
    remove_columns = dataset_config["remove_columns"]
    task_type = dataset_config["task_type"]
    if task_type not in ["qa"]:
        label_names = dataset_config["label_names"]
        num_labels = len(label_names)
        id2label = {i: label for i, label in enumerate(label_names)}
        label2id = {label: i for i, label in enumerate(label_names)}
    
     # Initialize config and tokenizer
    if task_type not in ["qa"]:
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
        do_lower_case=False,
        cache_dir=args.cache_dir,
        use_fast=not args.use_slow_tokenizer,
    )

    # Initialize model 
    if task_type in ["token"]:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
        )
    elif task_type in ["sequence"]:	
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
        )
    elif task_type in ["qa"]:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir,
        )

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if args.pad_to_max_length:
        data_collator = default_data_collator
    elif args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Load train or source dataset
    if args.source_languages is not None:
        source_languages = args.source_languages.split(",")
        logger.info(f"Loading training dataset from {args.dataset_name}")
        source_datasets = []
        for source_language in source_languages:
            # Get source language and dataset name
            if args.dataset_name in ["udpos", "PAN-X"]:
                source_language = args.dataset_name + "." + source_language
                dataset_name = "xtreme"
            else:
                dataset_name = args.dataset_name
        
    
            if args.dataset_name == "tydiqa":
                source_dataset = get_tydiqa_dataset(language=source_language, split="train", cache_dir=args.cache_dir)
            else:
                source_dataset = load_dataset(dataset_name, source_language, split="train", cache_dir=args.cache_dir)
        
            if args.debug:
                source_dataset = source_dataset.select(range(1000))

            # Deduplicate train dataset
            with accelerator.main_process_first():
                if args.dataset_name in ["udpos", "PAN-X"]:
                    deduplicated_source_dataset = deduplicate_dataset(source_dataset, check_column="tokens")
                else:
                    deduplicated_source_dataset = source_dataset

            # Select a subset of the train dataset if needed
            if args.per_language_subset_size:
                raw_source_dataset = deduplicated_source_dataset.select(range(args.per_language_subset_size))
            else:
                raw_source_dataset = deduplicated_source_dataset
            source_datasets.append(raw_source_dataset)

        source_dataset = concatenate_datasets(source_datasets)
    else:
        logger.info(f"Loading training dataset from {args.source_dataset_path}")
        source_dataset = load_from_disk(args.source_dataset_path)

    logger.info(f"Loaded {len(source_dataset)} training examples")
    with accelerator.main_process_first():
        processed_source_dataset = source_dataset.map(
            partial(preprocess_datasets,
            args=args,
            remove_columns=remove_columns,
            tokenizer=tokenizer,
            dataset=args.dataset_name,
            padding="max_length" if args.pad_to_max_length else False,
            max_seq_length=args.max_seq_length),
            batched=True,
            remove_columns=remove_columns,
        )

        if args.dataset_name in ['tydiqa']:
            processed_source_dataset = processed_source_dataset.remove_columns(remove_columns)

    # Load test splits of target languages
    if args.target_languages is not None:
        logger.info(f"Loading test splits of target languages {args.target_languages}")
        test_datasets = []
        raw_test_datasets = []
        for target_language in args.target_languages.split(","):
            if args.dataset_name in ["udpos", "PAN-X"]:
                target_language = args.dataset_name + "." + target_language
                dataset_name = "xtreme"
            else:
                dataset_name = args.dataset_name

            if args.dataset_name == "tydiqa":
                test_dataset = get_tydiqa_dataset(language=source_language, split="test", cache_dir=args.cache_dir)
            else:
                test_dataset = load_dataset(dataset_name, source_language, split="test", cache_dir=args.cache_dir)

            with accelerator.main_process_first():
                processed_test_dataset = test_dataset.map(
                    partial(preprocess_datasets,
                    args=args,
                    remove_columns=remove_columns,
                    tokenizer=tokenizer,
                    dataset=args.dataset_name,
                    padding="max_length" if args.pad_to_max_length else False,
                    max_seq_length=args.max_seq_length),
                    batched=True,
                    remove_columns=remove_columns,
                )
            test_datasets.append(processed_test_dataset)
            raw_test_datasets.append(test_dataset)
        processed_target_dataset = concatenate_datasets(test_datasets)
        target_dataset = concatenate_datasets(raw_test_datasets)
    else:
        logger.info(f"Loading test dataset from {args.target_dataset_path}")
        processed_target_dataset = load_from_disk(args.target_dataset_path)
    logger.info(f"Loaded {len(processed_target_dataset)} test examples")

    if args.dataset_name in ['tydiqa']:
        processed_target_dataset = processed_target_dataset.remove_columns(remove_columns)

    # Get embeddings of source and target datasets
    logger.info("Getting embeddings of source and target datasets")
    source_dataloader = DataLoader(processed_source_dataset, collate_fn=data_collator, batch_size=args.inference_batch_size, shuffle=False)
    target_dataloader = DataLoader(processed_target_dataset, collate_fn=data_collator, batch_size=args.inference_batch_size, shuffle=False)

    # Embeddings
    source_embeddings = torch.empty((0, model.config.hidden_size), dtype=torch.float32).to(accelerator.device)
    target_embeddings = torch.empty((0, model.config.hidden_size), dtype=torch.float32).to(accelerator.device)

    # TODO: add tydiqa info
    if task_type in ["token"]:
        source_logits = torch.empty([0, args.max_seq_length, model.config.num_labels]).to(accelerator.device)
        target_logits = torch.empty([0, args.max_seq_length, model.config.num_labels]).to(accelerator.device)
    elif task_type in ["qa"]:
        source_start_logits = torch.empty([0, args.max_seq_length]).to(accelerator.device)
        source_end_logits = torch.empty([0, args.max_seq_length]).to(accelerator.device)
        target_start_logits = torch.empty([0, args.max_seq_length]).to(accelerator.device)
        target_end_logits = torch.empty([0, args.max_seq_length]).to(accelerator.device)

        source_logits = (source_start_logits, source_end_logits)
        target_logits = (target_start_logits, target_end_logits)
    else:
        source_logits = torch.empty((0, model.config.num_labels), dtype=torch.float32).to(accelerator.device)
        target_logits = torch.empty((0, model.config.num_labels), dtype=torch.float32).to(accelerator.device)

    # Source loss and uncertainty margin
    source_loss = torch.empty([0], dtype=torch.float32).to(accelerator.device)
    source_uncertainty_margin = torch.empty((0), dtype=torch.float32).to(accelerator.device)

    # Target loss and uncertainty margin
    target_loss = torch.empty([0], dtype=torch.float32).to(accelerator.device)
    target_uncertainty_margin = torch.empty((0), dtype=torch.float32).to(accelerator.device)

    # Get embeddings of source and target datasets and calculate uncertainty margin
    logger.info("Getting embeddings of source and target datasets and calculating uncertainty margin")
    source_embeddings, source_loss, source_uncertainty_margin = get_embeddings_and_uncertainty(
        source_dataloader, model, accelerator, args, source_embeddings, source_logits, source_loss, source_uncertainty_margin)

    target_embeddings, target_loss, target_uncertainty_margin = get_embeddings_and_uncertainty(
        target_dataloader, model, accelerator, args, target_embeddings, target_logits, target_loss, target_uncertainty_margin)

    if args.task_type in ["qa"]:
        original_to_processed = collections.defaultdict(list)
        for idx, example_id in enumerate(processed_source_dataset['example_id']):
            original_to_processed[example_id].append(idx)
        
        original_to_processed_target = collections.defaultdict(list)
        for idx, example_id in enumerate(processed_target_dataset['example_id']):
            original_to_processed_target[example_id].append(idx)
        
        source_uncertainity = {}
        target_uncertainity = {}
        src_loss = {}
        tgt_loss = {}
        source_average_cls_embedding = {}
        target_average_cls_embedding = {}
        for ind in original_to_processed.keys():
            uncertainity = source_uncertainty_margin[original_to_processed[ind]]
            uncertainity = torch.max(uncertainity)
            src_loss[ind] = torch.max(source_loss[original_to_processed[ind]])
            source_uncertainity[ind]=uncertainity 

            # take the average of CLS Embeddings of all features within an example
            source_average_cls_embedding[ind] = torch.mean(source_embeddings[original_to_processed[ind]],0).reshape(1,-1)
            
        
        for ind in original_to_processed_target.keys():
            uncertainity = target_uncertainty_margin[original_to_processed_target[ind]]
            uncertainity = torch.max(uncertainity)
            target_uncertainity[ind]=uncertainity
            tgt_loss[ind] = torch.max(target_loss[original_to_processed_target[ind]])

            # take the average of CLS Embeddings of all features within an example
            target_average_cls_embedding[ind] = torch.mean(target_embeddings[original_to_processed_target[ind]],0).reshape(1,-1)

        temp_source_uncertainity = []
        temp_target_uncertainity = []
        temp_source_loss = []
        temp_target_loss = []
        temp_source_embeddings = torch.empty([0, model.config.hidden_size]).to(accelerator.device)
        temp_target_embeddings = torch.empty([0, model.config.hidden_size]).to(accelerator.device)
        for i in source_dataset['id']:
            temp_source_uncertainity.append(source_uncertainity[i].item())
            temp_source_embeddings = torch.cat((temp_source_embeddings,source_average_cls_embedding[i]))
            temp_source_loss.append(src_loss[i].item())

        for i in target_dataset['id']:
            temp_target_uncertainity.append(target_uncertainity[i].item())
            temp_target_embeddings = torch.cat((temp_target_embeddings,target_average_cls_embedding[i]))
            temp_target_loss.append(tgt_loss[i].item())
    
        source_uncertainty_margin = torch.tensor(temp_source_uncertainity).to(accelerator.device)
        target_uncertainty_margin = torch.tensor(temp_target_uncertainity).to(accelerator.device)
        source_loss = torch.tensor(temp_source_loss).to(accelerator.device)
        target_loss = torch.tensor(temp_target_loss).to(accelerator.device)
        source_embeddings = temp_source_embeddings
        target_embeddings = temp_target_embeddings
        
        del temp_source_uncertainity, source_uncertainity, original_to_processed, source_average_cls_embedding, target_average_cls_embedding, temp_target_embeddings, original_to_processed_target
    
    # Calculate L2 distance of target with source
    logger.info("Calculating L2 distance of target with source")
    dist_matrix = torch.cdist(target_embeddings.cpu(), source_embeddings.cpu(), p=2)
    dist_matrix = dist_matrix.to(accelerator.device)

    mean_dists = torch.mean(dist_matrix, dim=1)
    logger.info("Mean dists shape: {}".format(mean_dists.shape))

    # Clear files where correlation coefficients will be written
    open(os.path.join(args.output_dir, "correlation_target_uncertainty_knn_uncertainty"), "w").close()
    open(os.path.join(args.output_dir, "correlation_target_loss_knn_uncertainty"), "w").close()

    # Get k nearest neighbours for each target example
    logger.info("Getting k nearest neighbours for each target example")
    knn_list = args.knn_list.split(",")
    knn_list = [int(k) for k in knn_list]
    with accelerator.main_process_first():
        for k in knn_list:
            logger.info("Getting k={} nearest neighbours".format(k))
            _, k_nearest_neighbours = torch.topk(dist_matrix, k=k, dim=1, largest=False)
            logger.info("k_nearest_neighbours shape: {}".format(k_nearest_neighbours.shape))
            # Calculate average source loss of k nearest neighbours
            logger.info("Calculating average source uncertainty of k nearest neighbours by finding margin values")
            # We need to repeat the source logits to match the shape of k_nearest_neighbours
            source_uncertainty_margin_repeated = source_uncertainty_margin.repeat(k_nearest_neighbours.shape[0], 1)
            source_loss_repeated = source_loss.repeat(k_nearest_neighbours.shape[0], 1)
            k_nearest_neighbours_uncertainty_margin = torch.gather(source_uncertainty_margin_repeated, 1, k_nearest_neighbours)
            k_nearest_neighbours_uncertainty_margin = torch.mean(k_nearest_neighbours_uncertainty_margin, dim=1)
            k_nearest_neighbours_loss = torch.gather(source_loss_repeated, 1, k_nearest_neighbours)
            k_nearest_neighbours_loss = torch.mean(k_nearest_neighbours_loss, dim=1)
            logger.info("k_nearest_neighbours_uncertainty_margin shape: {}".format(k_nearest_neighbours_uncertainty_margin.shape))
            logger.info("k_nearest_neighbours_loss shape: {}".format(k_nearest_neighbours_loss.shape))

            # Find correlation coefficient between target uncertainty margin and k nearest neighbours uncertainty margin
            logger.info("Finding correlation coefficient between target uncertainty margin and k nearest neighbours uncertainty margin")
            r, p = scipy.stats.pearsonr(target_uncertainty_margin.cpu().numpy(), k_nearest_neighbours_uncertainty_margin.cpu().numpy())
            # Write to file
            with open(os.path.join(args.output_dir, "correlation_target_uncertainty_knn_uncertainty"), "a") as f:
                f.write("k={}, r={}, p={} \n".format(k, r, p))
            logger.info("Correlation coefficient between target uncertainty margin and k nearest neighbours uncertainty margin, k={}: {}, with p {}".format(k, r, p))
            logger.info("Finding correlation coefficient between target loss and k nearest neighbours uncertainty margin")
            r, p = scipy.stats.pearsonr(target_loss.cpu().numpy(), k_nearest_neighbours_uncertainty_margin.cpu().numpy())
            # Write to file
            with open(os.path.join(args.output_dir, "correlation_target_loss_knn_uncertainty"), "a") as f:
                f.write("k={}, r={}, p={} \n".format(k, r, p))
            
            # Plot target uncertainty margin vs k nearest neighbours uncertainty margin
            logger.info("Plotting target margin vs k nearest neighbours margin")
            plt.scatter(target_uncertainty_margin.cpu(), k_nearest_neighbours_uncertainty_margin.cpu())
            plt.xlabel("Target margin")
            plt.ylabel("k_nn uncertainty margin")
            plt.title("Target margin vs k_nn uncertainty margin, k={}".format(k))
            plt.legend()
            plt.savefig(os.path.join(args.output_dir, "all_target_uncertainty_vs_knn_uncertainty_margin_k_{}.png".format(k)))
            plt.clf()

    # Log sample target loss values
    logger.info("Sample target loss values: {}".format(target_loss[:10]))
    logger.info("Target loss shape: {}".format(target_loss.shape))

    # Scatter plot of loss vs mean distance
    logger.info("Plotting loss vs mean distance")
    plt.scatter(target_loss.cpu(), mean_dists.cpu())
    plt.xlabel("Target loss")
    plt.ylabel("Mean distance")
    plt.title("Target loss vs mean distance")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "all_target_loss_vs_mean_distance.png"))
    plt.clf()

    del dist_matrix, mean_dists, source_embeddings, target_embeddings, \
        source_logits, target_logits, source_loss, target_loss, source_uncertainty_margin, target_uncertainty_margin

    
if __name__ == "__main__":
    main()