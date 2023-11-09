import torch
from accelerate.logging import get_logger
import collections

import os
from transformers import default_data_collator
from datasets import concatenate_datasets
import logging
from torch.utils.data import DataLoader
import numpy as np
import gc
import math
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from helper.data_utils import UDPOS_ID_MAP, create_output_dirs
from google.cloud import translate_v2 as translate

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"src/helper/googlekey.json"
translate_client = translate.Client()

logger = get_logger(__name__)

logging.basicConfig(level=logging.INFO)

def compute_uncertainty(args, task_type, logits, batch, model):
    # Find uncertainty margin for token level tasks 
    if task_type in ['token']:
        # Calculate the cross entropy loss  each example and pass mask to ignore padding tokens
        batch_log_softmax = torch.nn.functional.log_softmax(logits, dim=2)
        mask = batch["labels"] != -100
        batch_log_softmax = batch_log_softmax*mask.unsqueeze(2)
        batch_log_softmax_sorted, _ = torch.sort(batch_log_softmax, dim=2, descending=True)
        uncertainty_margin_sample = batch_log_softmax_sorted[:,:,0] - batch_log_softmax_sorted[:,:,1]
        if args.token_task_margin == "min":
            margin_mask = uncertainty_margin_sample!=0
            masked_uncertainty_margin_sample = uncertainty_margin_sample.clone()
            masked_uncertainty_margin_sample[~margin_mask] = float('inf')
            uncertainty_margin_sample_min = torch.min(masked_uncertainty_margin_sample, dim=1)[0]
            uncertainty = uncertainty_margin_sample_min
        elif args.token_task_margin == "mean":
            uncertainty_margin_sample_mean = torch.sum(uncertainty_margin_sample, dim=1) / mask.sum(dim=1)
            uncertainty = uncertainty_margin_sample_mean
        elif args.token_task_margin == "max":
            uncertainty_margin_sample_max = torch.max(uncertainty_margin_sample, dim=1)[0]
            uncertainty = uncertainty_margin_sample_max
        elif args.token_task_margin == "mnlp":
            uncertainty_margin_sample_mnlp = torch.sum(batch_log_softmax_sorted[:,:,0], dim=1) / torch.sum(mask, dim=1)
            uncertainty = uncertainty_margin_sample_mnlp
    
    # Find uncertainty value for classification tasks by subtracting the log softmax of the top 2 logits; lower margin means higher uncertainty
    elif task_type in ['sequence']:
        batch_log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
        batch_log_softmax_sorted, _ = torch.sort(batch_log_softmax, dim=1, descending=True)
        uncertainty = batch_log_softmax_sorted[:,0] - batch_log_softmax_sorted[:,1]
    
    # Find uncertainty value for QA tasks by adding start and end logits; lower the final value higher the uncertainty
    elif task_type in ['qa']:
        start_logits =  torch.nn.functional.log_softmax(logits[0], dim=1)
        end_logits = torch.nn.functional.log_softmax(logits[1], dim=1)
        if args.qa_uncertainty_method == "logits":
            start_logits_max = torch.max(start_logits, dim=1)
            end_logits_max = torch.max(end_logits, dim=1)
            uncertainty = start_logits_max.values + end_logits_max.values
        elif args.qa_uncertainty_method == "margin":
            start_logits_sorted, _ = torch.sort(start_logits, dim=1, descending=True)
            end_logits_sorted, _ = torch.sort(end_logits, dim=1, descending=True)
            uncertainty = (start_logits_sorted[:,0] - start_logits_sorted[:,1]) + (end_logits_sorted[:,0] - end_logits_sorted[:,1])
    
    # Find uncertainty value of MT tasks by calculating the cross entropy loss for each example and pass mask to ignore padding tokens
    elif task_type in ['mt']:
        loss_fct = CrossEntropyLoss(reduction='none')
        batch_size = logits.shape[0]
        batch_loss_per_token = loss_fct(logits.view(-1, args.vocab_size), batch['labels'].view(-1)).view(batch_size, args.max_seq_length)
        mask = batch_loss_per_token!=0
        batch_loss = torch.sum(batch_loss_per_token, dim=1) / mask.sum(dim=1)
        # Taking negative of loss to make it consistent with other tasks where higher uncertainty is associated with lower margins
        uncertainty = -batch_loss
        del logits
    
    return uncertainty

def process_batch(inputs, task_type, device, model, dataset, embedding_method):
    encoding = inputs
    encoding = {k:v.to(device) for k, v in encoding.items()}
    logits = None
    if task_type in ["token"]:
        outputs = model(**encoding, output_hidden_states=True)
        # Get average of all token embeddings from last hidden layer for unmasked tokens only
        mask = encoding['labels']!=-100
        mask = mask.unsqueeze(-1).expand(outputs.hidden_states[-1].size()).float()
        denominator = torch.clamp(mask.sum(dim=1), min=1e-9)
        masked_hidden_states = outputs.hidden_states[-1] * mask
        embeddings = torch.sum(masked_hidden_states, dim=1) / denominator
        logits = outputs.logits # shape: (batch_size, sequence_length, num_labels)
    elif task_type in ["qa", "sequence"]:
        outputs = model(**encoding, output_hidden_states=True)
        if embedding_method == "cls":
            embeddings = outputs.hidden_states[-1][:,0,:]
        elif embedding_method == "last_layer_mean":
            mask = encoding['attention_mask']
            mask = mask.unsqueeze(-1).expand(outputs.hidden_states[-1].size()).float()
            denominator = torch.clamp(mask.sum(dim=1), min=1e-9)
            masked_hidden_states = outputs.hidden_states[-1] * mask
            embeddings = torch.sum(masked_hidden_states, dim=1) / denominator
        if task_type in ["qa"]:
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            logits = [start_logits, end_logits]
        else:
            logits = outputs.logits # shape: (batch_size, num_labels)
    elif task_type in ["mt"]:
        outputs = model(**encoding)
        if embedding_method == "cls":
            embeddings = outputs.encoder_last_hidden_state[:,0,:] # shape: (batch_size, hidden_size)
        elif embedding_method == "last_layer_mean":
            mask = encoding['attention_mask']
            mask = mask.unsqueeze(-1).expand(outputs.encoder_last_hidden_state.size()).float()
            denominator = torch.clamp(mask.sum(dim=1), min=1e-9)
            masked_hidden_states = outputs.encoder_last_hidden_state * mask
            embeddings = torch.sum(masked_hidden_states, dim=1) / denominator
        logits = outputs.logits # shape: (batch_size, sequence_length, vocab_size)
    return embeddings, logits


def process_embeddings_for_qa(raw_train_dataset, processed_train_dataset, raw_target_dataset, processed_target_dataset, train_embeddings, target_embeddings, train_uncertainty, accelerator):
    train_indices_map = collections.defaultdict(list)
    target_indices_map = collections.defaultdict(list)
    for idx, example_id in enumerate(processed_train_dataset['example_id']):
        train_indices_map[example_id].append(idx)
        
    for idx, example_id in enumerate(processed_target_dataset['example_id']):
        target_indices_map[example_id].append(idx)
        
    train_uncertainty_map = {}
    train_embedding_map = {}
    target_embedding_map = {}
    for ind in train_indices_map.keys():
        uncertainty = torch.max(train_uncertainty[train_indices_map[ind]])
        train_uncertainty_map[ind] = uncertainty
        
        # take the average of CLS Embeddings of all features within an example
        train_embedding_map[ind] = torch.mean(train_embeddings[train_indices_map[ind]], 0).reshape(1, -1)
        
    for ind in target_indices_map.keys():
        # take the average of CLS Embeddings of all features within an example
        target_embedding_map[ind] = torch.mean(target_embeddings[target_indices_map[ind]], 0).reshape(1, -1)

    train_uncertainty_list = []
    train_embedding_list = []
    target_embedding_list = []
    for ind in raw_train_dataset['id']:
        train_uncertainty_list.append(train_uncertainty_map[ind])
        train_embedding_list.append(train_embedding_map[ind])

    for ind in raw_target_dataset['id']:
        target_embedding_list.append(target_embedding_map[ind])
    
    train_embeddings = torch.cat(train_embedding_list, 0)
    target_embeddings = torch.cat(target_embedding_list, 0)
    train_uncertainty = torch.tensor(train_uncertainty_list).to(accelerator.device)

    train_embeddings = accelerator.gather(train_embeddings)
    target_embeddings = accelerator.gather(target_embeddings)
    train_uncertainty = accelerator.gather(train_uncertainty)

    return train_embeddings, target_embeddings, train_uncertainty


def get_embeddings(args, task_type, processed_dataset, split, model, accelerator, iteration, save_embeddings_path, save_embeddings=False):
    # Initialize empty lists instead of tensors, as each GPU will compute a part of the final tensor
    embeddings_list = []
    logits_list = []
    uncertainty_list = []

    # Initialize start_logits and end_logits lists for 'qa' task type
    if task_type in ['qa']:
        start_logits_list = []
        end_logits_list = []
    
    # Preserve the order of the dataset when batching and getting embeddings
    # TODO: check if we need to do this and why
    if args.dataset_name in ['tydiqa']:
        for i in ['offset_mapping', 'example_id', 'language']:
            if i in processed_dataset.features:
                processed_dataset = processed_dataset.remove_columns([i])

    # Initialize dataloader and bring in eval mode
    dataloader = DataLoader(processed_dataset,
                            collate_fn=default_data_collator,
                            batch_size=args.inference_batch_size,
                            shuffle=False)
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    # Get embeddings and logits for each batch
    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            batch_embeddings, batch_logits = process_batch(batch, task_type, accelerator.device, model, args.dataset_name, args.embedding_method)
            # Add the computed batch embeddings and logits to the respective lists
            embeddings_list.append(batch_embeddings.cpu())
            # Concatenate logits for each batch (except MT for which this is very large)
            if task_type in ['qa']:
                start_logits_list.append(batch_logits[0].cpu())
                end_logits_list.append(batch_logits[1].cpu())
            elif task_type in ['token', 'sequence']:
                logits_list.append(batch_logits.cpu())

            # Compute the uncertainty based on the task type and add it to the uncertainty list
            uncertainty = compute_uncertainty(args, task_type, batch_logits, batch, model)
            uncertainty_list.append(uncertainty.cpu())

            # Log every 100 batches
            if step % 100 == 0:
                logger.info('completed batch %s', step)
                logger.info('Current batch size: %s', len(batch))
                logger.info('Current embedding size:  %s', batch_embeddings.size())
    
    # After the loop ends, concatenate the tensors in the lists and gather them from all GPUs
    embeddings = torch.cat(embeddings_list, dim=0).to(accelerator.device)
    uncertainty = torch.cat(uncertainty_list, dim=0).to(accelerator.device)
    if task_type not in ['mt']:
        logits = torch.cat(logits_list, dim=0) if task_type not in ['qa'] else [torch.cat(start_logits_list, dim=0), torch.cat(end_logits_list, dim=0)]
    else:
        logits = None
    
    # embeddings = accelerator.gather(embeddings)
    # uncertainty = accelerator.gather(uncertainty)
    # if task_type not in ['mt']:
    #     logits = accelerator.gather(logits)
    # else:
    #     logits = None
    
    # Check if we are in a multi-GPU environment and if so, remove duplicates from the last batch
    if accelerator.num_processes > 1:
        dataset_size = len(processed_dataset)
        if embeddings.shape[0] > dataset_size:
            embeddings = embeddings[:dataset_size]
            uncertainty = uncertainty[:dataset_size]
            if logits is not None:
                if task_type in ['qa']:
                    logits[0] = logits[0][:dataset_size]
                    logits[1] = logits[1][:dataset_size]
                else:
                    logits = logits[:dataset_size]


    # Save embeddings to disk
    if save_embeddings:
        with accelerator.main_process_first():
            if not os.path.exists(save_embeddings_path + '/iter_' + str(iteration)):
                os.makedirs(save_embeddings_path + '/iter_' + str(iteration), exist_ok=True)
            torch.save(embeddings, save_embeddings_path + '/iter_' + str(iteration) + '/' + str(split) + '_embeddings.pt')
            logger.info("Saved embeddings for split: %s", split[:5])

    return embeddings, logits, uncertainty


def get_indices(args,
                raw_train_dataset,
                raw_train_datasets,
                train_embeddings,
                target_embeddings,
                train_uncertainty,
                budget,
                strategy,
                iteration,
                previously_selected_indices,
                accelerator):
    ################################# Code for all strategies begins here #################################
    ########################################### AVERAGE-DIST ##############################################

    ################################# TODO: ask if better way to deal with larger train embedding matrices #####################################
    if strategy=="average_dist":
        print("Diversity mean strategy")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if len(train_embeddings) > args.max_train_batch_size:
            mean_budget_dists_batches = torch.empty([0]).to(device)
            mean_budget_sim_batches = torch.empty([0]).to(device)
            topk_indices_batches = []

            for i in range(0, len(train_embeddings), args.max_train_batch_size):
                print("Batch: ", i)
                end_idx = min(i + args.max_train_batch_size, len(train_embeddings))
                # Calculate distance between train and target embeddings
                dist_matrix_sample = torch.cdist(train_embeddings[i:end_idx], target_embeddings, p=2).to(device)
                dist_matrix_sample_mean = torch.mean(dist_matrix_sample, dim=1)
                k = min(budget, len(dist_matrix_sample_mean))
        
                # Get topk samples with least distance
                mean_budget_dists_sample, topk_indices_sample = torch.topk(dist_matrix_sample_mean, k, largest=False)
                mean_budget_dists_batches = torch.cat((mean_budget_dists_batches, mean_budget_dists_sample))
        
                adjusted_indices = topk_indices_sample + i
                topk_indices_batches.extend(adjusted_indices.tolist())
                
                # cosine similarity
                # cosine_sim_sample = F.cosine_similarity(train_embeddings[i:end_idx], target_embeddings, dim=1).to(device)
                # cosine_sim_sample_mean = torch.mean(cosine_sim_sample, dim=1)
                # # Get top k points having highest mean cosine similarity
                # mean_budget_sim_sample, topk_indices_cosine_sim_sample = torch.topk(cosine_sim_sample_mean, k, largest=True)
                # mean_budget_sim_batches = torch.cat((mean_budget_sim_batches, mean_budget_sim_sample))
                # adjusted_indices = topk_indices_cosine_sim_sample + i
                # topk_indices_batches.extend(adjusted_indices.tolist())

            # Select the top 'budget' mean distances and indices
            _, topk_indices_mean_dist = torch.topk(mean_budget_dists_batches, budget, largest=False)
            topk_indices = [topk_indices_batches[idx] for idx in topk_indices_mean_dist.tolist()]
            topk_indices = torch.tensor(topk_indices).to(device)

            # _, topk_indices_mean_sim = torch.topk(mean_budget_sim_batches, budget, largest=True)
            # topk_indices = [topk_indices_batches[idx] for idx in topk_indices_mean_sim.tolist()]

        else:
            print("Single batch")
            dist_matrix = torch.cdist(train_embeddings, target_embeddings, p=2).to(accelerator.device)
            top_mean_distances, topk_indices = torch.topk(torch.mean(dist_matrix, dim=1), min(budget, dist_matrix.shape[0]), largest=False)

            logger.info(f"Min mean dist: {top_mean_distances[0]}")
            logger.info(f"Max mean dist: {top_mean_distances[-1]}")
            # sim_matrix = F.cosine_similarity(train_embeddings, target_embeddings, dim=1).to(device)
            # top_mean_sim, topk_indices = torch.topk(torch.mean(sim_matrix, dim=1), min(budget, sim_matrix.shape[0]), largest=True)

            # logger.info(f"Min mean sim: {top_mean_sim[0]}")
            # logger.info(f"Max mean sim: {top_mean_sim[-1]}")

        # Cast top_k indices to list of ints
        topk_indices = topk_indices.to(torch.int)
        selected_indices = previously_selected_indices.copy()
        for i, index in enumerate(topk_indices.tolist()):
            if len(selected_indices) == budget:
                break
            if index not in selected_indices:
                selected_indices.append(index)
        
    
    ############################################## UNCERTAINTY ###################################################
    if strategy.startswith("uncertainty"):
        # Calculate uncertainty of train logits. Choose train samples with lowest average uncertainty.
        logger.info("Uncertainty strategy")
        # Select train samples with the lowest average margin / negative loss (for QA), to choose samples with highest uncertainty
        _, top_indices = torch.topk(train_uncertainty, min(budget, train_uncertainty.shape[0]), largest=False)
        selected_indices = previously_selected_indices.copy()
        for i in top_indices.tolist():
            if len(selected_indices) == budget:
                break
            if i not in selected_indices:
                selected_indices.append(i)
    
    ############################################## KNN-UNCERTAINTY ################################################

    ############################### TODO: ask if better way to deal with larger train embedding matrices ###########
    if strategy.startswith("knn_uncertainty"):
        # Calculate K nearest neighbors for each target and choose most uncertain train samples among them
        logger.info("KNN Uncertainty Margin strategy")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        max_nn = args.knn_max_neighbors  # maximum number of nearest neighbors to consider
        batch_size = args.max_target_batch_size  # size of each target batch

        # Initialize tensors to store sorted values and indices
        sorted_values = torch.empty([0, max_nn], dtype=torch.long).to(device)
        sorted_indices = torch.empty([0, max_nn], dtype=torch.long).to(device)

        # Check if either tensor exceeds its length threshold
        if len(train_embeddings) > args.max_train_batch_size or len(target_embeddings) > args.max_target_batch_size:
            for i in range(0, len(target_embeddings), batch_size):
                end_idx = min(i + batch_size, len(target_embeddings))
                dist_matrix_sample = torch.cdist(target_embeddings[i:end_idx], train_embeddings, p=2).to(device)

                values, indices = torch.sort(dist_matrix_sample, dim=1)
                sorted_values = torch.cat((sorted_values, values[:, :max_nn]))
                sorted_indices = torch.cat((sorted_indices, indices[:, :max_nn]))

                # sim_matrix_sample = F.cosine_similarity(target_embeddings[i:end_idx].unsqueeze(1), train_embeddings.unsqueeze(0), dim=2).to(device)

                # values, indices = torch.sort(sim_matrix_sample, dim=1, descending=True)
                # sorted_values = torch.cat((sorted_values, values[:, :max_nn]))
                # sorted_indices = torch.cat((sorted_indices, indices[:, :max_nn]))
        
        else:
            dist_matrix = torch.cdist(target_embeddings, train_embeddings, p=2).to(device)
            sorted_values, sorted_indices = torch.sort(dist_matrix, dim=1)
            # sim_matrix = F.cosine_similarity(target_embeddings, train_embeddings, dim=1).to(device)
            # sorted_values, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)

        k_value = args.k_value if args.k_value is not None else 1
        candidate_indices = previously_selected_indices.copy()

        while True:
            k_nn = sorted_indices[:, :k_value]
            all_k_nn = torch.unique(k_nn.reshape(-1))
            candidate_indices.extend(all_k_nn.tolist())
            
            if len(set(candidate_indices)) > budget:
                logger.info(f"k_value: {k_value}")
                logger.info(f"Length of candidate indices set: {len(candidate_indices)}")
                break

            if k_value > sorted_values.shape[1]:
                logger.info("k_value exceeds number of train samples")
                break
            
            k_value *= 2
            candidate_indices = previously_selected_indices.copy()

        logger.info("k_value: {}".format(k_value))

        all_k_nn_uncertainty = train_uncertainty[all_k_nn]
        _, sorted_indices_uncertainty = torch.sort(all_k_nn_uncertainty, descending=False)
        train_indices_sorted = all_k_nn[sorted_indices_uncertainty]

        selected_indices = previously_selected_indices.copy()
        for i in train_indices_sorted.tolist():
            if len(selected_indices) == budget:
                break
            if i not in selected_indices:
                selected_indices.append(i)
        logger.info("Selected indices shape: {}".format(len(selected_indices)))
    
    ############################################## RANDOM and GOLD ##############################################
    if strategy=="random" or strategy.startswith("gold"):
        logger.info("Random strategy")
        selected_indices = previously_selected_indices.copy()
        if budget > len(raw_train_dataset):
            selected_indices = list(range(len(raw_train_dataset)))
        else:
            if args.per_language_allocation_file is None:
                while len(selected_indices) < budget:
                        random_index = np.random.randint(0, len(raw_train_dataset))
                        if random_index not in selected_indices:
                            selected_indices.append(random_index)
            else:
                # read tsv file with per language allocation
                with open(args.per_language_allocation_file, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    fields = line.strip().split("\t")
                    dataset_config, iteration_number, allocation = fields
                    dataset, config = dataset_config.split(":")
                    iteration_number = int(iteration_number.split("_")[1])
                    if dataset == args.dataset_name and config == args.target_config_name and iteration_number == iteration:
                        per_language_allocation = allocation.split(",")
                        break

                per_language_allocation_each_round = {}
                for item in per_language_allocation:
                    language, count = item.split(":")
                    if args.dataset_name == "udpos":
                        language = UDPOS_ID_MAP[language]
                    per_language_allocation_each_round[language] = int(count)
                    logger.info("Language: {}, Count: {}".format(language, per_language_allocation_each_round[language]))

                train_dataset_size = 0
                if iteration_number is not None:
                    selected_indices = []
                for language, train_dataset in raw_train_datasets.items():
                    if language not in per_language_allocation_each_round:
                        train_dataset_size += len(train_dataset)
                        continue
                    else:
                        per_language_budget = int(per_language_allocation_each_round[language])
                        count = 0
                        while count < per_language_budget:
                            random_index = np.random.randint(train_dataset_size, train_dataset_size + len(train_dataset))
                            if random_index not in selected_indices:
                                    selected_indices.append(random_index)
                                    count += 1
                        train_dataset_size += len(train_dataset)
    
    ############################################## EGALITARIAN ###################################################
    if strategy=="egalitarian":
        logger.info("Egalitarian strategy")
        selected_indices = previously_selected_indices.copy()
        to_select = budget - len(selected_indices)
        budget_per_language = int(to_select/len(raw_train_datasets.values()))
        total_dataset_len = 0
        newly_selected_indices = []
        for i, dataset in enumerate(raw_train_datasets.values()):
            if len(dataset) <= math.floor(budget/len(raw_train_datasets.values())):
                for i in range(len(dataset)):
                    if total_dataset_len + i not in selected_indices:
                        newly_selected_indices.append(total_dataset_len + i)
            else:   
                while len(newly_selected_indices) < budget_per_language:
                    random_index = np.random.randint(total_dataset_len, total_dataset_len + len(dataset))
                    if random_index not in selected_indices:
                        newly_selected_indices.append(random_index)
                    
                selected_indices.extend(newly_selected_indices)
            newly_selected_indices = []
            total_dataset_len += len(dataset)
        if len(selected_indices) < budget:
            while len(selected_indices) < budget:
                random_index = np.random.randint(0, len(raw_train_dataset))
                if random_index not in selected_indices:
                    selected_indices.append(random_index)
                
                if len(selected_indices) >= len(raw_train_dataset):
                    break
    ###################################################################################################################
    
    return selected_indices


def save_dataset(
    args,
    save_dataset_path,
    save_embeddings_path, 
    raw_train_dataset, 
    selected_indices, 
    train_embeddings, 
    budget, 
    iteration
):
    save_dataset_path += '/iter_' + str(iteration)
    save_embeddings_path += '/iter_' + str(iteration)
    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path, exist_ok=True)
    if not os.path.exists(save_embeddings_path):
        os.makedirs(save_embeddings_path, exist_ok=True)
    
    # Save dataset and embeddings
    selected_train_dataset = raw_train_dataset.select(selected_indices)
    # get google translate translations of "source" column
    if args.task_type in ["mt"]:
        source_inputs = selected_train_dataset.to_pandas()["translation"].tolist()
        translations = []
        for source_input in source_inputs:
            input = source_input["source"]
            target = translate_client.translate(input, target_language="en")
            translations.append(target["translatedText"])
        selected_train_dataset = selected_train_dataset.add_column("gtrans", translations)

    # Save in csv format
    selected_train_dataset.to_csv(save_dataset_path + '/' + str(budget) + '.csv')
    # Save in arrow format
    selected_train_dataset.save_to_disk(save_dataset_path)
    logger.info("train dataset first sample: {}".format(selected_train_dataset[0]))

    # Save language distribution of selected dataset
    logger.info("selected train dataset first sample: {}".format(selected_train_dataset[0]))
    language_distribution = selected_train_dataset.to_pandas()["language"].value_counts()
    language_distribution.to_csv(save_dataset_path + '/' + str(budget) + '_language_distribution.csv')

    # Save embeddings
    if args.save_embeddings and args.compute_embeddings:
        selected_train_embeddings = train_embeddings[selected_indices]
        torch.save(selected_train_embeddings, save_embeddings_path + '/selected_train_' + str(budget) + '.pt')
    
    return selected_train_dataset


def get_train_dataset(args, 
                      task_type,
                      raw_train_datasets,
                      raw_target_datasets,
                      processed_train_dataset,
                      processed_target_dataset,
                      previously_selected_indices,
                      save_dataset_path,
                      save_embeddings_path,
                      model,
                      budget,
                      strategy,
                      iteration,
                      accelerator):
    
    raw_train_dataset = concatenate_datasets(list(raw_train_datasets.values()))
    raw_target_dataset = concatenate_datasets(list(raw_target_datasets.values()))    

    if args.compute_embeddings:
        # Get train embeddings
        train_embeddings, _, train_uncertainty = get_embeddings(args, task_type, processed_train_dataset, "train", model, accelerator, iteration, save_embeddings_path, save_embeddings=True)
        # Get target embeddings
        target_embeddings, _, _ = get_embeddings(args, task_type, processed_target_dataset, "valid", model, accelerator, iteration, save_embeddings_path, save_embeddings=True)

        ########################### TODO: confirm whether true for all qa datasets ###########################
        if args.dataset_name in ['tydiqa']:
            with accelerator.main_process_first():
                train_embeddings, target_embeddings, train_uncertainty = process_embeddings_for_qa(raw_train_dataset, processed_train_dataset, \
                    raw_target_dataset, processed_target_dataset, train_embeddings, target_embeddings, train_uncertainty, accelerator)
        ################################################ END TODO #############################################
   
        # Log size of train and target embeddings
        logger.info("Train embeddings shape: %s", train_embeddings.shape)
        logger.info("Target embeddings shape: %s", target_embeddings.shape)
    
    else:
        train_embeddings = None
        target_embeddings = None
        train_uncertainty = None

    # Check whether we need to run training for multiple budgets and one round
    if not args.multiple_budgets_one_round:
        with accelerator.main_process_first():
            # Get the indices of the train examples to be selected
            selected_indices = get_indices(args, raw_train_dataset, raw_train_datasets, train_embeddings, target_embeddings, train_uncertainty, budget, strategy, iteration, previously_selected_indices, accelerator)
            # Save dataset, embeddings and language distribution
            selected_train_dataset = save_dataset(args, save_dataset_path, save_embeddings_path, raw_train_dataset, selected_indices, train_embeddings, budget, iteration)

    else:
        if "," in args.budget:
            budget_list = [int(b) for b in args.budget.split(",")]
        else:
            budget_list = [int(args.budget)]
        for budget in budget_list:
            # Create dirs for this budget
            save_dataset_path, _, _, save_embeddings_path = create_output_dirs(args, budget, accelerator)

            print(save_dataset_path)
            print(save_embeddings_path)

            # Save dataset, embeddings and language distribution
            with accelerator.main_process_first():
                # Get the indices of the train examples to be selected
                selected_indices = get_indices(args, raw_train_dataset, raw_train_datasets, train_embeddings, \
                    target_embeddings, train_uncertainty, budget, strategy, iteration, previously_selected_indices, accelerator)
                selected_train_dataset = save_dataset(args, save_dataset_path, save_embeddings_path, raw_train_dataset, selected_indices, train_embeddings, budget, iteration)
            
    # Empty cache
    gc.collect()
    torch.cuda.empty_cache()
    return selected_train_dataset, selected_indices
