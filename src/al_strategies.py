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
import random
import math
from accelerate.utils import set_seed
from torch.nn import CrossEntropyLoss

from helper.data_utils import UDPOS_ID_MAP

logger = get_logger(__name__)


logging.basicConfig(level=logging.INFO)

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
        logits = outputs.logits
    return embeddings, logits

def get_embeddings(args, task_type, processed_dataset, split, model, accelerator, iteration, save_embeddings=False):
    embeddings = torch.empty([0, model.config.hidden_size]).to(accelerator.device)

    # Initialize logits to empty tensor
    if task_type in ["token"]:
        logits = torch.empty([0, args.max_seq_length, model.config.num_labels]).to(accelerator.device)
    elif task_type in ['qa']:
        start_logits = torch.empty([0, args.max_seq_length]).to(accelerator.device)
        end_logits = torch.empty([0, args.max_seq_length]).to(accelerator.device)
    elif task_type in ['mt']:
        logits = torch.empty([0, args.max_target_length, model.config.vocab_size]).to(accelerator.device)
    elif task_type in ['sequence']:
        logits = torch.empty([0, model.config.num_labels]).to(accelerator.device)

    # Initialize uncertainty to empty tensor
    uncertainty = torch.empty([0]).to(accelerator.device)
    
    # Preserve the order of the dataset when batching and getting embeddings
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
            embeddings = torch.cat((embeddings, batch_embeddings))
            batch_size = batch_embeddings.shape[0]

            # Concatenate logits for each batch (except MT for which this is very large)
            if task_type in ['qa']:
                start_logits = torch.cat((start_logits, batch_logits[0]))
                end_logits = torch.cat((end_logits, batch_logits[1]))
            elif task_type in ['token', 'sequence']:
                logits = torch.cat((logits, batch_logits))
            
            # Find uncertainty margin for token level tasks 
            if task_type in ['token']:
                # Calculate the cross entropy loss for each example and pass mask to ignore padding tokens
                batch_log_softmax = torch.nn.functional.log_softmax(batch_logits, dim=2)
                mask = batch["labels"] != -100
                batch_log_softmax = batch_log_softmax*mask.unsqueeze(2)
                batch_log_softmax_sorted, _ = torch.sort(batch_log_softmax, dim=2, descending=True)
                uncertainty_margin_sample = batch_log_softmax_sorted[:,:,0] - batch_log_softmax_sorted[:,:,1]
                if args.token_task_margin == "min":
                    margin_mask = uncertainty_margin_sample!=0
                    masked_uncertainty_margin_sample = uncertainty_margin_sample.clone()
                    masked_uncertainty_margin_sample[~margin_mask] = float('inf')
                    uncertainty_margin_sample_min = torch.min(masked_uncertainty_margin_sample, dim=1)[0]
                    uncertainty = torch.cat((uncertainty, uncertainty_margin_sample_min))
                elif args.token_task_margin == "mean":
                    uncertainty_margin_sample_mean = torch.sum(uncertainty_margin_sample, dim=1) / mask.sum(dim=1)
                    uncertainty = torch.cat((uncertainty, uncertainty_margin_sample_mean))
                elif args.token_task_margin == "max":
                    uncertainty_margin_sample_max = torch.max(uncertainty_margin_sample, dim=1)[0]
                    uncertainty = torch.cat((uncertainty, uncertainty_margin_sample_max))
                elif args.token_task_margin == "mnlp":
                    uncertainty_margin_sample_mnlp = torch.sum(batch_log_softmax_sorted[:,:,0], dim=1) / torch.sum(mask, dim=1)
                    uncertainty = torch.cat((uncertainty, uncertainty_margin_sample_mnlp))
            
            # Find uncertainty value for classification tasks by subtracting the log softmax of the top 2 logits; lower margin means higher uncertainty
            elif task_type in ['sequence']:
                batch_log_softmax = torch.nn.functional.log_softmax(batch_logits, dim=1)
                batch_log_softmax_sorted, _ = torch.sort(batch_log_softmax, dim=1, descending=True)
                uncertainty = torch.cat((uncertainty, batch_log_softmax_sorted[:,0] - batch_log_softmax_sorted[:,1]))
            
            # Find uncertainty value for QA tasks by adding start and end logits; lower the final value higher the uncertainty
            elif task_type in ['qa']:
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
                uncertainty = torch.cat((uncertainty, batch_uncertainty))
            
            # Find uncertainty value of MT tasks by calculating the cross entropy loss for each example and pass mask to ignore padding tokens
            elif task_type in ['mt']:
                loss_fct = CrossEntropyLoss(reduction='none')
                batch_loss_per_token = loss_fct(batch_logits.view(-1, model.config.vocab_size), batch['labels'].view(-1)).view(batch_size, args.max_seq_length)
                mask = batch_loss_per_token!=0
                batch_loss = torch.sum(batch_loss_per_token, dim=1) / mask.sum(dim=1)
                # Taking negative of loss to make it consistent with other tasks where higher uncertainty is associated with lower margins
                uncertainty = torch.cat((uncertainty, -batch_loss))
                del batch_loss_per_token, mask, batch_loss
            
            # Log every 100 batches
            if step%100==0:
                logger.info('completed batch %s',step)
    
    # Save embeddings to disk
    if save_embeddings:
        torch.save(embeddings, args.save_embeddings_path + '/iter_' + str(iteration) + '/' + str(split) + '_embeddings.pt')
        logger.info("Saved embeddings for split: %s", split[:5])

    if task_type in ['qa']:
        logits = [start_logits, end_logits]
    
    return embeddings, logits, uncertainty


def get_train_dataset(args, 
                      task_type,
                      raw_train_datasets,
                      raw_target_datasets,
                      processed_train_dataset,
                      processed_target_dataset,
                      previously_selected_indices,
                      model,
                      budget,
                      strategy,
                      iteration,
                      accelerator,
                      train_embeddings=None,
                      train_logits=None,
                      train_uncertainty=None,
                      target_embeddings=None,
                      target_logits=None,
                      target_uncertainty=None,
                      return_embeddings=False):
    
    raw_train_dataset = concatenate_datasets(list(raw_train_datasets.values()))
    raw_target_dataset = concatenate_datasets(list(raw_target_datasets.values()))
    
    if args.seed is not None:
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True

    with accelerator.main_process_first():
        if not os.path.exists(args.save_embeddings_path + '/iter_' + str(iteration)):
            os.makedirs(args.save_embeddings_path + '/iter_' + str(iteration))

    # Get train embeddings
    if train_embeddings is None and args.strategy not in ['random', 'egalitarian'] and not args.strategy.startswith('gold'):
        split = "train"
        train_embeddings, train_logits, train_uncertainty = get_embeddings(args, task_type, processed_train_dataset, split, model, accelerator, iteration, save_embeddings=True)

    # Get target embeddings
    if target_embeddings is None and args.strategy not in ['random', 'egalitarian'] and not args.strategy.startswith('gold'):
        split = "valid"
        target_embeddings, target_logits, target_uncertainty = get_embeddings(args, task_type, processed_target_dataset, split, model, accelerator, iteration, save_embeddings=True)

    ########################### TODO: move to function; better variable names; confirm whether true for all qa datasets ###########################
    ###############################################################################################################################################
    if args.dataset_name in ['tydiqa'] and args.strategy not in ['random', 'egalitarian'] and not args.strategy.startswith('gold'):
        # train_embeddings, train_logits, train_uncertainty = get_embeddings(args, task_type, processed_train_dataset, "train", model, accelerator, iteration, save_embeddings=True)
        # target_embeddings, target_logits, target_uncertainty = get_embeddings(args, task_type, processed_target_dataset, "valid", model, accelerator, iteration, save_embeddings=True)
        original_to_processed = collections.defaultdict(list)
        for idx, example_id in enumerate(processed_train_dataset['example_id']):
            original_to_processed[example_id].append(idx)
        
        original_to_processed_target = collections.defaultdict(list)
        for idx, example_id in enumerate(processed_target_dataset['example_id']):
            original_to_processed_target[example_id].append(idx)
        
        original_uncertainity = {}
        train_average_cls_embedding = {}
        target_average_cls_embedding = {}
        for ind in original_to_processed.keys():
            uncertainity = train_uncertainty[original_to_processed[ind]]
            uncertainity = torch.max(uncertainity)
            original_uncertainity[ind]=uncertainity 

            # take the average of CLS Embeddings of all features within an example
            train_average_cls_embedding[ind] = torch.mean(train_embeddings[original_to_processed[ind]],0).reshape(1,-1)
            
        
        for ind in original_to_processed_target.keys():
            target_average_cls_embedding[ind] = torch.mean(target_embeddings[original_to_processed_target[ind]],0).reshape(1,-1)

        temp_train_uncertainity = []
        temp_train_embeddings = torch.empty([0, model.config.hidden_size]).to(accelerator.device)
        temp_target_embeddings = torch.empty([0, model.config.hidden_size]).to(accelerator.device)
        for i in raw_train_dataset['id']:
            temp_train_uncertainity.append(original_uncertainity[i].item())
            temp_train_embeddings = torch.cat((temp_train_embeddings,train_average_cls_embedding[i]))

        for i in raw_target_dataset['id']:
            temp_target_embeddings = torch.cat((temp_target_embeddings,target_average_cls_embedding[i]))
    
        train_uncertainty = torch.tensor(temp_train_uncertainity).to(accelerator.device)
        train_embeddings = temp_train_embeddings
        target_embeddings = temp_target_embeddings

    if args.save_embeddings and args.strategy not in ['random', 'egalitarian'] and not args.strategy.startswith('gold'):
        logger.info("Train embeddings shape: {}".format(train_embeddings.shape))
        logger.info("Target embeddings shape: {}".format(target_embeddings.shape))
    
    ################################################## Code modification ends here #####################################################
    #####################################################################################################################################

    ################################# Code for all strategies begins here #################################
    ########################################### AVERAGE-DIST ##############################################

    ################################# TODO: clean this up; ask if better way to deal with larger train embedding matrices #########################################
    ##############################################################################################################################################################
    if strategy=="average_dist":
        logger.info("Diversity mean strategy")
        # Calculate the mean distance between each train example and all target examples in train batches of args.max_train_emb_len examples on GPU
        if len(train_embeddings) > args.max_train_emb_len:
            mean_budget_dists_batches = torch.empty([0]).to(accelerator.device)
            topk_indices_batches = []
            dist_matrix = torch.empty([0, len(target_embeddings)]).to(accelerator.device)
            for i in range(0, len(train_embeddings), args.max_train_emb_len):
                if i+args.max_train_emb_len > len(train_embeddings):
                    dist_matrix_sample = torch.cdist(train_embeddings[i:], target_embeddings, p=2)
                else:
                    dist_matrix_sample = torch.cdist(train_embeddings[i:i+args.max_train_emb_len], target_embeddings, p=2)
                dist_matrix_sample_mean = torch.mean(dist_matrix_sample, dim=1)
                if budget > len(dist_matrix_sample_mean):
                    mean_budget_dists_sample, topk_indices_sample = torch.topk(dist_matrix_sample_mean, len(dist_matrix_sample_mean), largest=False)
                else:
                    mean_budget_dists_sample, topk_indices_sample = torch.topk(dist_matrix_sample_mean, budget, largest=False)
                mean_budget_dists_batches = torch.cat((mean_budget_dists_batches, mean_budget_dists_sample))
                topk_indices_sample = topk_indices_sample.to(torch.int)
                for j in topk_indices_sample.tolist():
                    topk_indices_batches.append(i + j)
                del dist_matrix_sample, dist_matrix_sample_mean, mean_budget_dists_sample, topk_indices_sample
            # Get budget number of top k mean distances and indices
            mean_budget_dists, topk_indices_mean_dist = torch.topk(mean_budget_dists_batches, budget, largest=False)
            # Get the indices of the top k mean distances in the original train embeddings
            topk_indices_mean_dist = topk_indices_mean_dist.to(torch.int).tolist()
            topk_indices_batches = torch.tensor(topk_indices_batches).to(accelerator.device)
            topk_indices = topk_indices_batches[topk_indices_mean_dist]
            del mean_budget_dists_batches, topk_indices_batches, topk_indices_mean_dist
        else:
            dist_matrix = torch.cdist(train_embeddings, target_embeddings, p=2).to(accelerator.device)
            mean_budget_dists, topk_indices = torch.topk(torch.mean(dist_matrix, dim=1), min(budget,dist_matrix.shape[0]), largest=False)

        logger.info("Min mean dist: {}".format(mean_budget_dists[0]))
        logger.info("Max mean dist: {}".format(mean_budget_dists[-1]))

        # Cast top_k indices to list of ints
        topk_indices = topk_indices.to(torch.int)

        selected_train_embeddings = torch.empty([0, model.config.hidden_size]).to(accelerator.device)
        selected_indices = previously_selected_indices.copy()
        for i, index in enumerate(topk_indices.tolist()):
            if len(selected_indices) == budget:
                break
            if index not in selected_indices:
                selected_indices.append(index)

        del dist_matrix, mean_budget_dists, topk_indices
    
    ############################################## TODO ends here ##############################################
    ############################################################################################################

    ############################################## RANDOM and GOLD ##############################################
    if strategy=="random" or strategy.startswith("gold"):
        logger.info("Random strategy")
        selected_indices = previously_selected_indices.copy()
        if budget > len(raw_train_dataset):
            selected_indices = list(range(len(raw_train_dataset)))
        else:
            if args.per_language_allocation_file is not None:
                # read tsv file
                iteration_number = None
                with open(args.per_language_allocation_file, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    fields = line.strip().split("\t")
                    if len(fields) == 2:
                        dataset_config, allocation = fields
                        dataset, config = dataset_config.split(":")
                        if dataset == args.dataset_name and config == args.target_config_name:
                            per_language_allocation = allocation.split(",")
                            break
                    elif len(fields) == 3:
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
                    if not iteration_number:
                        per_language_allocation_each_round[language] = int(count) / args.total_rounds
                    else:
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
                        print(language, per_language_budget)
                        count = 0
                        
                        while count < per_language_budget:
                            random_index = np.random.randint(train_dataset_size, train_dataset_size + len(train_dataset))
                            if random_index not in selected_indices:
                                    selected_indices.append(random_index)
                                    count += 1
                        train_dataset_size += len(train_dataset)

            else:
                while len(selected_indices) < budget:
                    random_index = np.random.randint(0, len(raw_train_dataset))
                    if random_index not in selected_indices:
                        selected_indices.append(random_index)
    
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

    ############################################## UNCERTAINTY ###################################################
    if strategy.startswith("uncertainty"):
        # Calculate uncertainty of train logits. Choose train samples with lowest average uncertainty.
        logger.info("Uncertainty strategy")
        # Select train samples with lowest average margin
        margin = train_uncertainty
        margin_topk, topk_indices = torch.topk(margin, min(budget, margin.shape[0]), largest=False)
        logger.info("Min margin: {}".format(margin_topk[0]))
        logger.info("Max margin: {}".format(margin_topk[-1]))

        selected_indices = previously_selected_indices.copy()
        for i in topk_indices.tolist():
            if len(selected_indices) == budget:
                break
            if i not in selected_indices:
                selected_indices.append(i)

    ############################################## KNN-UNCERTAINTY ################################################

    ################################# TODO: clean this up; ask if better way to deal with larger train embedding matrices #########################################
    ##############################################################################################################################################################
    if strategy.startswith("knn_uncertainty"):
        logger.info("KNN Uncertainty Margin strategy")
        # Calculate K nearest neighbors for each target based on minimum CLSD
        if len(train_embeddings) > args.max_train_emb_len or len(target_embeddings) > args.max_target_emb_len:
            logger.info("Train embeddings shape: {}".format(len(train_embeddings)))
            logger.info("Target embeddings shape: {}".format(len(target_embeddings)))
            logger.info("Max train emb len: {}".format(args.max_train_emb_len))
            logger.info("Max target emb len: {}".format(args.max_target_emb_len))
            # The two tensors below are of type long
            dist_matrix_sorted_values = torch.empty([0, 512], dtype=torch.long).to(accelerator.device)
            dist_matrix_sorted_indices = torch.empty([0, 512], dtype=torch.long).to(accelerator.device)
            logger.info("Dist matrix sorted values shape: {}".format(dist_matrix_sorted_values.shape))
            for i in range(0, len(target_embeddings), 100):
                if i+100 > len(target_embeddings):
                    dist_matrix_sample = torch.cdist(target_embeddings[i:], train_embeddings, p=2).to(accelerator.device)
                else:
                    dist_matrix_sample = torch.cdist(target_embeddings[i:i+100], train_embeddings, p=2).to(accelerator.device)
                dist_matrix_sample_sorted_values, dist_matrix_sample_sorted_indices = torch.sort(dist_matrix_sample, dim=1)
                dist_matrix_sorted_values = torch.cat((dist_matrix_sorted_values, dist_matrix_sample_sorted_values[:, :512]))
                dist_matrix_sorted_indices = torch.cat((dist_matrix_sorted_indices, dist_matrix_sample_sorted_indices[:, :512]))
                del dist_matrix_sample, dist_matrix_sample_sorted_values, dist_matrix_sample_sorted_indices
                torch.cuda.empty_cache()
        else:
            dist_matrix = torch.cdist(target_embeddings, train_embeddings, p=2).to(accelerator.device)
            dist_matrix_sorted_values, dist_matrix_sorted_indices = torch.sort(dist_matrix, dim=1)
            del dist_matrix
        
        ############################################## TODO ends here ##############################################
        ############################################################################################################
        
        logger.info("dist_matrix shape: {}".format(dist_matrix_sorted_values.shape))
        if args.k_value is not None:
            k_value = args.k_value
        else:
            k_value = 1
        candidate_indices = previously_selected_indices.copy()
        while True:
            k_nearest_neighbours = dist_matrix_sorted_indices[:, :k_value]
            all_k_nearest_neighbours = torch.unique(k_nearest_neighbours.reshape(-1))
            candidate_indices.extend(all_k_nearest_neighbours.tolist())
            candidate_indices_set = set(candidate_indices)
            if len(candidate_indices_set) > budget:
                logger.info("k_value: {}".format(k_value))
                logger.info("Length of candidate indices set: {}".format(len(candidate_indices_set)))
                break
            elif k_value > dist_matrix_sorted_values.shape[1]:
                logger.info("k_value exceeds number of train samples")
                break
            else:
                del candidate_indices, candidate_indices_set, all_k_nearest_neighbours, k_nearest_neighbours
                k_value *= 2
                candidate_indices = previously_selected_indices.copy()
        logger.info("k_value: {}".format(k_value))
        all_k_nn_uncertainty_margin = train_uncertainty[all_k_nearest_neighbours]
        all_k_nn_uncertainty_margin_sorted, all_k_nn_uncertainty_margin_sorted_indices = torch.sort(all_k_nn_uncertainty_margin, descending=False)
        new_train_indices_sorted = all_k_nearest_neighbours[all_k_nn_uncertainty_margin_sorted_indices]

        selected_indices = previously_selected_indices.copy()
        for i in new_train_indices_sorted.tolist():
            if len(selected_indices) == budget:
                break
            if i not in selected_indices:
                selected_indices.append(i)
        logger.info("Selected indices shape: {}".format(len(selected_indices)))

        del dist_matrix_sorted_values, dist_matrix_sorted_indices, k_nearest_neighbours, all_k_nearest_neighbours, \
        all_k_nn_uncertainty_margin, all_k_nn_uncertainty_margin_sorted, all_k_nn_uncertainty_margin_sorted_indices

    ############################################## Code for strategies ends here ################################################
    #############################################################################################################################

    if strategy.startswith("knn") and k_value is not None:
        save_dataset_path = args.save_dataset_path + '/iter_' + str(iteration) + '_start_' + str(args.k_value) + 'end_' + str(k_value) 
    else:
        save_dataset_path = args.save_dataset_path + '/iter_' + str(iteration)

    with accelerator.main_process_first():
        if not os.path.exists(save_dataset_path):
            os.makedirs(save_dataset_path)
    
    # Save dataset and embeddings
    raw_train_dataset.select(selected_indices).to_csv(save_dataset_path + '/' + str(budget) + '.csv')
    # Save in arrow format
    raw_train_dataset.select(selected_indices).save_to_disk(save_dataset_path)
    logger.info("train dataset first sample: {}".format(raw_train_dataset[0]))

    # Save language distribution of selected dataset
    selected_train_dataset = raw_train_dataset.select(selected_indices)
    logger.info("selected train dataset first sample: {}".format(selected_train_dataset[0]))
    language_distribution = selected_train_dataset.to_pandas()["language"].value_counts()
    language_distribution.to_csv(save_dataset_path + '/' + str(budget) + '_language_distribution.csv')

    # Save embeddings
    if args.save_embeddings and args.strategy not in ['random', 'egalitarian'] and not args.strategy.startswith('gold'):
        selected_train_embeddings = train_embeddings[selected_indices]
        torch.save(selected_train_embeddings, args.save_embeddings_path + '/iter_' + str(iteration) + '/selected_train_' + str(budget) + '.pt')

        if not return_embeddings:
            del train_embeddings, target_embeddings

    # Empty cache
    gc.collect()
    torch.cuda.empty_cache()
    if return_embeddings:
        return selected_train_dataset, selected_indices, train_embeddings, train_logits, train_uncertainty, target_embeddings, target_logits, target_uncertainty
    else:
        return selected_train_dataset, selected_indices, None, None, None, None, None, None
