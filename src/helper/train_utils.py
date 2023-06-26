import torch
import collections
from typing import Tuple
import argparse
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate.logging import get_logger
import numpy as np
from find_correlation import calculate_embeddings
import scipy

logger = get_logger(__name__)

def get_embeddings_and_uncertainty(
    dataloader: DataLoader,
    model: torch.nn.Module,
    task_type: str,
    accelerator: Accelerator,
    args: argparse.Namespace,
    embeddings: torch.Tensor,
    uncertainty_margin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate embeddings and uncertainty margin.
    
    Args:
        dataloader (torch.utils.data.DataLoader): The data loader.
        model (torch.nn.Module): The model.
        task_type (str): The type of task (token, sequence, QA).
        accelerator: Accelerator.
        args: Arguments.
        embeddings (torch.Tensor): Calculated embeddings.
        uncertainty_margin (torch.Tensor): Calculated uncertainty margin.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The embeddings and the uncertainty margin.
    """
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            batch_embeddings, batch_logits = calculate_embeddings(batch, accelerator.device, model, args.dataset_name, args.embedding_method)
            embeddings = torch.cat((embeddings, batch_embeddings))
            if task_type in ["token"]:
                # Calculate train uncertainty margin
                batch_log_softmax = torch.nn.functional.log_softmax(batch_logits, dim=2)
                mask = batch["labels"] != -100
                batch_log_softmax = batch_log_softmax*mask.unsqueeze(2)
                batch_log_softmax_sorted, _ = torch.sort(batch_log_softmax, dim=2, descending=True)
                uncertainty_margin_sample = batch_log_softmax_sorted[:,:,0] - batch_log_softmax_sorted[:,:,1]
                # Find mean of the difference between the log softmax of the most probable class and the log softmax of the second most probable class
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
            elif task_type in ['sequence']:
                batch_log_softmax = torch.nn.functional.log_softmax(batch_logits, dim=1)
                batch_log_softmax_sorted, _ = torch.sort(batch_log_softmax, dim=1, descending=True)
                uncertainty_margin = torch.cat((uncertainty_margin, batch_log_softmax_sorted[:,0] - batch_log_softmax_sorted[:,1]))
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
                uncertainty_margin = torch.cat((uncertainty_margin, batch_uncertainty))
                
            if step%100==0:
                logger.info(f'completed batch {step}')
    return embeddings, uncertainty_margin


def process_tydiqa_uncertainty(dataset, processed_dataset_before, uncertainty_margin, accelerator):
    original_to_processed = collections.defaultdict(list)
    for idx, example_id in enumerate(processed_dataset_before['example_id']):
        original_to_processed[example_id].append(idx)

    original_uncertainty = {}
    for ind in original_to_processed.keys():
        uncertainty = uncertainty_margin[original_to_processed[ind]]
        uncertainty = torch.max(uncertainty)
        original_uncertainty[ind] = uncertainty.item()

    temp_uncertainty = []
    for i in dataset['id']:
        temp_uncertainty.append(original_uncertainty[i])

    uncertainty_margin = torch.tensor(temp_uncertainty).to(accelerator.device)

    del temp_uncertainty, original_uncertainty, original_to_processed, processed_dataset_before
    return uncertainty_margin


def postprocess_token_classification(predictions, labels, label_names):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


def eval_model_token_classification(model, eval_dataloader, metric, label_names, accelerator):
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()
    all_predictions = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, labels = accelerator.gather((predictions, batch["labels"]))
        true_predictions, true_labels = postprocess_token_classification(predictions, labels, label_names)
        metric.add_batch(predictions=true_predictions, references=true_labels)
        all_predictions.extend(true_predictions)
    eval_metric = metric.compute()
    eval_metric["predictions"] = all_predictions
    for key in ["precision", "recall", "f1", "accuracy"]:
        eval_metric[key] = eval_metric[f"overall_{key}"]

    return eval_metric, outputs.loss.item()


def eval_model_sequence_classification(model, eval_dataloader, metric, accelerator):
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()
    all_predictions = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
        all_predictions.extend(predictions.cpu().numpy().tolist())

    eval_metric = metric.compute(average="weighted")
    eval_metric["predictions"] = all_predictions
    
    return eval_metric, outputs.loss.item()

def postprocess_question_answering(start_logits, end_logits, features, examples, n_best=20, max_answer_length=1000):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return predicted_answers, theoretical_answers

    
def eval_model_question_answering(model, eval_dataloader, metric,  validation_dataset, examples, accelerator):
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()
    start_logits = []
    end_logits = []
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)

    start_logits = start_logits[: len(eval_dataloader.dataset)]
    end_logits = end_logits[: len(eval_dataloader.dataset)]
    predicted_answers, theoretical_answers = postprocess_question_answering(start_logits, end_logits, validation_dataset, examples, n_best=20, max_answer_length=1000)
    eval_metric = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    eval_metric["predictions"] = predicted_answers

    return eval_metric, outputs.loss


def postprocess_mt_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def eval_model_mt(args, tokenizer, model, eval_dataloader, metric, accelerator):
    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": args.num_beams,
    }
    samples_seen = 0
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = batch["labels"]
            if not args.pad_to_max_length:
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
            labels = accelerator.gather(labels).cpu().numpy()

            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = postprocess_mt_text(decoded_preds, decoded_labels)

            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    decoded_preds = decoded_preds[: len(eval_dataloader.dataset) - samples_seen]
                    decoded_labels = decoded_labels[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += len(decoded_labels)

            metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    eval_metric = metric.compute()
    logger.info({"bleu": eval_metric["score"]})
    return eval_metric


def eval_dev_test(
        model,
        tokenizer,
        task_type,
        dataset_name,
        processed_validation_datasets,
        processed_test_datasets,
        raw_validation_datasets,
        raw_test_datasets,
        metric,
        completed_steps,
        iteration,
        total_steps,
        args,
        data_collator,
        accelerator,
        label_names,
    ):
    dev_test_accuracies = {}
    if iteration is not None:
        if args.multiple_budgets_one_round:
            iteration_suffix = f"_b{iteration}"
        else:
            iteration_suffix = f"_iter{iteration}"
    else:
        iteration_suffix = ""
    if completed_steps==0 or completed_steps==total_steps:
        for language in processed_validation_datasets:
            processed_validation_dataset = processed_validation_datasets[language]
            if task_type in ["qa"] and "example_id" in processed_validation_dataset.features:
                processed_validation_dataset= processed_validation_dataset.remove_columns(["example_id", "offset_mapping"])
                
            validation_dataloader = DataLoader(processed_validation_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

            metric_to_track = "f1"
            if task_type in ["token"]:
                eval_metric, _ = eval_model_token_classification(model, validation_dataloader, metric, label_names, accelerator)
            elif task_type in ["sequence"]:
                eval_metric, _ = eval_model_sequence_classification(model, validation_dataloader, metric, accelerator)
            elif task_type in ["qa"]:
                eval_metric, _ = eval_model_question_answering(model, validation_dataloader, metric,  processed_validation_datasets[language], raw_validation_datasets[language], accelerator)
            elif task_type in ["mt"]:
                eval_metric = eval_model_mt(args, tokenizer, model, validation_dataloader, metric, accelerator)
                metric_to_track = "score"
            logger.info(f"Validation {metric_to_track} for {language}: {eval_metric[metric_to_track]}")
            if args.with_tracking:
                accelerator.log(
                    {
                        f"{language}_eval_{metric_to_track}{iteration_suffix}": eval_metric[metric_to_track],
                    },
                    step=completed_steps,
                )
            dev_test_accuracies[language+"_val"] = eval_metric[metric_to_track]

    for language in processed_test_datasets:
        processed_test_dataset = processed_test_datasets[language]
        if task_type in ["qa"] and "example_id" in processed_test_dataset.features:
            processed_test_dataset= processed_test_dataset.remove_columns(["example_id", "offset_mapping"])
            
        test_dataloader = DataLoader(processed_test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

        metric_to_track = "f1"
        if task_type in ["token"]:
            eval_metric, _ = eval_model_token_classification(model, test_dataloader, metric, label_names, accelerator)
        elif task_type in ["sequence"]:
            eval_metric, _ = eval_model_sequence_classification(model, test_dataloader, metric, accelerator)
        elif task_type in ["qa"]:
            eval_metric, _ = eval_model_question_answering(model, test_dataloader, metric,  processed_test_datasets[language], raw_test_datasets[language], accelerator)
        elif task_type in ["mt"]:
            eval_metric = eval_model_mt(args, tokenizer, model, test_dataloader, metric, accelerator)
            metric_to_track = "score"
        logger.info(f"Test {metric_to_track} for {language}: {eval_metric[metric_to_track]}")
    
        if args.with_tracking:
            accelerator.log(
                {
                    f"{language}_test_{metric_to_track}{iteration_suffix}": eval_metric[metric_to_track],
                },
                step=completed_steps,
            )
        dev_test_accuracies[language+"_test"] = eval_metric[metric_to_track]
    return dev_test_accuracies


def find_optimal_k(
    processed_source_dataset,
    raw_source_dataset,
    processed_target_dataset,
    raw_target_dataset,
    model,
    accelerator,
    data_collator,
    args
):
    # Get embeddings of source and target datasets
    if args.dataset_name in ['tydiqa']:
        processed_source_dataset_before = processed_source_dataset
        processed_target_dataset_before = processed_target_dataset
        for i in ['offset_mapping', 'example_id', 'language']:
            if i in processed_source_dataset.features:
                processed_source_dataset = processed_source_dataset.remove_columns([i])
            if i in processed_target_dataset.features:
                processed_target_dataset = processed_target_dataset.remove_columns([i])

    logger.info("Getting embeddings of source and target datasets")
    source_dataloader = DataLoader(processed_source_dataset, collate_fn=data_collator, batch_size=args.inference_batch_size, shuffle=False)
    target_dataloader = DataLoader(processed_target_dataset, collate_fn=data_collator, batch_size=args.inference_batch_size, shuffle=False)

    # Embeddings
    source_embeddings = torch.empty((0, model.config.hidden_size), dtype=torch.float32).to(accelerator.device)
    target_embeddings = torch.empty((0, model.config.hidden_size), dtype=torch.float32).to(accelerator.device)

    # Source loss and uncertainty margin
    source_uncertainty_margin = torch.empty((0), dtype=torch.float32).to(accelerator.device)
    target_uncertainty_margin = torch.empty((0), dtype=torch.float32).to(accelerator.device)

    # Get embeddings of source and target datasets and calculate uncertainty margin
    logger.info("Getting embeddings of source and target datasets and calculating uncertainty margin")
    source_embeddings, source_uncertainty_margin = get_embeddings_and_uncertainty(
        source_dataloader, model, args.task_type, accelerator, args, source_embeddings, source_uncertainty_margin)

    target_embeddings, target_uncertainty_margin = get_embeddings_and_uncertainty(
        target_dataloader, model, args.task_type, accelerator, args, target_embeddings, target_uncertainty_margin)
    

    if args.task_type in ["qa"]:
        original_to_processed = collections.defaultdict(list)
        for idx, example_id in enumerate(processed_source_dataset_before['example_id']):
            original_to_processed[example_id].append(idx)
        
        original_to_processed_target = collections.defaultdict(list)
        for idx, example_id in enumerate(processed_target_dataset_before['example_id']):
            original_to_processed_target[example_id].append(idx)
        
        source_uncertainity = {}
        target_uncertainity = {}
        source_average_cls_embedding = {}
        target_average_cls_embedding = {}
        for ind in original_to_processed.keys():
            uncertainity = source_uncertainty_margin[original_to_processed[ind]]
            uncertainity = torch.max(uncertainity)
            source_uncertainity[ind]=uncertainity 

            # take the average of CLS Embeddings of all features within an example
            source_average_cls_embedding[ind] = torch.mean(source_embeddings[original_to_processed[ind]],0).reshape(1,-1)
            
        
        for ind in original_to_processed_target.keys():
            uncertainity = target_uncertainty_margin[original_to_processed_target[ind]]
            uncertainity = torch.max(uncertainity)
            target_uncertainity[ind]=uncertainity

            # take the average of CLS Embeddings of all features within an example
            target_average_cls_embedding[ind] = torch.mean(target_embeddings[original_to_processed_target[ind]],0).reshape(1,-1)

        temp_source_uncertainity = []
        temp_target_uncertainity = []
        temp_source_embeddings = torch.empty([0, model.config.hidden_size]).to(accelerator.device)
        temp_target_embeddings = torch.empty([0, model.config.hidden_size]).to(accelerator.device)
        for i in raw_source_dataset['id']:
            temp_source_uncertainity.append(source_uncertainity[i].item())
            temp_source_embeddings = torch.cat((temp_source_embeddings,source_average_cls_embedding[i]))
            

        for i in raw_target_dataset['id']:
            temp_target_uncertainity.append(target_uncertainity[i].item())
            temp_target_embeddings = torch.cat((temp_target_embeddings,target_average_cls_embedding[i]))
    
        source_uncertainty_margin = torch.tensor(temp_source_uncertainity).to(accelerator.device)
        target_uncertainty_margin = torch.tensor(temp_target_uncertainity).to(accelerator.device)
        source_embeddings = temp_source_embeddings
        target_embeddings = temp_target_embeddings
        
        del temp_source_uncertainity, source_uncertainity, original_to_processed, source_average_cls_embedding, target_average_cls_embedding, temp_target_embeddings, original_to_processed_target
    
    # Calculate L2 distance of target with source
    logger.info("Calculating L2 distance of target with source")
    dist_matrix = torch.cdist(target_embeddings.cpu(), source_embeddings.cpu(), p=2)
    dist_matrix = dist_matrix.to(accelerator.device)

    mean_dists = torch.mean(dist_matrix, dim=1)
    logger.info("Mean dists shape: {}".format(mean_dists.shape))


    # Get k nearest neighbours for each target example
    logger.info("Getting k nearest neighbours for each target example")

    knn_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    best_corr = 0
    best_k = 0
    corr_coeffs = {}
    with accelerator.main_process_first():
        for k in knn_list:
            if k > dist_matrix.shape[1]:
                break
            logger.info("Getting k={} nearest neighbours".format(k))
            _, k_nearest_neighbours = torch.topk(dist_matrix, k=k, dim=1, largest=False)
            logger.info("k_nearest_neighbours shape: {}".format(k_nearest_neighbours.shape))
            # Calculate average source loss of k nearest neighbours
            logger.info("Calculating average source uncertainty of k nearest neighbours by finding margin values")
            # We need to repeat the source logits to match the shape of k_nearest_neighbours
            source_uncertainty_margin_repeated = source_uncertainty_margin.repeat(k_nearest_neighbours.shape[0], 1)
            k_nearest_neighbours_uncertainty_margin = torch.gather(source_uncertainty_margin_repeated, 1, k_nearest_neighbours)
            k_nearest_neighbours_uncertainty_margin = torch.mean(k_nearest_neighbours_uncertainty_margin, dim=1)
            logger.info("k_nearest_neighbours_uncertainty_margin shape: {}".format(k_nearest_neighbours_uncertainty_margin.shape))

            # Find correlation coefficient between target uncertainty margin and k nearest neighbours uncertainty margin
            logger.info("Finding correlation coefficient between target uncertainty margin and k nearest neighbours uncertainty margin")
            r, p = scipy.stats.pearsonr(target_uncertainty_margin.cpu().numpy(), k_nearest_neighbours_uncertainty_margin.cpu().numpy())
            logger.info("Correlation coefficient = {}".format(r))
            corr_coeffs[k] = r
            if r > best_corr:
                best_corr = r
                best_k = k
        logger.info("Best k = {}".format(best_k))
        logger.info("Best correlation = {}".format(best_corr))

    del dist_matrix, mean_dists, source_embeddings, target_embeddings, source_uncertainty_margin, target_uncertainty_margin

    return best_k, corr_coeffs

    
