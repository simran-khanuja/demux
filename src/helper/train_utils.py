import torch
import collections
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM
)
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate.logging import get_logger
import numpy as np

logger = get_logger(__name__)


def get_model(model_path, config, tokenizer, accelerator, args):
    logger.info("Loading model from %s", model_path)
    with accelerator.main_process_first():
        # Initialize model 
        if args.task_type in ["token"]:
            model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                config=config,
                cache_dir=args.cache_dir,
            )
        elif args.task_type in ["sequence"]:	
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=config,
                cache_dir=args.cache_dir,
            )
        elif args.task_type in ["qa"]:
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_path,
                config=config,
                cache_dir=args.cache_dir,
            )
        elif args.task_type in ["mt"]:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                config=config,
                cache_dir=args.cache_dir,
            )
            # Set model.config.decoder_start_token_id to decoder_target_language
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[args.decoder_target_language]
    
    accelerator.wait_for_everyone()
    return model


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


def predict_test(
    raw_test_datasets,
    processed_test_datasets,
    model,
    tokenizer, 
    args, 
    accelerator, 
    metric, 
    label_names, 
    data_collator,
    pred_output_dir_AL
):
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
            with accelerator.main_process_first():
                if pred_output_dir_AL is not None:
                    os.makedirs(pred_output_dir_AL, exist_ok=True)
                    pred_output_file = os.path.join(pred_output_dir_AL, "predictions_" + language + ".txt")
                    with open(pred_output_file, "w") as f:
                        for pred in eval_metric["predictions"]:
                            f.write(str(pred) + "\n")
    return eval_metric


