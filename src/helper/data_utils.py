from datasets import load_dataset, Dataset, load_dataset_builder
from typing import Dict, List, Tuple, Any
from transformers import PreTrainedTokenizerFast
from accelerate import Accelerator
import argparse
import yaml
import os
from accelerate.logging import get_logger
from functools import partial
from sklearn.model_selection import train_test_split

logger = get_logger(__name__)

TYDIQA_ID_MAP = {
    'indonesian': 'id',
    'arabic': 'ar',
    'russian': 'ru',
    'swahili': 'sw',
    'korean': 'ko',
    'finnish': 'fi',
    'english': 'en',
    'bengali': 'bn',
    'telugu': 'te'
}

UDPOS_ID_MAP = {
    'af': 'Afrikaans',
    'ar': 'Arabic',
    'eu': 'Basque',
    'bg': 'Bulgarian',
    'nl': 'Dutch',
    'en': 'English',
    'et': 'Estonian',
    'fi': 'Finnish',
    'fr': 'French',
    'de': 'German',
    'el': 'Greek',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hu': 'Hungarian',
    'id': 'Indonesian',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh': 'Chinese',
    'mr': 'Marathi',
    'fa': 'Persian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'es': 'Spanish',
    'ta': 'Tamil',
    'te': 'Telugu',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'ur': 'Urdu'
}

MBART_ID_MAP = {
    "ar": "ar_AR",
    "cs": "cs_CZ",
    "de": "de_DE",
    "en": "en_XX",
    "es": "es_XX",
    "et": "et_EE",
    "fi": "fi_FI",
    "fr": "fr_XX",
    "gu": "gu_IN",
    "hi": "hi_IN",
    "it": "it_IT",
    "ja": "ja_XX",
    "kk": "kk_KZ",
    "ko": "ko_KR",
    "lt": "lt_LT",
    "lv": "lv_LV",
    "my": "my_MM",
    "ne": "ne_NP",
    "nl": "nl_XX",
    "ro": "ro_RO",
    "ru": "ru_RU",
    "si": "si_LK",
    "tr": "tr_TR",
    "vi": "vi_VN",
    "zh": "zh_CN",
    "af": "af_ZA",
    "az": "az_AZ",
    "bn": "bn_IN",
    "fa": "fa_IR",
    "he": "he_IL",
    "hr": "hr_HR",
    "id": "id_ID",
    "ka": "ka_GE",
    "km": "km_KH",
    "mk": "mk_MK",
    "ml": "ml_IN",
    "mn": "mn_MN",
    "mr": "mr_IN",
    "pl": "pl_PL",
    "ps": "ps_AF",
    "pt": "pt_XX",
    "sv": "sv_SE",
    "sw": "sw_KE",
    "ta": "ta_IN",
    "te": "te_IN",
    "th": "th_TH",
    "tl": "tl_XX",
    "uk": "uk_UA",
    "ur": "ur_PK",
    "xh": "xh_ZA",
    "gl": "gl_ES",
    "sl": "sl_SI"
}

flores_id_map = {
    "ind_Latn": "id_ID",
    "vie_Latn": "vi_VN",
    "mya_Mymr": "my_MM",
    "khm_Khmr": "km_KH",
    "tha_Thai": "th_TH",
    "tgl_Latn": "tl_XX",
    "eng_Latn": "en_XX",
}


def create_output_dirs(args, budget, accelerator):
    # Suffix creation for saving files
    if "uncertainty" in args.strategy and args.task_type in ["token"]:
        strategy_folder_name = args.strategy + "_margin_" + str(args.token_task_margin)
    else:
        strategy_folder_name = args.strategy

    # Handle the repository creation
    hparam_suffix = f"/{args.embedding_model}" + f"/{args.dataset_name}" + f"/{args.target_config_name}" + f"/{budget}"+ f"/{args.learning_rate}" + f"_{args.num_train_epochs}" + f"_seed_{args.seed}" + f"/{strategy_folder_name}"
    save_dataset_path = args.save_dataset_path + hparam_suffix
    output_dir = args.output_dir + hparam_suffix
    pred_output_dir = args.pred_output_dir + hparam_suffix
    save_embeddings_path = args.save_embeddings_path + hparam_suffix
    # Create repository for saving files
    if accelerator.is_main_process:
        if not os.path.exists(save_dataset_path):
            os.makedirs(save_dataset_path, exist_ok=True)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(pred_output_dir):
            os.makedirs(pred_output_dir, exist_ok=True)
        if not os.path.exists(save_embeddings_path):
            os.makedirs(save_embeddings_path, exist_ok=True)
        logger.info(f"Saving dataset to {save_dataset_path}")
        logger.info(f"Saving model to {output_dir}")
        logger.info(f"Saving predictions to {pred_output_dir}")
        logger.info(f"Saving embeddings to {save_embeddings_path}")
    accelerator.wait_for_everyone()

    return save_dataset_path, output_dir, pred_output_dir, save_embeddings_path


def get_tydiqa_dataset(
    language: str = "en",
    split: str = "train",
    cache_dir: str = "",
) -> Dataset:
    """
    Get the TyDiQA dataset for a specific language and split.

    Args:
        language (str): The language code of the dataset (default: "en").
        split (str): The split of the dataset to load (default: "train").
        cache_dir (str): Directory to cache the dataset (default: None).

    Returns:
        Dataset: The TyDiQA dataset for the specified language and split, or None if not found.
    """
    tydiqa_dataset = {}
    tydiqa_dataset['test'] = load_dataset('tydiqa', 'secondary_task', cache_dir=cache_dir, split='validation')
    train_validation_split = load_dataset('tydiqa', 'secondary_task', cache_dir=cache_dir, split='train').train_test_split(test_size=0.2, seed=42)  # type: ignore
    tydiqa_dataset['train'] = train_validation_split['train']
    tydiqa_dataset['validation'] = train_validation_split['test']

    unique_languages = {item['id'].split('-')[0] for item in tydiqa_dataset[split]}
    dataset_by_language = {
        lang: tydiqa_dataset[split].filter(lambda x: lang in x['id']) for lang in unique_languages
    }
    mapped_dataset_by_language = {TYDIQA_ID_MAP[lang]: dataset for lang, dataset in dataset_by_language.items()}

    return mapped_dataset_by_language.get(language, None)

# Recast feature names for MT
def recast_mt_features(
    examples: Dict[str, Any],
    dataset_name: str,
    language_pair: str
) -> Dict[str, List[str]]:
    """
    Recast feature names for MT.

    Args:
        examples (Dict[str, Any]): Dictionary of features.
        dataset_name (str): Name of the dataset.
        language_pair (str): Language pair.

    Returns:
        Dictionary of features with recasted feature names.
    """
    # Apply the update function to the dataset.
    sources = []
    targets = []
    if dataset_name == 'opus100':
        source_language, target_language = language_pair.split('-')
        # If source_language is en, then we need to swap the source and target languages.
        if source_language == 'en':
            source_language, target_language = target_language, source_language
        for example in examples['translation']:
            if source_language and target_language in example:
                sources.append(example[source_language])
                targets.append(example[target_language])
            else:
                sources.append('')
                targets.append('')
    elif dataset_name == 'facebook/flores':
        source_language, target_language = language_pair.split('-')
        # If source_language is eng*, then we need to swap the source and target languages.
        if source_language.startswith('eng'):
            source_language, target_language = target_language, source_language
        for src, tgt in zip(examples["sentence_"+source_language], examples["sentence_"+target_language]):
            sources.append(src)
            targets.append(tgt)
    
    # Update the 'translation' feature.
    examples['translation'] = [{'source': s, 'target': t} for s, t in zip(sources, targets)]

    return examples

# Get MT config names
def get_mt_config_names(
    dataset_name: str
) -> List[str]:
    """
    Get MT config names.
    """
    config_names = []
    if dataset_name == 'opus100':
        builder = load_dataset_builder('opus100', 'af-en')
        # Get a list of all configuration names
        config_names = list(builder.builder_configs.keys())
        mbart_languages = list(MBART_ID_MAP.keys())
        mbart_supported_languages = []
        # Check if the language is supported by mbart
        for config in config_names:
            src, tgt  = config.split("-")
            if src in mbart_languages and tgt in mbart_languages:
                mbart_supported_languages.append(config)               
        config_names = mbart_supported_languages
    elif dataset_name == 'facebook/flores':
        builder = load_dataset_builder('facebook/flores', 'eng_Latn-ukr_Cyrl')
        # Get a list of all configuration names
        config_names = list(builder.builder_configs.keys())

    return config_names

# Load datasets
def load_train_datasets(
    args : argparse.Namespace,
    accelerator : Accelerator,
    tokenizer : PreTrainedTokenizerFast,
    remove_columns : List[str] = []
) -> Tuple[Dict[str, Dataset], Dict[str, Dataset], Dict[str, Dataset], Dict[str, Dataset],  Dict[str, Dataset], Dict[str, Dataset]]:
    """
    Load datasets for training.

    Args:
        args: Arguments object.
        accelerator: Accelerator object.
        tokenizer: Tokenizer object.
        remove_columns: Columns to remove from the datasets.

    Returns:
        Dictionaries containing the raw and processed train, validation and target datasets.
    """
    raw_validation_datasets = {}
    processed_validation_datasets = {}
    raw_train_datasets = {}
    processed_train_datasets = {}
    train_languages = []

    # First check whether source custom dataset is specified; 
    # if yes the source dataset path should have train and validation datasets stored in arrow format
    with accelerator.main_process_first():
        if args.source_dataset_path is not None:
            raw_train_dataset = Dataset.from_file(args.source_dataset_path + "/train.arrow")
            raw_validation_dataset = Dataset.from_file(args.source_dataset_path + "/validation.arrow")
            # Check if the dataset has a language column
            if "language" not in raw_train_dataset.column_names:
                raw_train_dataset = raw_train_dataset.map(partial(add_language, language="unknown"), batched=True)
                raw_validation_dataset = raw_validation_dataset.map(partial(add_language, language="unknown"), batched=True)
            # Make different train and validation datasets for each language
            for language in list(set(raw_train_dataset["language"])):
                raw_train_datasets[language] = raw_train_dataset.filter(lambda x: x["language"] == language)
            for language in list(set(raw_validation_dataset["language"])):
                raw_validation_datasets[language] = raw_validation_dataset.filter(lambda x: x["language"] == language)
        else:
            if args.task_type in ['mt'] and args.dataset_name in ['opus100'] and args.get_all_en_configs:
                all_config_names = get_mt_config_names(args.dataset_name)
                source_languages = [config_name for config_name in all_config_names if 'en' in config_name]
                logger.info(f"Using all en configs: {source_languages}")
            else:
                assert args.source_languages is not None, "Source languages/dataset path should be specified"
                source_languages = args.source_languages.split(",")
            # Check if two letter code is used for udpos, if yes replace with full name
            if args.dataset_name == "udpos":
                if len(source_languages[0]) == 2:
                    source_languages = [UDPOS_ID_MAP[lang] for lang in source_languages]
            for source_language in source_languages:
                if args.dataset_name in ["udpos", "PAN-X"]:
                    train_dataset = load_dataset("xtreme", args.dataset_name + "." + source_language, split="train", cache_dir=args.cache_dir)
                    validation_dataset = load_dataset("xtreme", args.dataset_name + "." + source_language, split="validation", cache_dir=args.cache_dir)
                elif args.dataset_name == "tydiqa":
                    train_dataset = get_tydiqa_dataset(language=source_language, split="train", cache_dir=args.cache_dir)
                    validation_dataset = get_tydiqa_dataset(language=source_language, split="validation", cache_dir=args.cache_dir)
                else:
                    train_dataset = load_dataset(args.dataset_name, source_language, split="train", cache_dir=args.cache_dir)
                    try:
                        validation_dataset = load_dataset(args.dataset_name, source_language, split="validation", cache_dir=args.cache_dir)
                    except Exception as e:
                        print("Failed to load validation dataset, splitting train into 80/20 train/val", e)
                        train_dataset, validation_dataset = train_test_split(train_dataset, test_size=0.2)
                        train_dataset = Dataset.from_dict(train_dataset)
                        validation_dataset = Dataset.from_dict(validation_dataset)

                # Deduplicate train dataset: Token classification datasets (udpos and PAN-X) have duplicate examples
                if args.dataset_name in ["udpos", "PAN-X"]:
                    deduplicated_train_dataset = deduplicate_dataset(train_dataset, check_column="tokens")
                else:
                    deduplicated_train_dataset = train_dataset
            
                # Select a subset of the train dataset if needed
                if args.per_language_subset_size:
                    if len(deduplicated_train_dataset) > args.per_language_subset_size:
                        raw_train_dataset = deduplicated_train_dataset.select(range(args.per_language_subset_size))
                else:
                    raw_train_dataset = deduplicated_train_dataset
            
                if args.debug:
                    if len(raw_train_dataset) > 100:
                        raw_train_dataset = raw_train_dataset.select(range(100))
                    if len(validation_dataset) > 100:
                        validation_dataset = validation_dataset.select(range(100))
            
                # Remove columns from the datasets
                if remove_columns is None:
                    remove_columns = raw_train_dataset.column_names

                logger.info("Adding language to train dataset with dataset name %s and language %s", args.dataset_name, source_language)
                raw_train_dataset = raw_train_dataset.map(partial(add_language, language=source_language), batched=True)
                raw_validation_dataset = validation_dataset.map(partial(add_language, language=source_language), batched=True)
                if args.task_type in ["mt"]:
                    raw_train_dataset = raw_train_dataset.map(partial(recast_mt_features, dataset_name=args.dataset_name, language_pair=source_language), batched=True)
                    raw_validation_dataset = raw_validation_dataset.map(partial(recast_mt_features, dataset_name=args.dataset_name, language_pair=source_language), batched=True)
            
                # raw_train_datasets.append(raw_train_dataset)
                raw_train_datasets[source_language] = raw_train_dataset
                train_languages.append(source_language)
                raw_validation_datasets[source_language]=raw_validation_dataset

                logger.info(f"Loaded train and validation datasets for {source_language}")
                logger.info(f"Train dataset size: {len(raw_train_dataset)}")
                logger.info(f"Validation dataset size: {len(validation_dataset)}")

    
        # Preprocess train and validation datasets
        for language, raw_train_dataset in raw_train_datasets.items():
            if args.task_type in ["mt"]:
                src = language.split("-")[0]
                tgt = language.split("-")[1]
                if args.dataset_name == "opus100":
                    # If src is "en", swap src and tgt
                    if src == "en":
                        src, tgt = tgt, src
                    tokenizer.src_lang = MBART_ID_MAP[src]
                    tokenizer.tgt_lang = MBART_ID_MAP[tgt]
                elif args.dataset_name == "facebook/flores":
                    # If src is "eng", swap src and tgt
                    if src.startswith("eng"):
                        src, tgt = tgt, src
                    tokenizer.src_lang = flores_id_map[src]
                    tokenizer.tgt_lang = flores_id_map[tgt]
            processed_train_datasets[language] = raw_train_dataset.map(
                partial(preprocess_datasets,
                    args=args,
                    remove_columns=remove_columns,
                    tokenizer=tokenizer,
                    dataset=args.dataset_name,
                    padding="max_length" if args.pad_to_max_length else False,
                    max_seq_length=args.max_seq_length,
                    train=True),
                    batched=True, 
                    remove_columns=remove_columns,
                )
            logger.info(f"Processed train dataset size for {language}: {len(processed_train_datasets[language])}")
        
        for language, raw_validation_dataset in raw_validation_datasets.items():
            if args.task_type in ["mt"]:
                if args.dataset_name == "opus100":
                    src = language.split("-")[0]
                    tgt = language.split("-")[1]
                    if args.dataset_name == "opus100":
                        # If src is "en", swap src and tgt
                        if src == "en":
                            src, tgt = tgt, src
                        tokenizer.src_lang = MBART_ID_MAP[src]
                        tokenizer.tgt_lang = MBART_ID_MAP[tgt]
                    elif args.dataset_name == "facebook/flores":
                        # If src is "eng", swap src and tgt
                        if src.startswith("eng"):
                            src, tgt = tgt, src
                        tokenizer.src_lang = flores_id_map[src]
                        tokenizer.tgt_lang = flores_id_map[tgt] 
            processed_validation_datasets[language] = raw_validation_dataset.map(
                partial(preprocess_datasets,
                    args=args,
                    remove_columns=remove_columns,
                    tokenizer=tokenizer,
                    dataset=args.dataset_name,
                    padding="max_length" if args.pad_to_max_length else False,
                    max_seq_length=args.max_seq_length,
                    train=False),
                    batched=True, 
                    remove_columns=remove_columns,
                )
            logger.info(f"Processed validation dataset size for {language}: {len(processed_validation_datasets[language])}")
    
        # For target, first check whether target custom dataset is specified;
        # if yes the target dataset path should have an unlabelled target dataset in arrow format
        raw_target_datasets = {}
        processed_target_datasets = {}
        if args.target_dataset_path is not None:
            raw_target_dataset = Dataset.from_file(args.target_dataset_path + "/target.arrow")
            # Check if the dataset has a language column
            if "language" not in raw_target_dataset.column_names:
                raw_target_dataset = raw_target_dataset.map(partial(add_language, language="unknown"), batched=True)
            # Make different target datasets for each language
            for language in raw_target_dataset["language"]:
                raw_target_datasets[language] = raw_target_dataset.filter(lambda x: x["language"] == language)
        else:
            target_remove_columns = None
            if args.target_dataset_name is not None:
                target_dataset_name = args.target_dataset_name
                # Get target_dataset_config from the dataset_name
                if args.target_dataset_name != args.dataset_name:
                    with open(args.dataset_config_file, "r") as f:
                        target_dataset_config = yaml.safe_load(f)[args.dataset_name]
                        # Assuming that everything else, like labels, metrics, task_type, should be the same as source dataset
                        if "remove_columns" in target_dataset_config:        
                            target_remove_columns = target_dataset_config["remove_columns"]
            else:
                target_dataset_name = args.dataset_name
                target_remove_columns = remove_columns
            
                        
            assert args.target_languages is not None, "Target languages/dataset path should be specified"
            target_languages = args.target_languages.split(",")
            if target_dataset_name == "udpos":
                if len(target_languages[0]) == 2:
                    target_languages = [UDPOS_ID_MAP[lang] for lang in target_languages]
            for target_language in target_languages:
                
                if target_dataset_name in ["udpos", "PAN-X"]:
                    raw_target_dataset = load_dataset("xtreme", target_dataset_name + "." + target_language, split="validation", cache_dir=args.cache_dir)
                elif target_dataset_name == "tydiqa":
                    raw_target_dataset = get_tydiqa_dataset(language=target_language, split="validation", cache_dir=args.cache_dir)
                else:
                    if target_dataset_name == "facebook/flores":
                        split = "dev"
                    else:
                        split = "validation"
                    raw_target_dataset = load_dataset(target_dataset_name, target_language, split=split, cache_dir=args.cache_dir)

                # Deduplicate target dataset: Token classification datasets (udpos and PAN-X) have duplicate examples
                if target_dataset_name in ["udpos", "PAN-X"]:
                    deduplicated_target_dataset = deduplicate_dataset(raw_target_dataset, check_column="tokens")
                else:
                    deduplicated_target_dataset = raw_target_dataset
            
                if args.debug:
                    if len(deduplicated_target_dataset) > 100:
                        deduplicated_target_dataset = deduplicated_target_dataset.select(range(100))
            
                # Add language column
                deduplicated_target_dataset = deduplicated_target_dataset.map(partial(add_language, language=target_language), batched=True)
                if args.task_type in ["mt"]:
                    deduplicated_target_dataset = deduplicated_target_dataset.map(partial(recast_mt_features, dataset_name=target_dataset_name,language_pair=target_language), batched=True)
            
                raw_target_datasets[target_language] = deduplicated_target_dataset

                # If target_remove_columns is None, then get column names
                if target_remove_columns is None:
                    target_remove_columns = deduplicated_target_dataset.column_names
    
        # Preprocess target datasets
        for language, raw_target_dataset in raw_target_datasets.items():
            if args.task_type in ["mt"]:
                src = language.split("-")[0]
                tgt = language.split("-")[1]
                if target_dataset_name == "opus100":
                    # If src is "en", swap src and tgt
                    if src == "en":
                        src, tgt = tgt, src
                    tokenizer.src_lang = MBART_ID_MAP[src]
                    tokenizer.tgt_lang = MBART_ID_MAP[tgt]
                elif target_dataset_name == "facebook/flores":
                    # If src is "eng", swap src and tgt
                    if src.startswith("eng"):
                        src, tgt = tgt, src
                    tokenizer.src_lang = flores_id_map[src]
                    tokenizer.tgt_lang = flores_id_map[tgt]   
            processed_target_datasets[language] = raw_target_dataset.map(
                partial(preprocess_datasets,
                    args=args,
                    remove_columns=target_remove_columns,
                    tokenizer=tokenizer,
                    dataset=target_dataset_name,
                    padding="max_length" if args.pad_to_max_length else False,
                    max_seq_length=args.max_seq_length,
                    train=False),
                    batched=True, 
                    remove_columns=target_remove_columns,
                )
            logger.info(f"Processed target dataset size for {language}: {len(processed_target_datasets[language])}")

    return raw_train_datasets, processed_train_datasets, raw_validation_datasets, \
        processed_validation_datasets, raw_target_datasets, processed_target_datasets


def load_test_datasets(
    args: argparse.Namespace,
    accelerator: Accelerator, 
    tokenizer: PreTrainedTokenizerFast,
    remove_columns: List[str]
) -> Tuple[Dict[str, Dataset], Dict[str, Dataset]]:
    """Load test datasets for evaluation.

    Args:
        args (argparse.Namespace): Command-line arguments.
        accelerator (Any): Accelerator for distributed training.
        tokenizer (Any): Tokenizer for preprocessing.
        remove_columns (List[str]): Columns to remove from the dataset.

    Returns:
        Tuple[Dict[str, Dataset], Dict[str, Dataset]]: A tuple of raw and processed test datasets.
    """
    with accelerator.main_process_first():
        raw_test_datasets = {}
        processed_test_datasets = {}
        if args.target_dataset_path is not None:
            raw_test_dataset = Dataset.from_file(args.target_dataset_path + "/test.arrow")
            # Check if the dataset has a language column
            if "language" not in raw_test_dataset.column_names:
                raw_test_dataset = raw_test_dataset.map(partial(add_language, language="unknown"), batched=True)
            # Make different target datasets for each language
            for language in raw_test_dataset["language"]:
                raw_test_datasets[language] = raw_test_dataset.filter(lambda x: x["language"] == language)
        else:
            target_remove_columns = None
            if args.target_dataset_name is not None:
                target_dataset_name = args.target_dataset_name
                # Get target_dataset_config from the dataset_name
                if args.target_dataset_name != args.dataset_name:
                    with open(args.dataset_config_file, "r") as f:
                        target_dataset_config = yaml.safe_load(f)[args.dataset_name]
                        # Assuming that everything else, like labels, metrics, task_type, should be the same as source dataset
                        if "remove_columns" in target_dataset_config:        
                            target_remove_columns = target_dataset_config["remove_columns"]
            else:
                target_dataset_name = args.dataset_name
                target_remove_columns = remove_columns
            assert args.target_languages is not None, "Target languages/dataset path should be specified"
            target_languages = args.target_languages.split(",")
            if target_dataset_name == "udpos":
                if len(target_languages[0]) == 2:
                    target_languages = [UDPOS_ID_MAP[lang] for lang in target_languages]
            for target_language in target_languages:
                if target_dataset_name in ["udpos", "PAN-X"]:
                    raw_test_dataset = load_dataset("xtreme", target_dataset_name + "." + target_language, split="test", cache_dir=args.cache_dir)
                elif target_dataset_name == "tydiqa":
                    raw_test_dataset = get_tydiqa_dataset(language=target_language, split="test", cache_dir=args.cache_dir)
                else:
                    if target_dataset_name == "facebook/flores":
                        split = "devtest"
                    else:
                        split = "test"
                    raw_test_dataset = load_dataset(target_dataset_name, target_language, split=split, cache_dir=args.cache_dir)
        
                # Take samples if debug
                if args.debug:
                    if len(raw_test_dataset) > 100:
                        raw_test_dataset = raw_test_dataset.select(range(100))
                # Add language column
                with accelerator.main_process_first():
                    raw_test_dataset = raw_test_dataset.map(partial(add_language, language=target_language), batched=True)
                    if args.task_type in ["mt"]:
                        raw_test_dataset = raw_test_dataset.map(partial(recast_mt_features, dataset_name=target_dataset_name, language_pair=target_language), batched=True)

                raw_test_datasets[target_language] = raw_test_dataset

                # Remove columns from the datasets
                if target_remove_columns is None:
                    target_remove_columns = raw_test_dataset.column_names

        for language, raw_test_dataset in raw_test_datasets.items():
            if args.task_type in ["mt"]:
                src = language.split("-")[0]
                tgt = language.split("-")[1]
                if target_dataset_name == "opus100":
                    # If src is "en", swap src and tgt
                    if src == "en":
                        src, tgt = tgt, src
                    tokenizer.src_lang = MBART_ID_MAP[src]
                    tokenizer.tgt_lang = MBART_ID_MAP[tgt]
                elif target_dataset_name == "facebook/flores":
                    # If src is "eng", swap src and tgt
                    if src.startswith("eng"):
                        src, tgt = tgt, src
                    tokenizer.src_lang = flores_id_map[src]
                    tokenizer.tgt_lang = flores_id_map[tgt]
            processed_test_datasets[language] = raw_test_dataset.map(
                partial(preprocess_datasets,
                    args=args,
                    remove_columns=target_remove_columns,
                    tokenizer=tokenizer,
                    dataset=target_dataset_name,
                    padding="max_length" if args.pad_to_max_length else False,
                    max_seq_length=args.max_seq_length,
                    train=False),
                    batched=True, 
                    remove_columns=target_remove_columns,
                )
            logger.info(f"Processed test dataset size for {language}: {len(processed_test_datasets[language])}")
    
    return raw_test_datasets, processed_test_datasets

def preprocess_datasets(
    examples: Dict[str, Any],
    args: argparse.Namespace,
    remove_columns: List[str],
    tokenizer: PreTrainedTokenizerFast,
    dataset: str, 
    padding: str, 
    max_seq_length: int, 
    train: bool = True,
    add_labels: bool = True, 
) -> Dict[str, Any]:
    """
    Preprocess datasets.

    Args:
        args (argparse.Namespace): Command-line arguments.
        examples (Dict[str, Any]): Input examples.
        tokenizer: Tokenizer for preprocessing.
        dataset (str): Dataset type.
        padding (str): Padding type.
        max_seq_length (int): Maximum sequence length.
        add_labels (bool, optional): Whether to add labels. Defaults to True.
        train (bool, optional): Whether it is a training dataset. Defaults to True.

    Returns:
        Dict[str, Any]: Processed examples.
    """
    if dataset == "xnli":
        for i, premise in enumerate(examples["premise"]):
            if not isinstance(premise, str):
                examples["premise"][i] = str(premise)

        for i, hypothesis in enumerate(examples["hypothesis"]):
            if not isinstance(hypothesis, str):
                examples["hypothesis"][i] = str(hypothesis)

        processed_examples = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=padding,
            max_length=max_seq_length,
            truncation=True,
        )

        if add_labels:
            processed_examples["labels"] = [int(label) for label in examples["label"]]

    elif dataset == 'tydiqa':
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
                questions,
                examples["context"],
                max_length=max_seq_length,
                truncation="only_second",
                stride=max_seq_length//2,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

        if train:
            offset_mapping = inputs.pop("offset_mapping")
            sample_map = inputs.pop("overflow_to_sample_mapping")
            answers = examples["answers"]
            start_positions = []
            end_positions = []
            example_ids = []
       
            for i, offset in enumerate(offset_mapping):
                sample_idx = sample_map[i]
                example_ids.append(examples["id"][sample_idx])
                answer = answers[sample_idx]
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context, label is (0, 0)
                if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)
            
            for key, values in examples.items():
                inputs[key] = [values[i] for i in sample_map]
                inputs["start_positions"] = start_positions
                inputs["end_positions"] = end_positions
                inputs["example_id"] = example_ids
        
        else:
            sample_map = inputs.pop("overflow_to_sample_mapping")
            example_ids = []

            for i in range(len(inputs["input_ids"])):
                sample_idx = sample_map[i]
                example_ids.append(examples["id"][sample_idx])

                sequence_ids = inputs.sequence_ids(i)
                offset = inputs["offset_mapping"][i]
                inputs["offset_mapping"][i] = [
                    o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
                ]
            for key, values in examples.items():
                inputs[key] = [values[i] for i in sample_map]
                inputs["example_id"] = example_ids

        processed_examples = {}
        for key, values in inputs.items():
            if key not in remove_columns:
                processed_examples[key] = values

    elif dataset == "udpos" or dataset == "PAN-X":
        processed_examples = tokenizer(examples["tokens"], 
                                        padding=padding,
                                        max_length=max_seq_length,
                                        truncation=True,
                                        is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["pos_tags"] if dataset == "udpos" else examples["ner_tags"]):
            word_ids = processed_examples.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        processed_examples["labels"] = labels
    
    elif args.task_type in ["mt"]:
        inputs = [ex["source"] for ex in examples["translation"]]
        targets = [ex["target"] for ex in examples["translation"]]
        
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    else:
        raise ValueError("Invalid dataset type")

    return processed_examples


def add_language(examples: Dict[str, Any], language: str) -> Dict[str, Any]:
    """
    Add language to dataset.

    Args:
        examples (Dict[str, Any]): Input examples.
        language (str): Language to add.

    Returns:
        Dict[str, Any]: Examples with added language.
    """
    examples_length = len(list(examples.values())[0])
    examples["language"] = [language for _ in range(examples_length)]
    return examples


def deduplicate_dataset(dataset: Dataset, check_column: str) -> Dataset:
    """
    Deduplicate dataset.

    Args:
        dataset (Dataset): Input dataset.
        check_column (str): Column to check for deduplication.

    Returns:
        Dataset: Deduplicated dataset.
    """
    logger.info("Dataset size before deduplication: {}".format(len(dataset)))
    deduplicated_dataset = {}
    dataset_dict = dataset.to_dict()
    for key in dataset_dict.keys():
        deduplicated_dataset[key] = []
    for i in range(len(dataset_dict[check_column])):
        if dataset_dict[check_column][i] not in deduplicated_dataset[check_column]:
            for key in dataset_dict.keys():
                deduplicated_dataset[key].append(dataset_dict[key][i])
    deduplicated_dataset = Dataset.from_dict(deduplicated_dataset)
    logger.info("Dataset size after deduplication: {}".format(len(deduplicated_dataset)))
    return deduplicated_dataset
