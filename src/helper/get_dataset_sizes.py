import csv
import argparse
from datasets import load_dataset
from datasets import Dataset
from accelerate.logging import get_logger
from data_utils import TYDIQA_ID_MAP

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get dataset sizes for a given split.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        default="udpos"
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Data split for which you want sizes",
        default="validation",
        choices=['train', 'validation', 'test']
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path where we want to store the results",
        default="src/helper/validation_counts"
    )
    args = parser.parse_args()
    return args


def main():
    """ Main function."""
    args = parse_args()
    dataset_name = args.dataset_name
    split = args.split

    if dataset_name in ['udpos', 'PAN-X']:
        # Get the dataset builder for the xtreme dataset
        dataset_base = 'xtreme'
        all_configs = [
            'XNLI', 'tydiqa', 'PAN-X.af', 'PAN-X.ar', 'PAN-X.bg', 'PAN-X.bn', 'PAN-X.de','PAN-X.el', 'PAN-X.en', 
            'PAN-X.es', 'PAN-X.et', 'PAN-X.eu', 'PAN-X.fa', 'PAN-X.fi', 'PAN-X.fr', 'PAN-X.he', 'PAN-X.hi', 'PAN-X.hu',
            'PAN-X.id', 'PAN-X.it', 'PAN-X.ja', 'PAN-X.jv', 'PAN-X.ka', 'PAN-X.kk', 'PAN-X.ko', 'PAN-X.ml', 'PAN-X.mr',
            'PAN-X.ms', 'PAN-X.my', 'PAN-X.nl', 'PAN-X.pt', 'PAN-X.ru', 'PAN-X.sw', 'PAN-X.ta', 'PAN-X.te', 'PAN-X.th',
            'PAN-X.tl', 'PAN-X.tr', 'PAN-X.ur', 'PAN-X.vi', 'PAN-X.yo', 'PAN-X.zh', 'udpos.Afrikaans', 'udpos.Arabic',
            'udpos.Basque', 'udpos.Bulgarian', 'udpos.Dutch', 'udpos.English', 'udpos.Estonian', 'udpos.Finnish', 'udpos.French',
            'udpos.German', 'udpos.Greek', 'udpos.Hebrew', 'udpos.Hindi', 'udpos.Hungarian', 'udpos.Indonesian', 'udpos.Italian', 
            'udpos.Japanese', 'udpos.Kazakh', 'udpos.Korean', 'udpos.Chinese', 'udpos.Marathi', 'udpos.Persian', 'udpos.Portuguese', 
            'udpos.Russian', 'udpos.Spanish', 'udpos.Tagalog', 'udpos.Tamil', 'udpos.Telugu', 'udpos.Thai', 'udpos.Turkish', 
            'udpos.Urdu', 'udpos.Vietnamese', 'udpos.Yoruba'
        ]

        # Filter the configurations to include only those starting with dataset_name
        configs = [config for config in all_configs if config.startswith(dataset_name)]
        samples_counts = {}

        # Loop over all configurations
        for config in configs:
            # Load the dataset for the specific configuration
            dataset = load_dataset(dataset_base, config)

            # Remove duplicates
            check_column = "tokens"
            deduplicated_dataset = {}
            if split not in dataset.keys():
                continue
            dataset_dict = dataset[split].to_dict()
            for key in dataset_dict.keys():
                deduplicated_dataset[key] = []
            for i in range(len(dataset_dict[check_column])):
                if dataset_dict[check_column][i] not in deduplicated_dataset[check_column]:
                    for key in dataset_dict.keys():
                        deduplicated_dataset[key].append(dataset_dict[key][i])
            deduplicated_dataset = Dataset.from_dict(deduplicated_dataset)

            samples_counts[config] = {}
            samples_counts[config][split] = len(deduplicated_dataset)
    
    elif dataset_name == 'xnli':
        languages = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
        samples_counts = {}

        # Loop over all languages
        for language in languages:
            # Load the dataset for the specific language
            dataset = load_dataset('xnli', language)[split]
            samples_counts[language] = {}
            samples_counts[language][split] = len(dataset)
    
    elif dataset_name == 'tydiqa':
        if split == 'test':
            base_dataset=load_dataset('tydiqa', 'secondary_task')['validation']
        else:
            train_validation_dataset=load_dataset('tydiqa', 'secondary_task')['train'].train_test_split(test_size=0.2, seed=42)
            if split == 'validation':
                base_dataset = train_validation_dataset['test']
            else:
                base_dataset = train_validation_dataset['train']

        samples_counts = {}
        languages = {item['id'].split('-')[0] for item in base_dataset}
        dataset_dict = {
            language: base_dataset.filter(lambda x: language in x['id']) for language in languages
        }
        dataset_dict = {TYDIQA_ID_MAP[language]: dataset for language, dataset in dataset_dict.items()}
        for language, dataset in dataset_dict.items():
            samples_counts[language] = {}
            samples_counts[language][split] = len(dataset)

    # Save the counts to a CSV file
    output_file = f'{args.output_path}/{dataset_name}_{split}_counts.csv'
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['language', split]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for language, counts in samples_counts.items():
            row = {'language': language, **counts}
            writer.writerow(row)
    logger.info(f"Saved counts to {output_file}")


if __name__ == "__main__":
    main()



