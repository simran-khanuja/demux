from datasets import load_dataset

def csv_to_arrow(csv_path, arrow_directory):
    dataset = load_dataset('csv', data_files=csv_path)
    # can you add a language column to the dataset?
    dataset = dataset.map(lambda example: {'language': 'mya_Mymr-eng_Latn'})

    # Save the dataset in Arrow format
    dataset.save_to_disk(arrow_directory)


if __name__ == "__main__":
    csv_path = "/home/skhanuja/demux/social-burmese.csv"
    arrow_file_path = "/home/skhanuja/demux/target-burmese"
    csv_to_arrow(csv_path, arrow_file_path)
