import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14  # Increased font size

def parse_args():
    parser = argparse.ArgumentParser(description='Plot data for language models.')
    parser.add_argument('--filepath_panx', type=str, default='./src/evaluation/panx_urdu_test.csv',
                        help='File path to PANX data')
    parser.add_argument('--filepath_udpos', type=str, default='./src/evaluation/udpos_urdu_test.csv',
                        help='File path to UDPOS data')
    parser.add_argument('--filepath_xnli', type=str, default='./src/evaluation/xnli_ur_test.csv',
                        help='File path to XNLI data')
    parser.add_argument('--filepath_tydiqa', type=str, default='./src/evaluation/tydiqa_fi_test.csv',
                        help='File path to TyDiQA data')

    return parser.parse_args()

def plot_data(args):
    output_directory = "./src/evaluation"
    datasets = {
        'panx': {'filepath': args.filepath_panx, 'title': 'PANX'},
        'udpos': {'filepath': args.filepath_udpos, 'title': 'UDPOS'},
        'xnli': {'filepath': args.filepath_xnli, 'title': 'XNLI'},
        'tydiqa': {'filepath': args.filepath_tydiqa, 'title': 'TyDiQA'}
    }

    strategy_labels = {
        "average": "AVG-DIST",
        "egalitarian": "EGAL",
        "gold": "GOLD",
        "knn": "KNN-UNC",
        "random": "RAND",
        "uncertainty": "UNC"
    }

    strategy_order = ["gold", "knn", "average", "uncertainty", "egalitarian", "random"]

    sns.set_style('darkgrid')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Create a 2x2 grid of subplots

    for i, (dataset_key, dataset_info) in enumerate(datasets.items()):
        filepath = dataset_info['filepath']
        dataset_title = dataset_info['title'] + " : Finnish" if dataset_key == 'tydiqa' else dataset_info['title'] + " : Urdu"

        df = pd.read_csv(filepath)
        df['Iteration'] = df['Iteration'].apply(lambda x: int(x.split()[0]))
        languages = [col for col in df.columns if col.endswith("_test")]

        ax = axes[i // 2, i % 2]  # Get the appropriate subplot axes

        for language in languages:
            for strategy_name in strategy_order:
                subset = df[df.Strategy.str.startswith(strategy_name)]
                ax.plot(subset.Iteration, subset[language], marker='o', markersize=10,
                        label=strategy_labels[strategy_name])

        ax.set_title(dataset_title, fontsize=26)
        ax.set_xlabel('Round', fontsize=22)
        ax.set_ylabel('Performance (%)', fontsize=22)
        ax.tick_params(axis='both', labelsize=16)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # Display y-axis ticks as whole numbers

    # Create a single legend for the whole plot
    handles, labels = ax.get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, ncol=2, loc='upper left', prop={'size': 14})

    fig.tight_layout(rect=[0, 0, 0.9, 0.97])  # Adjust spacing between subplots and main title

    fig_filepath = os.path.join(output_directory, 'performance_comparison.png')
    os.makedirs(os.path.dirname(fig_filepath), exist_ok=True)
    fig.savefig(fig_filepath, dpi=600, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    args = parse_args()
    plot_data(args)
