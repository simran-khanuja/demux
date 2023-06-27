from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import os
from accelerate.logging import get_logger
import argparse
import numpy as np
import pandas as pd
import glob

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding_path",
        type = str,
        help = "Path to stored embeddings",
    )
    parser.add_argument(
        "--algorithm",
        type = str,
        help = "Dimensionality reduction algorithm to use",
        choices=['tsne','pca'],
        default='pca'
    )
    parser.add_argument(
        "--unlabelled_source_sample_size",
        type = int,
        default=50000,
        help = "Random subsample of source train (unlabelled) datapoints to show"
    )
    parser.add_argument(
        "--unlabelled_target_sample_size",
        type = int,
        default=1000,
        help = "Random subsample of target (unlabelled) datapoints to show"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        help="(optional) Comma-separated list of strategies to collect results for. If not provided, will take all strategies in model_path",
        default=None
    )
    parser.add_argument(
        "--al_round_to_show",
        type=int,
        help="The AL round to visualize data selection for; by default it takes the last round",
        default=None
    )
    
    args=parser.parse_args()
    return args

def main(args):
    """ Main function."""
    embedding_path = args.embedding_path
    # Check if embedding path exists
    if not os.path.exists(embedding_path):
        raise ValueError(f"Embedding path {embedding_path} does not exist")

    if args.strategies:
        strategies = args.strategies.split(",")
    else:
        # Iterate over model path dirs and collect all strategies
        strategies = []
        for dir in os.listdir(embedding_path):
            if os.path.isdir(os.path.join(embedding_path, dir)):
                strategies.append(dir)
    
    # if random in strategies, remove it
    if 'random' in strategies:
        strategies.remove('random')
    
    for strategy in strategies:
        strategy_path = os.path.join(embedding_path, strategy)
        if args.al_round_to_show:
            iteration_embedding_path = os.path.join(strategy_path, "iter_" + str(args.al_round_to_show))
        else:
            # Find last created iteration directory of strategy (by timestamp); not file
            iterations = [d for d in os.scandir(strategy_path) if d.is_dir()]
            last_iteration = max(iterations, key=lambda d: d.stat().st_mtime, default=None)
            iteration_embedding_path = os.path.join(strategy_path, last_iteration.name)

        print("Visualizing embeddings for strategy: %s, iteration: %s" % (strategy, iteration_embedding_path))
    
        train_embeddings = torch.load(iteration_embedding_path  + "/train_embeddings.pt", map_location=torch.device('cpu')).cpu().numpy()
        selected_train_embeddings_path = glob.glob(iteration_embedding_path  + "/selected_train_*")[0]
        selected_train_embeddings = torch.load(selected_train_embeddings_path, map_location=torch.device('cpu')).cpu().numpy()
        target_embeddings = torch.load(iteration_embedding_path  + "/valid_embeddings.pt", map_location=torch.device('cpu')).cpu().numpy()

        choice = args.algorithm
        train_and_target_embeddings = np.concatenate((train_embeddings, target_embeddings))

        
        # Set font style and size
        plt.rcParams["font.family"] = "Spectral"
        plt.rcParams["font.size"] = 18
        
        # Color palette
        palette = sns.color_palette("BuPu_r", n_colors=3)
        sns.set_palette("pastel")

        if choice == 'pca':
            # Perform PCA to reduce the dimensions to 2
            logger.info("Performing PCA to reduce the dimensions to 2..." )
            pca = PCA(n_components=2)
            
            train_and_target_fit = pca.fit(train_and_target_embeddings)
            # print length of train and target embeddings and selected train embeddings
            print("Train embeddings shape: %s" % str(train_embeddings.shape))
            print("Target embeddings shape: %s" % str(target_embeddings.shape))
            print("Selected train embeddings shape: %s" % str(selected_train_embeddings.shape))

            unlabelled_source_sample_size = int(args.unlabelled_source_sample_size) if int(args.unlabelled_source_sample_size) < len(train_embeddings) else len(train_embeddings)
            unlabelled_target_sample_size = int(args.unlabelled_target_sample_size) if int(args.unlabelled_target_sample_size) < len(target_embeddings) else len(target_embeddings)
            # Select random args.unlabelled_source_sample_size datapoints from the train embeddings to show
            train_to_show = train_embeddings[np.random.choice(train_embeddings.shape[0], unlabelled_source_sample_size, replace=False), :]
            target_to_show = target_embeddings[np.random.choice(target_embeddings.shape[0], unlabelled_target_sample_size, replace=False), :]
            train_to_show_pca = train_and_target_fit.transform(train_to_show)
            target_pca = train_and_target_fit.transform(target_to_show)
            selected_train_pca = train_and_target_fit.transform(selected_train_embeddings)

            # Plot all three above with different colours
            sns.set(style='white')
            plt.figure(figsize=(16, 12))
            df1 = pd.DataFrame(selected_train_pca,columns=['x','y'])
            df1['type'] = 'Selected Train'
            df2 = pd.DataFrame(train_to_show_pca,columns=['x','y'])
            df2['type'] = 'Train'
            df3 = pd.DataFrame(target_pca,columns=['x','y'])
            df3['type'] = 'Target'
            df = pd.concat([df2, df1, df3])
            palette = ["#7DF9FF", "#0047AB", "#F72585"]  #40E0D0
            sns.set_palette(palette=palette)
            ax=sns.scatterplot(x='x',y='y',hue='type',palette=palette,data=df,s=75, alpha=0.5, marker='o')
            ax.set_ylabel("PCA-1", fontsize=24)
            ax.set_xlabel("PCA-2", fontsize=24)
            ax.tick_params(axis='x', labelsize=24)
            ax.tick_params(axis='y', labelsize=24)
            plt.legend(fontsize=24,  markerscale=2)
            plt.title("PCA Plot of Embedding Matrices", fontsize=36)
            plt.grid(linestyle='--', linewidth=1, alpha=0.5)
            plt.savefig(iteration_embedding_path + "/pca_plot.png")
            plt.clf()


        elif choice == 'tsne':
            # Perform T-SNE to reduce the dimensions to 2
            logger.info("Performing T-SNE to reduce the dimensions to 2..." )
            tsne = TSNE(n_components=2, perplexity=20, n_iter=300, init='pca')

            train_and_target_fit = tsne.fit(train_and_target_embeddings)
            
            # Select random args.unlabelled_source_sample_size datapoints from the train embeddings to show
            train_to_show = train_embeddings[np.random.choice(train_embeddings.shape[0], args.unlabelled_source_sample_size, replace=False), :]
            train_to_show_pca = train_and_target_fit.transform(train_to_show)
            target_pca = train_and_target_fit.transform(target_embeddings)
            selected_train_pca = train_and_target_fit.transform(selected_train_embeddings)

            # Plot all three above with different colours
            sns.set(style='white')
            plt.figure(figsize=(16, 12))
            df1 = pd.DataFrame(selected_train_pca,columns=['x','y'])
            df1['type'] = 'Selected Train'
            df2 = pd.DataFrame(train_to_show_pca,columns=['x','y'])
            df2['type'] = 'Train'
            df3 = pd.DataFrame(target_pca,columns=['x','y'])
            df3['type'] = 'Target'
            df = pd.concat([df2, df1, df3])
            palette = ["#7DF9FF", "#0047AB", "#F72585"]  #40E0D0
            sns.set_palette(palette=palette)
            ax=sns.scatterplot(x='x',y='y',hue='type',palette=palette,data=df,s=75, alpha=0.5, marker='o')
            ax.set_ylabel("PCA-2", fontsize=30)
            ax.set_xlabel("PCA-1", fontsize=30)
            ax.tick_params(axis='x', labelsize=24)
            ax.tick_params(axis='y', labelsize=24)
            plt.legend(fontsize=24,  markerscale=2)
            plt.title("TSNE Plot of Embedding Matrices", fontsize=36)
            plt.grid(linestyle='--', linewidth=1, alpha=0.5)
            plt.savefig(iteration_embedding_path + "/tsne_plot.png")
            plt.clf()

if __name__ == "__main__":
    args=parse_args()
    main(args)
    


