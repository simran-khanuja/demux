import argparse
import os
import numpy as np

from accelerate.logging import get_logger
from datasets import disable_caching
import json

logger = get_logger(__name__)
disable_caching()

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Collect results from a set of experiments")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Model base path which contains runs for all strategies",
        default="./outputs/models/xlm-roberta-large/udpos/target_lp/10000/2e-05_10_seed_42"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        help="(optional) Comma-separated list of strategies to collect results for. If not provided, will take all strategies in model_path",
        default=None
    )
    parser.add_argument(
        "--al_rounds",
        type=int,
        help="Number of AL rounds to collect results for",
        default=5
    )

    args = parser.parse_args()
    return args


def calculate_deltas(output_path, results):
    """
    Calculate deltas between strategies and write to a CSV file.
    """
    # First check if values in results are between 0 and 1, if yes then convert to percentages
    for strategy in results:
        for i in range(len(results[strategy])):
            if results[strategy][i] < 1:
                results[strategy][i] = results[strategy][i] * 100

    # Extracting required values
    for strategy in results:
        if strategy.startswith("gold"): gold_values = np.array(results[strategy]) 
        if strategy.startswith("egal"): egal_values = np.array(results[strategy])
        if strategy.startswith("knn"): knn_values = np.array(results[strategy]) 
        if strategy.startswith("average"): avg_dist_values = np.array(results[strategy])
        if strategy.startswith("uncertainty"): uncertainty_values = np.array(results[strategy]) 

    # Calculate deltas
    deltas = {
        'knn_gold_delta': (knn_values - gold_values).tolist(),
        'knn_egal_delta': (knn_values - egal_values).tolist(),
        'avg_dist_gold_delta': (avg_dist_values - gold_values).tolist(),
        'avg_dist_egal_delta': (avg_dist_values - egal_values).tolist(),
        'uncertainty_gold_delta': (uncertainty_values - gold_values).tolist(),
        'uncertainty_egal_delta': (uncertainty_values - egal_values).tolist()
    }

    # Calculate averages of deltas
    averages = {key: np.mean(value) for key, value in deltas.items()}

    # Write deltas and averages to a new file
    with open(os.path.join(output_path, "deltas_last_iter.csv"), "w") as f:
        f.write("delta,average\n")
        for key,_ in deltas.items():
            f.write(f"{key},{averages[key]}\n")
    
    

def main():
    """ Main function."""
    args = parse_args()
    model_path = args.model_path

    # Check if model path exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    if args.strategies:
        strategies = args.strategies.split(",")
    else:
        # Iterate over model path dirs and collect all strategies
        strategies = []
        for dir in os.listdir(model_path):
            if os.path.isdir(os.path.join(model_path, dir)):
                strategies.append(dir)
    print("Model path: " + args.model_path)
    print("Strategies: " + str(strategies))
    # If first strategy is gold, move it to the end
    if strategies[0].startswith("gold"):
        strategies = strategies[1:] + strategies[:1]

    # Write outputs of validation and test to CSV file
    output_path = model_path
    fp_test = open(os.path.join(output_path, "results_test.csv"), "w")
    fp_val = open(os.path.join(output_path, "results_val.csv"), "w")

    for iteration in range(1, args.al_rounds + 1):
        print(f"Collecting results for iteration {iteration}")
        for idx, strategy in enumerate(strategies):
            print(f"Collecting results for strategy {strategy}")
            strategy_model_path = model_path + f"/{strategy}"
            iteration_model_path = strategy_model_path + f"/iter_{iteration}"
            if os.path.exists(os.path.join(iteration_model_path, "all_results.json")):
                with open(os.path.join(iteration_model_path, "all_results.json"), "r") as f:
                    metrics = json.load(f)
                    # Write header
                    if iteration == 1 and idx == 0:
                        all_vals = f"iteration,strategy"
                        all_tests = f"iteration,strategy"
                        for metric in metrics:
                            if "_val" in metric:
                                all_vals += f",{metric}"
                            elif "_test" in metric:
                                all_tests += f",{metric}"
                        fp_test.write(f"{all_tests}\n")
                        fp_val.write(f"{all_vals}\n")
                    all_vals = f"iter_{iteration},{strategy}"
                    all_tests = f"iter_{iteration},{strategy}"
                    for metric in metrics:
                        if "_val" in metric:
                            all_vals += f",{metrics[metric]}"
                        elif "_test" in metric:
                            all_tests += f",{metrics[metric]}"
                    fp_val.write(f"{all_vals}\n")
                    fp_test.write(f"{all_tests}\n")
            else:
                print(f"No results found for strategy {strategy} at iteration {iteration}")
                continue

    fp_test.close()
    fp_val.close()

    last_iter_results = {strategy: [] for strategy in strategies}
    
    # calculate deltas for last iteration
    with open(os.path.join(output_path, "results_test.csv"), "r") as f:
        for line in f:
            if line.startswith(f"iter_{args.al_rounds}"):
                parts = line.strip().split(',')
                strategy, values = parts[1], parts[2:]
                print(strategy, values)
                if strategy in last_iter_results:
                    last_iter_results[strategy] = list(map(float, values))
    print(last_iter_results)
    calculate_deltas(output_path, last_iter_results)
    
if __name__ == "__main__":
    main()
