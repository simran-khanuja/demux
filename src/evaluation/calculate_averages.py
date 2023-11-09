import argparse
import pandas as pd
import numpy as np
import os

def calculate_averages_and_variance(directory, delta_metrics, strategy_metrics, seeds, hparam_suffix, al_rounds):
    results = {metric: [] for metric in delta_metrics}

    # # Loop through the seeds and collect delta values
    # for seed in seeds:
    #     file_path = os.path.join(directory, hparam_suffix + f"_{seed}", "deltas_last_iter.csv")
    #     if os.path.isfile(file_path):
    #         data = pd.read_csv(file_path)
    #         for metric in delta_metrics:
    #             # Append if the metric is found in the dataframe
    #             if metric in data['delta'].values:
    #                 results[metric].append((data.loc[data['delta'] == metric, 'average'].item()))
    #             else:
    #                 print(f"Metric {metric} not found in {file_path}")
    #     else:
    #         print(f"File not found: {file_path}")
    
    # # Write in file
    # with open(os.path.join(directory, "delta_avg.txt"), "w") as f:
    #     for metric in delta_metrics:
    #         values = results[metric]
    #         if values:
    #             avg = np.mean(values)
    #             var = np.var(values)
    #             f.write(f"{metric}: Average = {avg}, Variance = {var}\n")
    #             print(f"{metric}: Average = {avg}, Variance = {var}")
    #         else:
    #             f.write(f"No data collected for {metric}\n")
    #             print(f"No data collected for {metric}")
    
    last_iter_results = {}
    seed_values = {}
    # Loop through the seeds and collect delta values
    for seed in seeds:
        file_path = os.path.join(directory, hparam_suffix + f"_{seed}", "results_test.csv")
        for line in open(file_path):
            if line.startswith(f"iter_{al_rounds}"):
                parts = line.strip().split(',')
                strategy, values = parts[1], parts[2:]
                last_iter_results[strategy] = list(map(float, values))
                # convert to % if required but if not retain values
                last_iter_results[strategy] = [x * 100 if x < 1 else x for x in last_iter_results[strategy]]

        
        # get mean values of each strategy
        for strategy in last_iter_results:
            last_iter_results[strategy] = np.mean(last_iter_results[strategy])
            for metric in strategy_metrics:
                if strategy.startswith(metric):
                    if strategy not in seed_values:
                        seed_values[strategy] = {}
                    seed_values[strategy][seed] = last_iter_results[strategy]
        
    # calculate average and variance across seeds
    with open(os.path.join(directory, "last_iter_avg.txt"), "w") as f:
        for strategy in seed_values:
            values = list(seed_values[strategy].values())
            avg = np.mean(values)
            var = np.var(values)
            print(f"{strategy}: Average = {avg}, Variance = {var}")
            # write upto 2 decimal places
            f.write(f"{strategy}: Average = {avg:.1f}, Variance = {var}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate averages and variances of specified metrics across seeds.')
    parser.add_argument('--directory', type=str, help='Directory containing the seeds subdirectories')
    parser.add_argument('--delta_metrics', nargs='+', help='Metrics for which to calculate averages and variances')
    parser.add_argument('--strategy_metrics', nargs='+', help='Metrics for which to calculate averages and variances')
    parser.add_argument('--seeds', nargs='+', default=['2', '22', '42'], help='Seed values to be processed (default: 2 22 42)')
    parser.add_argument('--hparam_suffix', type=str, default='2e-05_10_seed', help='Suffix to be added to the directory name')
    parser.add_argument('--al_rounds', type=int, default=5, help='Number of active learning rounds')
    args = parser.parse_args()

    calculate_averages_and_variance(args.directory, args.delta_metrics, args.strategy_metrics, args.seeds, args.hparam_suffix, args.al_rounds)
