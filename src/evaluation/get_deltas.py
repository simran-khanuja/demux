import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--result_base_path', type=str, required=True, help='Base path to the results directories')
parser.add_argument('--strategy_prefixes', type=str, required=True, help='Comma-separated list of prefixes of the strategies to be considered')
parser.add_argument('--budget', type=str, required=True, help='Budget to be considered')
parser.add_argument('--hparam_suffix', type=str, required=True, help='Suffix of the hyperparameter configuration to be considered')

args = parser.parse_args()

# Split the string of strategy prefixes into a list
strategy_prefixes = args.strategy_prefixes.split(',')

# List of sub-directories to process
subdirs = ["hp", "mp", "lp", "geo", "lp-pool"]

# Container to store all delta dataframes
df_all_deltas = {}
df_all_values = {}

# Process each directory
for subdir in subdirs:
    # Construct the CSV file path
    csv_file_path = os.path.join(args.result_base_path, subdir, args.budget, args.hparam_suffix, "results_test.csv")

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Identify test columns dynamically
    tests = [col for col in df.columns if col.endswith('test')]

    # Identify unique strategies
    all_strategies = df['strategy'].unique()

    for strategy_prefix in strategy_prefixes:
        # Identify the required strategies based on the given prefixes
        strategies = {
            strategy_prefix: [s for s in all_strategies if s.startswith(strategy_prefix)],
            "gold": [s for s in all_strategies if s.startswith('gold')],
            "egalitarian": [s for s in all_strategies if s.startswith('egalitarian')],
        }

        # Reshape the dataframe
        df_pivot = df.pivot(index='iteration', columns='strategy')

        # Calculate the differences and store values
        deltas = {}
        values = {}
        for test in tests:
            if strategy_prefix in strategies and len(strategies[strategy_prefix]) > 0:
                deltas[(subdir, strategy_prefix, 'gold', test)] = (df_pivot[(test, strategies[strategy_prefix][0])] - df_pivot[(test, strategies['gold'][0])]) if 'gold' in strategies and len(strategies['gold']) > 0 else None
                deltas[(subdir, strategy_prefix, 'egalitarian', test)] = (df_pivot[(test, strategies[strategy_prefix][0])] - df_pivot[(test, strategies['egalitarian'][0])]) if 'egalitarian' in strategies and len(strategies['egalitarian']) > 0 else None

                values[(subdir, strategy_prefix, test)] = df_pivot[(test, strategies[strategy_prefix][0])]
                values[(subdir, 'gold', test)] = df_pivot[(test, strategies['gold'][0])] if 'gold' in strategies and len(strategies['gold']) > 0 else None
                values[(subdir, 'egalitarian', test)] = df_pivot[(test, strategies['egalitarian'][0])] if 'egalitarian' in strategies and len(strategies['egalitarian']) > 0 else None

        # Remove None values from the dictionaries
        deltas = {key: val for key, val in deltas.items() if val is not None}
        values = {key: val for key, val in values.items() if val is not None}

        # Convert the dictionaries to dataframes
        df_deltas = pd.DataFrame(deltas)
        df_values = pd.DataFrame(values)

        # If there are multiple languages, calculate the average deltas and values, and update df_all_deltas and df_all_values with these averages
        if len(tests) > 1:
            for comp in ['gold', 'egalitarian']:
                avg_deltas = df_deltas.xs((comp), level=2, axis=1).mean(axis=1)
                df_all_deltas.update({(subdir, strategy_prefix, comp, 'average'): avg_deltas})

            for strategy in [strategy_prefix, 'gold', 'egalitarian']:
                avg_values = df_values.xs((strategy), level=1, axis=1).mean(axis=1)
                df_all_values.update({(subdir, strategy, 'average'): avg_values})
        else:
            df_all_deltas.update(deltas)
            df_all_values.update(values)

# Convert the dictionaries of all deltas and values to dataframes
df_final_deltas = pd.DataFrame(df_all_deltas)
df_final_deltas.columns = pd.MultiIndex.from_tuples(df_final_deltas.columns, names=['Config', 'Strategy', 'Delta From', 'Test'])

df_final_values = pd.DataFrame(df_all_values)
df_final_values.columns = pd.MultiIndex.from_tuples(df_final_values.columns, names=['Config', 'Strategy', 'Test'])

# Save the final dataframes to CSV files
df_final_deltas.to_csv(args.result_base_path + f'/{args.hparam_suffix}_deltas.csv')
df_final_values.to_csv(args.result_base_path + f'/{args.hparam_suffix}_values.csv')