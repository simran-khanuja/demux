#!/bin/bash

# Base directory
BASE_DIR="./outputs/models"

# Models array
MODELS=( "infoxlm-large" "xlm-roberta-large" "rembert" )

DATASET="PAN-X"
HPARAM_SUFFIX="8e-06_10_seed"
BUDGET="10000"

# Datasets and their respective metrics
declare -A DELTA_METRICS
DELTA_METRICS[PAN-X]="knn_gold_delta knn_egal_delta"
DELTA_METRICS[udpos]="knn_gold_delta knn_egal_delta"
DELTA_METRICS[xnli]="avg_dist_gold_delta avg_dist_egal_delta"
DELTA_METRICS[tydiqa]="uncertainty_gold_delta uncertainty_egal_delta"

declare -A STRATEGY_METRICS
STRATEGY_METRICS[PAN-X]="knn gold egal"
STRATEGY_METRICS[udpos]="knn gold egal"
STRATEGY_METRICS[xnli]="average gold egal"
STRATEGY_METRICS[tydiqa]="uncertainty gold egal"

# Configurations array
CONFIGS=("target_geo" "target_mp" "target_hp" "target_lp" "target_lp-pool")

# Seeds array
SEEDS=( "2" "22" "42" )
AL_ROUNDS=5
budgets=( "10000" )
# budgets=( "5" "10" "20" "50" "100" "250" "500" "1000" "2500" "5000" )

for BUDGET in "${budgets[@]}"; do
    # Loop over models
    for MODEL in "${MODELS[@]}"; do
        # Get the metrics to process for this dataset
        DELTA_METRIC=(${DELTA_METRICS[$DATASET]})
        STRATEGY_METRIC=(${STRATEGY_METRICS[$DATASET]})
        # Loop over configurations
        for CONFIG in "${CONFIGS[@]}"; do
            # Build directory path
            DIR_PATH="${BASE_DIR}/${MODEL}/${DATASET}/${CONFIG}/${BUDGET}"
            # Call Python script with the directory path and metrics
            python src/evaluation/calculate_averages.py --directory "$DIR_PATH" --delta_metrics "${DELTA_METRIC[@]}" \
            --strategy_metrics "${STRATEGY_METRIC[@]}" --seeds "${SEEDS[@]}" --hparam_suffix "${HPARAM_SUFFIX}" \
            --al_rounds "${AL_ROUNDS}"
        done
    done
done
