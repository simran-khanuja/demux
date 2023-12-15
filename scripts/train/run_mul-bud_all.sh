#!/bin/bash

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate demux-env

echo ${HOSTNAME}
DATASET=udpos
FT_MODEL_PATH="./outputs/models/xlm-roberta-large_en-ft_udpos"
SEEDS=( 2 22 42 )

strategies=( "random" "egalitarian" "gold_ur" "average_dist" "knn_uncertainty_k_1" "uncertainty" )
# strategies=( "average_dist" )

for seed in "${SEEDS[@]}"
do
    for strategy in "${strategies[@]}"
    do
        echo "Running strategy: $strategy"
        bash scripts/train/run_ft_mul-bud.sh xlm-roberta-large $DATASET $FT_MODEL_PATH lp $strategy $seed
    done
done

