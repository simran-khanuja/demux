#!/bin/bash
# Path: scripts/evaluation/run_collect_results_single_model.sh
# Collect results for a single model, dataset, config and budget


MODEL_PATH=${1:-./outputs/models/xlm-roberta-large/udpos/target_lp/10000/2e-05_10_seed_42}

AL_ROUNDS=5
python src/evaluation/collect_results.py \
    --model_path ${MODEL_PATH} \
    --al_rounds ${AL_ROUNDS}
