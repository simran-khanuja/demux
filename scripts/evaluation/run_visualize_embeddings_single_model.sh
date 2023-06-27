#!/bin/bash
# Path: scripts/evaluation/run_visualize_embeddings_single_model.sh
# Collect visualization plots for embeddings

EMBEDDING_PATH=${1:-./outputs/embeddings/xlm-roberta-large/udpos/target_lp/10000/2e-05_10_seed_42}

ALGO='pca'
UNLABELLED_SOURCE_SAMPLE_SIZE=50000

python src/evaluation/visualize_embeddings.py \
    --embedding_path ${EMBEDDING_PATH} \
    --algorithm ${ALGO} \
    --al_round_to_show 1 \
    --unlabelled_source_sample_size ${UNLABELLED_SOURCE_SAMPLE_SIZE}
