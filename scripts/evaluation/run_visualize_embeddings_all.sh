#!/bin/bash
# Path: scripts/evaluation/run_visualize_embeddings_all.sh
# Collect visualization plots for all embeddings

models=( "xlm-roberta-large" "infoxlm-large" "rembert" )
datasets=( "PAN-X" "udpos" "xnli" "tydiqa" ) 
configs=( "hp" "mp" "lp" "geo" "lp-pool" )
budgets=( "10000" )

AL_ROUNDS=5
EPOCHS=10
SEED=42

ALGO='pca'
UNLABELLED_SOURCE_SAMPLE_SIZE=50000

for MODEL in "${models[@]}"
do
  for DATASET in "${datasets[@]}"
  do
    for CONFIG in "${configs[@]}"
    do
      if [ ${DATASET} == "xnli" ]
      then
        if [ ${MODEL} == "rembert" ]; 
        then
          LR=8e-06
        elif [ ${MODEL} == "xlm-roberta-large" ] || [ ${MODEL} == "infoxlm-large" ]; 
        then
          LR=5e-06
        fi
      elif [ ${DATASET} == "udpos" ] || [ ${DATASET} == "PAN-X" ]
      then
        if [ ${MODEL} == "rembert" ]; 
        then
          LR=8e-06
        elif [ ${MODEL} == "xlm-roberta-large" ] || [ ${MODEL} == "infoxlm-large" ]; 
        then
          LR=2e-05
        fi
      fi

      HPARAM_SUFFIX="${LR}_${EPOCHS}_seed_${SEED}"
      EMBEDDING_PATH=./outputs/embeddings/${MODEL}/${DATASET}/${CONFIG}/${BUDGET}/${HPARAM_SUFFIX}

      python src/evaluation/visualize_embeddings.py \
      --embedding_path ${EMBEDDING_PATH} \
      --algorithm ${ALGO} \
      --unlabelled_source_sample_size ${UNLABELLED_SOURCE_SAMPLE_SIZE} \
      --al_round_to_show 1 
    done
  done
done
