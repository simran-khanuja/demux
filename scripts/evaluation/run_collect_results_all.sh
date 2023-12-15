#!/bin/bash
# Path: scripts/evaluation/run_collect_results_all.sh
# Collect results for all models, datasets, configs and budgets
# "udpos" "xnli" "tydiqa"
models=( "infoxlm-large" "xlm-roberta-large" "rembert" )
datasets=( "PAN-X"  ) 
configs=("hp" "mp" "lp" "geo" "lp-pool")
# budgets=( "5" "10" "20" "50" "100" "250" "500" "1000" "2500" "5000" )
budgets=( "10000" )
seeds=( 2 22 42 )

AL_ROUNDS=1
EPOCHS=10
STRATEGY_PREFIX="knn,uncertain,average" # this is to collect deltas of strategies from gold and egalitarian 
RESULT_BASE_PATH="./outputs"

for SEED in "${seeds[@]}"
do
  for MODEL in "${models[@]}"
  do
    for DATASET in "${datasets[@]}"
    do
      for CONFIG in "${configs[@]}"
      do
        for BUDGET in "${budgets[@]}"
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
          elif [ ${DATASET} == "tydiqa" ]
          then
            LR=1e-05
          fi
          HPARAM_SUFFIX="${LR}_${EPOCHS}_seed_${SEED}"
          MODEL_PATH=${RESULT_BASE_PATH}/models/${MODEL}/${DATASET}/target_${CONFIG}/${BUDGET}/${HPARAM_SUFFIX}

          python src/evaluation/collect_results.py \
          --model_path ${MODEL_PATH} \
          --al_rounds ${AL_ROUNDS}

          if [ $? -ne 0 ]; then
            echo "Error occurred with item: ${MODEL_PATH}"
            # Handle error here (e.g., logging, continue, exit, etc.)
            continue  # Continue with the next item
            # exit 1    # Exit the script entirely
          fi
        
        done
      done
    done
  done
done