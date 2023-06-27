#!/bin/bash
# Path: scripts/evaluation/run_collect_results_all.sh
# Collect results for all models, datasets, configs and budgets

models=( "xlm-roberta-large" "infoxlm-large" "rembert" )
datasets=( "PAN-X" "udpos" "xnli" "tydiqa" ) 
configs=( "hp" "mp" "lp" "geo" "lp-pool" )
budgets=( "10000" )

AL_ROUNDS=5
EPOCHS=10
SEED=42
STRATEGY_PREFIX="knn,uncertain,average" # this is to collect deltas of strategies from gold and egalitarian 

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
        MODEL_PATH=./outputs/models/${MODEL}/${DATASET}/target_${CONFIG}/${BUDGET}/${HPARAM_SUFFIX}

        python src/evaluation/collect_results.py \
        --model_path ${MODEL_PATH} \
        --al_rounds ${AL_ROUNDS}

        
        python src/evaluation/get_deltas.py \
        --result_base_path "./outputs/models/${MODEL}/${DATASET}" \
        --strategy_prefix "${STRATEGY_PREFIX}" \
        --budget "${BUDGET}" \
        --hparam_suffix "${HPARAM_SUFFIX}"
       
      done
    done
  done
done
