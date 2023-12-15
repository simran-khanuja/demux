#!/bin/bash
# Path: scripts/train/run_find_correlation.sh
# This script is used to get correlation values between a target point's uncertainty and its neighbors' average uncertainty.

MODEL=${1:-xlm-roberta-large}
DATASET=${2:-udpos} # Pass a custom dataset name if you want to use a custom dataset
SOURCE_LANGUAGES=${3:-en}
TARGET_LANGUAGES=${4:-en} 
OUTPUT_BASE_PATH=${8:-${PWD}/outputs}

mkdir -p /scratch/${USER}/cache

export HF_HOME=/scratch/${USER}/cache
export WANDB_START_METHOD="thread"
export WANDB_DISABLE_SERVICE=True


# Inference arguments
EMBEDDING_METHOD="cls" 
INFERENCE_BATCH_SIZE=1024
MAX_SEQ_LENGTH=128
OUTPUT_DIR="${OUTPUT_BASE_PATH}/correlation"

# Check if source and target dataset paths are empty and initialize to None if yes
if [ -z ${SOURCE_DATASET_PATH} ]
then
    SOURCE_DATASET_PATH="None"
fi
if [ -z ${TARGET_DATASET_PATH} ]
then
    TARGET_DATASET_PATH="None"
fi

models=( "xlm-roberta-large" "rembert" "infoxlm-large" )
datasets=( "PAN-X" "udpos" "xnli" "tydiqa" )

for MODEL in "${models[@]}"
do
  for DATASET in "${datasets[@]}"
  do
    echo "Running for model: ${MODEL} and dataset: ${DATASET}"

    # Get model path
    MODEL_NAME_OR_PATH="${OUTPUT_BASE_PATH}/models/${MODEL}_en-ft_${DATASET}_${SEED}"

    python src/train/find_correlation.py \
      --source_languages ${SOURCE_LANGUAGES} \
      --target_languages ${TARGET_LANGUAGES} \
      --dataset_name ${DATASET} \
      --source_dataset_path ${SOURCE_DATASET_PATH} \
      --target_dataset_path ${TARGET_DATASET_PATH} \
      --model_name_or_path ${MODEL_NAME_OR_PATH} \
      --max_seq_length ${MAX_SEQ_LENGTH} \
      --pad_to_max_length \
      --embedding_method ${EMBEDDING_METHOD} \
      --inference_batch_size ${INFERENCE_BATCH_SIZE} \
      --output_dir ${OUTPUT_DIR} \
      --seed ${SEED}
  done
done

      