#!/bin/bash
# Path: scripts/train/run_ft_squad_v2.sh
# Script to run finetuning on SQuAD v2.0
# Usage: bash scripts/train/run_ft_squad_v2.sh [model] 
# For custom model, pass the path to the model as the 5th argument

set -e 

MODEL=${1:-xlm-roberta-large}
DATASET=${2:-squad_v2}
MODEL_PATH=${3:-xlm-roberta-large} # (Optional) Path to custom model
OUTPUT_BASE_PATH=${4:-${PWD}/outputs}

# Hyperparameters (change as needed)
BATCH_SIZE=32
MAX_SEQ_LENGTH=256
DOC_STRIDE=128
SEED=42
LR=1e-5
EPOCHS=2
SAVE_STEPS=10000

mkdir -p /scratch/${USER}/cache
mkdir -p ${OUTPUT_BASE_PATH}

export HF_HOME=/scratch/${USER}/cache
export WANDB_START_METHOD="thread"
export WANDB_DISABLE_SERVICE=True

# Setting model path
if [ ${MODEL} == "xlm-roberta-large" ]
then
    MODEL_NAME_OR_PATH="xlm-roberta-large"
elif [ ${MODEL} == "rembert" ]
then
    MODEL_NAME_OR_PATH=google/rembert
elif [ ${MODEL} == "infoxlm-large" ]
then
    MODEL_NAME_OR_PATH=microsoft/infoxlm-large
fi

# Check if model path was passed as an argument
if [ ! -z ${MODEL_PATH} ]
then
    MODEL_NAME_OR_PATH=${MODEL_PATH}
fi

python /home/${USER}/transformers/examples/pytorch/question-answering/run_qa.py \
  --model_name_or_path ${MODEL_PATH} \
  --dataset_name ${DATASET} \
  --version_2_with_negative \
  --do_train \
  --do_eval \
  --per_device_train_batch_size ${BATCH_SIZE} \
  --per_device_eval_batch_size ${BATCH_SIZE} \
  --learning_rate ${LR} \
  --num_train_epochs ${EPOCHS} \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --doc_stride ${DOC_STRIDE} \
  --output_dir ${OUTPUT_BASE_PATH}/models/${MODEL}_en-ft_${DATASET} \
  --save_steps ${SAVE_STEPS}


