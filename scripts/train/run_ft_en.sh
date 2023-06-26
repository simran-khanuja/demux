#!/bin/bash
# Path: scripts/train/run_ft_en.sh
# Script to run finetuning experiments on task-specific labelled data in English
# Usage: bash scripts/train/run_ft_en.sh [model] [dataset]
# For custom model, pass the path to the model as the 5th argument

set -e 

MODEL=${1:-xlm-roberta-large}
DATASET=${2:-udpos} # Pass a custom dataset name if you want to use a custom dataset
SOURCE_LANGUAGES=${3:-en}
TARGET_LANGUAGES=${4:-en}
MODEL_PATH=${5:-xlm-roberta-large} # (Optional) Path to custom model
SOURCE_DATASET_PATH=${6} # (Optional) Path to custom source dataset (must have train and dev)
TARGET_DATASET_PATH=${7} # (Optional) Path to custom target dataset (must have target and test)
OUTPUT_BASE_PATH=${8:-${PWD}/outputs}

mkdir -p /scratch/${USER}/cache
mkdir -p ${OUTPUT_BASE_PATH}

export HF_HOME=/scratch/${USER}/cache
export WANDB_START_METHOD="thread"
export WANDB_DISABLE_SERVICE=True

# Hyperparameters (change as needed)
BATCH_SIZE=32
MAX_SEQ_LENGTH=128
SEED=42
LR=2e-5
EPOCHS=10
MAX_TO_KEEP=1 # max checkpoints to keep
GRAD_ACC_STEPS=2 # gradient accumulation steps

echo "Training ${MODEL} on ${DATASET} with source languages ${SOURCE_LANGUAGES} and target languages ${TARGET_LANGUAGES}"
PROJECT_NAME="${MODEL}_en-ft_${DATASET}_${SEED}"
WANDB_NAME="en-ft"

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

# Setting fine-tune hparams
if [ ${MODEL} == "xlm-roberta-large" ] || [ ${MODEL} == "infoxlm-large" ]
then
    if [ ${DATASET} == "xnli" ]
    then
        LR=5e-6
        EPOCHS=2
    elif [ ${DATASET} == "udpos" ] || [ ${DATASET} == "PAN-X" ]
    then
        LR=2e-5
        EPOCHS=10
    fi
elif [ ${MODEL} == "rembert" ]
then
    if [ ${DATASET} == "xnli" ]
    then
        LR=8e-6   
        EPOCHS=2
    elif [ ${DATASET} == "udpos" ] || [ ${DATASET} == "PAN-X" ]
    then
        LR=8e-6
        EPOCHS=3
    fi
fi

# Check if source and target dataset paths are empty and initialize to None if yes
if [ -z ${SOURCE_DATASET_PATH} ]
then
    SOURCE_DATASET_PATH=None
fi
if [ -z ${TARGET_DATASET_PATH} ]
then
    TARGET_DATASET_PATH=None
fi

python src/train_al.py \
--do_active_learning false \
--source_languages ${SOURCE_LANGUAGES} \
--target_languages ${TARGET_LANGUAGES} \
--dataset_name ${DATASET} \
--source_dataset_path ${SOURCE_DATASET_PATH} \
--target_dataset_path ${TARGET_DATASET_PATH} \
--save_dataset_path ${OUTPUT_BASE_PATH}/selected_data/${MODEL}_en-ft_${DATASET} \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--with_tracking true \
--report_to wandb \
--project_name ${PROJECT_NAME} \
--wandb_name ${WANDB_NAME} \
--do_train true \
--do_predict true \
--max_seq_length ${MAX_SEQ_LENGTH} \
--pad_to_max_length true \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--gradient_accumulation_steps ${GRAD_ACC_STEPS} \
--learning_rate ${LR} \
--max_to_keep ${MAX_TO_KEEP} \
--num_train_epochs ${EPOCHS} \
--output_dir  ${OUTPUT_BASE_PATH}/models/${MODEL}_en-ft_${DATASET} \
--save_predictions true \
--pred_output_dir ${OUTPUT_BASE_PATH}/models/${MODEL}_en-ft_${DATASET}/predictions \
--seed ${SEED}
