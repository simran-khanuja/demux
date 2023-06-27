#!/bin/bash
# Path: scripts/train/run_ft_al.sh
# Script to run finetuning experiments on DeMuX for a given strategy 
# Usage: bash scripts/train/run_ft_al.sh [model] [dataset]
# For custom model, pass the path to the model as the 5th argument

MODEL=${1:-xlm-roberta-large}
DATASET=${2:-udpos} # Pass a custom dataset name if you want to use a custom dataset
FT_MODEL_PATH=${3:-outputs/models/xlm-roberta-large_en-ft_udpos}
CONFIG=${4:-lp}
STRATEGY=${5:-knn_uncertainty}
SOURCE_DATASET_PATH=${6} # (Optional) Path to custom source dataset (must have train and dev)
TARGET_DATASET_PATH=${7} # (Optional) Path to custom target dataset (must have target and test)
OUTPUT_BASE_PATH=${8:-${PWD}/outputs}

mkdir -p /scratch/${USER}/cache

export HF_HOME=/scratch/${USER}/cache
export WANDB_START_METHOD="thread"
export WANDB_DISABLE_SERVICE=True

# Training arguments
BUDGET="10000"
SEED=42
EPOCHS=10
PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=64
INFERENCE_BATCH_SIZE=1024
LR=2e-5
MAX_SEQ_LENGTH=128
MAX_TO_KEEP=1
TOTAL_ROUNDS=5
PAD_TO_MAX_LENGTH=true
GRAD_ACC_STEPS=1
DATASET_CONFIG_FILE="scripts/train/dataset-configs.yaml"

CONFIGS=( "hp" "mp" "lp" "geo" "lpp" )

if [ ${DATASET} == "PAN-X" ]; 
then
    # array of target languages arranged in the config order
    TARGET=( "fr" "tr" "ur" "id,my,vi" "ar,id,my,he,ja,kk,ms,ta,te,th,yo,zh,ur" )
    # array of source languages arranged in the config order
    SOURCE=( "af,ar,bg,bn,de,el,es,et,eu,fa,fi,he,hi,hu,id,it,ja,jv,ka,kk,ko,ml,mr,ms,my,pt,ru,sw,ta,te,th,tl,tr,ur,vi,yo,zh" \
    "af,ar,bg,bn,de,el,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,jv,ka,kk,ko,ml,mr,ms,my,pt,ru,sw,ta,te,th,tl,ur,vi,yo,zh" \
    "af,ar,bg,bn,de,el,es,et,eu,fa,fi,fr,he,hi,hu,id,it,ja,jv,ka,kk,ko,ml,mr,ms,my,pt,ru,sw,ta,te,th,tl,tr,vi,yo,zh" \
    "af,ar,bg,bn,de,el,es,et,eu,fa,fi,fr,he,hi,hu,it,ja,jv,ka,kk,ko,ml,mr,ms,pt,ru,sw,ta,te,th,tl,tr,ur,yo,zh" \
    "af,bg,bn,de,el,es,et,eu,fa,fi,fr,hi,hu,it,jv,ka,ko,ml,mr,pt,ru,sw,tl,tr,vi" )
    # Learning rate
    if [ ${MODEL} == "rembert" ]; 
    then
        LR=8e-6
    elif [ ${MODEL} == "xlm-roberta-large" ] || [ ${MODEL} == "infoxlm-large" ]; 
    then
        LR=2e-5
    fi

elif [ ${DATASET} == "udpos" ]; 
then
    # array of target languages arranged in the config order
    TARGET=( "French" "Turkish" "Urdu" "Telugu,Marathi,Urdu" "Arabic,Hebrew,Japanese,Korean,Chinese,Persian,Tamil,Vietnamese,Urdu" )
    # array of source languages arranged in the config order
    SOURCE=( "Afrikaans,Arabic,Basque,Bulgarian,Dutch,Estonian,Finnish,German,Greek,Hebrew,Hindi,Hungarian,Indonesian,Italian,Japanese,Korean,Chinese,Marathi,Persian,Portuguese,Russian,Spanish,Tamil,Telugu,Turkish,Vietnamese,Urdu" \
    "Afrikaans,Arabic,Basque,Bulgarian,Dutch,Estonian,Finnish,French,German,Greek,Hebrew,Hindi,Hungarian,Indonesian,Italian,Japanese,Korean,Chinese,Marathi,Persian,Portuguese,Russian,Spanish,Tamil,Telugu,Vietnamese,Urdu" \
    "Afrikaans,Arabic,Basque,Bulgarian,Dutch,Estonian,Finnish,French,German,Greek,Hebrew,Hindi,Hungarian,Indonesian,Italian,Japanese,Korean,Chinese,Marathi,Persian,Portuguese,Russian,Spanish,Tamil,Telugu,Turkish,Vietnamese" \
    "Afrikaans,Arabic,Basque,Bulgarian,Dutch,Estonian,Finnish,French,German,Greek,Hebrew,Hindi,Hungarian,Indonesian,Italian,Japanese,Korean,Chinese,Persian,Portuguese,Russian,Spanish,Tamil,Turkish,Vietnamese" \
    "Afrikaans,Basque,Bulgarian,Dutch,Estonian,Finnish,French,German,Greek,Hindi,Hungarian,Indonesian,Italian,Marathi,Portuguese,Russian,Spanish,Telugu,Turkish" )
    # Learning rate
    if [ ${MODEL} == "rembert" ]; 
    then
        LR=8e-6
    elif [ ${MODEL} == "xlm-roberta-large" ] || [ ${MODEL} == "infoxlm-large" ]; 
    then
        LR=2e-5
    fi

elif [ ${DATASET} == "xnli" ]; 
then
    # array of target languages arranged in the config order
    TARGET=( "fr" "tr" "ur" "bg,el,tr" "ar,th,sw,ur,hi" )
    # array of source languages arranged in the config order
    SOURCE=( "ar,bg,de,el,es,ru,th,vi,zh,tr,sw,ur,hi"  "ar,bg,de,el,es,ru,th,vi,zh,fr,sw,ur,hi" "ar,bg,de,el,es,ru,th,vi,zh,sw,fr,tr,hi" "ar,de,es,ru,th,vi,zh,sw,fr,ur,hi" "bg,de,el,es,ru,vi,zh,fr,tr" )
    # Learning rate
    if [ ${MODEL} == "rembert" ]; 
    then
        LR=8e-6
    elif [ ${MODEL} == "xlm-roberta-large" ] || [ ${MODEL} == "infoxlm-large" ]; 
    then
        LR=5e-6
    fi

elif [ ${DATASET} == "tydiqa" ]; 
then
    # array of target languages arranged in the config order
    TARGET=( "fi" "ar" "bn" "bn,te" "sw,bn,ko" )
    # array of source languages arranged in the config order
    SOURCE=( "id,te,ar,ru,sw,bn,ko" "id,fi,te,ru,sw,bn,ko" "id,fi,te,ar,ru,sw,ko" "id,fi,ar,ru,sw,ko" "id,fi,te,ar,ru" )
    # Hparams for QA
    MAX_SEQ_LENGTH=256
    PER_DEVICE_TRAIN_BATCH_SIZE=16
    PER_DEVICE_EVAL_BATCH_SIZE=16
    INFERENCE_BATCH_SIZE=256
    EPOCHS=3
    GRAD_ACC_STEPS=2
    BUDGET="5000"
    LR=1e-5
fi

echo "Training ${MODEL} on ${DATASET} with ${CONFIG} config and ${STRATEGY} strategy"
PROJECT_NAME="${MODEL}_ft_${DATASET}_${CONFIG}_${STRATEGY}_budget${BUDGET}"

# Get index of config
INDEX=-1
for i in "${!CONFIGS[@]}"; do
    if [[ "${CONFIGS[$i]}" == "${CONFIG}" ]]; then
        INDEX=$i
        break
    fi
done

# Get source and target languages based on index
SOURCE_LANGUAGES=${SOURCE[$INDEX]}
TARGET_LANGUAGES=${TARGET[$INDEX]}
echo "Source languages: ${SOURCE_LANGUAGES}"
echo "Target languages: ${TARGET_LANGUAGES}"

# Check if source and target dataset paths are empty and initialize to None if yes
if [ -z ${SOURCE_DATASET_PATH} ]
then
    SOURCE_DATASET_PATH="None"
fi
if [ -z ${TARGET_DATASET_PATH} ]
then
    TARGET_DATASET_PATH="None"
fi

# Check if custom ft_model_path is passed, else initialize to en-ft model path
if [ -z ${FT_MODEL_PATH} ]
then
    FT_MODEL_PATH=${OUTPUT_BASE_PATH}/models/${MODEL}_en-ft_${DATASET}
fi
python src/train_al.py \
--do_active_learning true \
--source_languages ${SOURCE_LANGUAGES} \
--target_languages ${TARGET_LANGUAGES} \
--dataset_name ${DATASET} \
--dataset_config_file ${DATASET_CONFIG_FILE} \
--source_dataset_path ${SOURCE_DATASET_PATH} \
--target_dataset_path ${TARGET_DATASET_PATH} \
--save_dataset_path ${OUTPUT_BASE_PATH}/selected_data \
--model_name_or_path ${FT_MODEL_PATH} \
--embedding_model ${MODEL} \
--save_embeddings true \
--save_embeddings_path ${OUTPUT_BASE_PATH}/embeddings \
--inference_batch_size ${INFERENCE_BATCH_SIZE} \
--with_tracking true \
--report_to wandb \
--project_name ${PROJECT_NAME} \
--wandb_name ${STRATEGY} \
--target_config_name target_${CONFIG} \
--do_train true \
--do_predict true \
--max_seq_length ${MAX_SEQ_LENGTH} \
--pad_to_max_length true \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
--gradient_accumulation_steps ${GRAD_ACC_STEPS} \
--learning_rate ${LR} \
--max_to_keep ${MAX_TO_KEEP} \
--num_train_epochs ${EPOCHS} \
--output_dir  ${OUTPUT_BASE_PATH}/models \
--save_predictions true \
--pred_output_dir ${OUTPUT_BASE_PATH}/predictions \
--seed ${SEED} \
--budget ${BUDGET} \
--total_rounds ${TOTAL_ROUNDS} \
--strategy ${STRATEGY}