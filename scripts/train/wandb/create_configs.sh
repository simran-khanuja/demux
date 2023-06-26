#!/bin/bash
# Path: scripts/train/wandb/create_configs.sh
# Script to create wandb config files for all strategies

set -e
# Create jobs file
CONFIG_DIR=scripts/train/wandb/configs
mkdir -p ${CONFIG_DIR}

OUTPUT_BASE_PATH="./outputs"

# Training arguments
BUDGET="10000"
SEED=42
EPOCHS=10
PER_DEVICE_TRAIN_BATCH_SIZE=64
PER_DEVICE_EVAL_BATCH_SIZE=64
INFERENCE_BATCH_SIZE=1024
MAX_SEQ_LENGTH=128
DO_TRAIN=true
DO_PREDICT=true
MAX_TO_KEEP=1
TOTAL_ROUNDS=5
PAD_TO_MAX_LENGTH=true
GRAD_ACC_STEPS=1

# loop over models, datasets, and configs
models=( "xlm-roberta-large" "infoxlm-large" "rembert" )
datasets=( "PAN-X" "udpos" "xnli" "tydiqa" ) 
configs=( "hp" "mp" "lp" "geo" "lp-pool" )

for MODEL in "${models[@]}"; do
  for DATASET in "${datasets[@]}"; do

    echo "Creating configs for ${MODEL} on ${DATASET}"

    # EN FT Model Path
    FT_MODEL_PATH="./outputs/models/${MODEL}_en-ft_${DATASET}"

    if [ ${DATASET} = "PAN-X" ]; then
      # array of target languages
      target=( "fr" "tr" "ur" "id,my,vi" "ar,id,my,he,ja,kk,ms,ta,te,th,yo,zh,ur" )
      # array of source languages
      source=( "af,ar,bg,bn,de,el,es,et,eu,fa,fi,he,hi,hu,id,it,ja,jv,ka,kk,ko,ml,mr,ms,my,pt,ru,sw,ta,te,th,tl,tr,ur,vi,yo,zh" \
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

    elif [ ${DATASET} = "udpos" ]; then
      # array of target languages
      target=( "French" "Turkish" "Urdu" "Telugu,Marathi,Urdu" "Arabic,Hebrew,Japanese,Korean,Chinese,Persian,Tamil,Vietnamese,Urdu" )
      # array of source languages
      source=( "Afrikaans,Arabic,Basque,Bulgarian,Dutch,Estonian,Finnish,German,Greek,Hebrew,Hindi,Hungarian,Indonesian,Italian,Japanese,Korean,Chinese,Marathi,Persian,Portuguese,Russian,Spanish,Tamil,Telugu,Turkish,Vietnamese,Urdu" \
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

    elif [ ${DATASET} = "xnli" ]; then
      # array of target languages
      target=( "fr" "tr" "ur" "bg,el,tr" "ar,th,sw,ur,hi" )
      # array of source languages
      source=( "ar,bg,de,el,es,ru,th,vi,zh,tr,sw,ur,hi"  "ar,bg,de,el,es,ru,th,vi,zh,fr,sw,ur,hi" "ar,bg,de,el,es,ru,th,vi,zh,sw,fr,tr,hi" "ar,de,es,ru,th,vi,zh,sw,fr,ur,hi" "bg,de,el,es,ru,vi,zh,fr,tr" )
      # Learning rate
      if [ ${MODEL} == "rembert" ]; 
      then
          LR=8e-6
      elif [ ${MODEL} == "xlm-roberta-large" ] || [ ${MODEL} == "infoxlm-large" ];
      then
          LR=5e-6
      fi
    elif [ ${DATASET} = "tydiqa" ]; 
    then
        # array of target languages
        target=( "fi" "ar" "bn" "bn,te" "sw,bn,ko" )
        # array of source languages
        source=( "id,te,ar,ru,sw,bn,ko" "id,fi,te,ru,sw,bn,ko" "id,fi,te,ar,ru,sw,ko" "id,fi,ar,ru,sw,ko" "id,fi,te,ar,ru" )
        # Hparams for QA
        PER_DEVICE_TRAIN_BATCH_SIZE=32
        PER_DEVICE_EVAL_BATCH_SIZE=32
        INFERENCE_BATCH_SIZE=512
        MAX_SEQ_LENGTH=256
        EPOCHS=3
        GRAD_ACC_STEPS=1
        BUDGET="5000"
        LR=1e-5
        # EN FT Model Path
        FT_MODEL_PATH="./outputs/models/${MODEL}_en-ft_squad_v2"
    fi

    # model path
    
    # loop over configs
    for i in "${!configs[@]}"; do
      mkdir -p ${CONFIG_DIR}/${MODEL}
      CONFIG=${CONFIG_DIR}/${MODEL}/${DATASET}_${configs[i]}.yaml
      # substitute , with "_" in target languages
      TARGET_LANG=$(echo ${target[i]} | sed 's/,/_/g')
      echo "program: ./src/train_al.py
method: grid
parameters:
  project_name:
    value: \"AL_sweep\"
  target_config_name:
    value: \"target_${configs[i]}\"
  num_train_epochs:
    value: ${EPOCHS}
  learning_rate:
    value: ${LR}
  source_languages:
    value: \"${source[i]}\"
  target_languages:
    value: \"${target[i]}\"
  dataset_name:
    value: \"${DATASET}\"
  model_name_or_path:
    value: \"${FT_MODEL_PATH}\"
  output_dir:
    value: \"${OUTPUT_BASE_PATH}/models\"
  pred_output_dir:
    value: \"${OUTPUT_BASE_PATH}/predictions\"
  save_dataset_path:
    value: \"${OUTPUT_BASE_PATH}/selected_data\"
  save_embeddings_path:
    value: \"${OUTPUT_BASE_PATH}/embeddings\"
  seed:
    value: ${SEED}
  do_train:
    value: ${DO_TRAIN}
  do_predict:
    value: ${DO_PREDICT}
  pad_to_max_length:
    value: ${PAD_TO_MAX_LENGTH}
  per_device_train_batch_size:
    value: ${PER_DEVICE_TRAIN_BATCH_SIZE}
  per_device_eval_batch_size:
    value: ${PER_DEVICE_EVAL_BATCH_SIZE}
  gradient_accumulation_steps:
    value: ${GRAD_ACC_STEPS}
  inference_batch_size:
    value: ${INFERENCE_BATCH_SIZE}
  max_seq_length:
    value: ${MAX_SEQ_LENGTH}
  qa_uncertainty_method:
    value: \"logits\"
  report_to:
    value: \"wandb\"
  with_tracking:
    value: true
  max_to_keep:
    value: ${MAX_TO_KEEP}
  total_rounds:
    value: ${TOTAL_ROUNDS}
  save_predictions:
    value: true
  budget:
    value: \"${BUDGET}\"
  strategy:
    values: [\"random\", \"egalitarian\", \"gold_${TARGET_LANG}\", \"average_dist\", \"knn_uncertainty_k_1\", \"uncertainty\"]
  embedding_model:
    value: \"${MODEL}\"
  save_embeddings:
    value: true" > ${CONFIG}
  
      if [ ${DATASET} = "xnli" ]; then
        echo "  per_language_subset_size: 
    value: 200000" >> ${CONFIG}
      fi
    done
  done
done