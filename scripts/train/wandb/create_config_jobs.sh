#!/bin/bash
# Path: scripts/train/wandb/create_configs.sh
# Script to create wandb jobs from config files

set -e
# Create jobs file
JOB_DIR=scripts/train/wandb/jobs
mkdir -p ${JOB_DIR}

# models=( "xlm-roberta-large" "infoxlm-large" "rembert" )
# datasets=( "PAN-X" "udpos" "xnli" "tydiqa" ) 
# configs=( "hp" "mp" "lp" "geo" "lp-pool" )

models=( "xlm-roberta-large" )
datasets=( "xnli" ) 
configs=( "hp" )

BUDGET="10000"
SEED=42
CONFIG_BASE_PATH="scripts/train/wandb/configs"

for MODEL in "${models[@]}"; do
  for DATASET in "${datasets[@]}"; do
    # loop over configs
    for i in "${!configs[@]}"; do
        PROJECT_NAME="AL_${DATASET}_${configs[i]}_${BUDGET}_${MODEL}_${SEED}"
        CONFIG_FILE="${CONFIG_BASE_PATH}/${MODEL}/${DATASET}_${configs[i]}.yaml"
        wandb sweep --project ${PROJECT_NAME} ${CONFIG_FILE} > ${JOB_DIR}/${PROJECT_NAME}.txt 2>&1
        # Get last word of the last line
        WANDB_SWEEP=$(tail -1 ${JOB_DIR}/${PROJECT_NAME}.txt | awk '{print $NF}')
        echo ${WANDB_SWEEP}
        # Create job file
        JOB_FILE=${JOB_DIR}/${PROJECT_NAME}.sh
        # Write job file
        echo "#!/bin/bash

HOST=\`hostname\`
echo \${HOST}\

set -e

mkdir -p /scratch/\${USER}/cache/model
mkdir -p /scratch/\${USER}/cache/data

export HF_HOME=/scratch/\${USER}/cache
export WANDB_START_METHOD=\"thread\"
export WANDB_DISABLE_SERVICE=True

wandb agent ${WANDB_SWEEP}" > ${JOB_FILE}
        done
    done
done
