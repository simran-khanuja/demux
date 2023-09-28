#!/bin/bash
# Path: scripts/train/wandb/create_configs.sh
# Script to create wandb jobs from config files

set -e
# Create jobs file
JOB_DIR=scripts/train/wandb/jobs/mt
mkdir -p ${JOB_DIR}

models=( "facebook/mbart-large-50-many-to-many-mmt" )
strategies=( "egalitarian" "knn_uncertainty" "average_dist" "uncertainty" )

BUDGET="100000"
SEED=42
CONFIG_BASE_PATH="scripts/train/wandb/configs"

for MODEL in "${models[@]}"; do
  for STRATEGY in "${strategies[@]}"; do
    # Create job file
    JOB_FILE=${JOB_DIR}/${STRATEGY}.sh
    # Write job file
    echo "#!/bin/bash
#SBATCH --job-name=AL_MT_${STRATEGY}
#SBATCH --output /projects/tir1/users/skhanuja/multilingual_ft/scratch-outputs/scratch_AL_MT_${STRATEGY} 
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:2
#SBATCH --exclude=tir-0-[3,11,32,36],tir-1-[13,18,7]
#SBATCH --time 3-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=skhanuja@andrew.cmu.edu

HOST=\`hostname\`
echo \${HOST}\

set -e

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate multilingual_ft

mkdir -p /scratch/\${USER}/cache/model
mkdir -p /scratch/\${USER}/cache/data

export HF_HOME=/scratch/\${USER}/cache
export WANDB_START_METHOD=\"thread\"
export WANDB_DISABLE_SERVICE=True

accelerate launch --config_file /home/skhanuja/.cache/huggingface/accelerate/default_config.yaml --multi_gpu --num_processes=2 src/train_al.py \
--do_active_learning true \
--get_all_en_configs true \
--target_languages mya_Mymr-eng_Latn \
--dataset_name opus100 \
--target_dataset_name facebook/flores \
--dataset_config_file scripts/train/dataset-configs.yaml \
--save_dataset_path /projects/tir1/users/skhanuja/multilingual_ft/selected_data \
--model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
--embedding_model mbart \
--save_embeddings true \
--save_embeddings_path /projects/tir1/users/skhanuja/multilingual_ft/embeddings \
--inference_batch_size 64 \
--do_train true \
--do_predict true \
--pad_to_max_length true \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 3 \
--max_to_keep 1 \
--output_dir /projects/tir1/users/skhanuja/multilingual_ft/models \
--save_predictions true \
--pred_output_dir /projects/tir1/users/skhanuja/multilingual_ft/predictions \
--with_tracking true \
--report_to wandb \
--project_name AL_MT \
--budget ${BUDGET} \
--total_rounds 5 \
--qa_uncertainty_method margin \
--strategy ${STRATEGY} \
--per_language_subset_size 50000 \
--cache_dir /scratch/\${USER}/cache" > ${JOB_FILE}
    done
done
