#!/bin/bash
#SBATCH --job-name=AL_MT_uncertainty
#SBATCH --output ./scratch-outputs/scratch_AL_MT_uncertainty 
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time 3-00:00:00
#SBATCH --mail-type=END

HOST=`hostname`
echo ${HOST}
set -e

# activate conda
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate demux-env

mkdir -p /scratch/${USER}/cache/model
mkdir -p /scratch/${USER}/cache/data

export HF_HOME=/scratch/${USER}/cache
export WANDB_START_METHOD="thread"
export WANDB_DISABLE_SERVICE=True

python src/train_al.py \
--do_active_learning true \
--source_languages mya_Mymr-eng_Latn \
--target_languages mya_Mymr-eng_Latn \
--dataset_name allenai/nllb \
--target_dataset_name custom-burmese-social \
--target_dataset_path target-burmese \
--dataset_config_file scripts/train/dataset-configs.yaml \
--save_dataset_path ./outputs/selected_data \
--model_name_or_path facebook/nllb-200-distilled-600M \
--embedding_model nllb \
--save_embeddings true \
--save_embeddings_path ./outputs/embeddings \
--inference_batch_size 32 \
--do_train true \
--do_predict true \
--pad_to_max_length true \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 5e-6 \
--num_train_epochs 5 \
--max_to_keep 1 \
--output_dir ./outputs/models \
--save_predictions true \
--pred_output_dir ./outputs/predictions \
--with_tracking true \
--report_to wandb \
--project_name AL_MT \
--budget 2000 \
--total_rounds 5 \
--qa_uncertainty_method margin \
--strategy uncertainty \
--cache_dir /scratch/${USER}/cache \
--checkpointing_steps 10000 \
--silent true
