accelerate launch --config_file /home/${USER}/.cache/huggingface/accelerate/default_config.yaml \
--multi_gpu --num_processes=2 src/train_al.py \
--do_active_learning true \
--get_all_en_configs true \
--target_languages mya_Mymr-eng_Latn \
--dataset_name opus100 \
--target_dataset_name facebook/flores \
--dataset_config_file scripts/train/dataset-configs.yaml \
--save_dataset_path ./outputs/selected_data \
--model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
--embedding_model mbart \
--save_embeddings true \
--save_embeddings_path ./outputs/embeddings \
--inference_batch_size 16 \
--do_train true --do_predict true \
--pad_to_max_length true \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 3 \
--max_to_keep 1 \
--output_dir ./outputs/models \
--save_predictions true \
--pred_output_dir ./outputs/predictions \
--with_tracking true \
--report_to wandb \
--project_name AL_MT \
--budget 100000 \
--total_rounds 5 \
--qa_uncertainty_method margin \
--strategy knn_uncertainty \
--per_language_subset_size 50000 \ 
--cache_dir /scratch/${USER}/cache


# ,be-en,bg-en,bn-en