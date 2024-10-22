## luhua/chinese_pretrain_mrc_macbert_large

python run_sentiment.py \
--do_train \
--do_eval \
--model_name_or_path hfl/chinese-macbert-large \
--max_seq_length 128 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 8 \
--lr_scheduler_type cosine_with_restarts \
--weight_decay 1e-5 \
--learning_rate 5e-5 \
--num_train_epochs 6 \
--output_dir ./ckpt_save \
--overwrite_output_dir 