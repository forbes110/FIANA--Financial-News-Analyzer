## do training: google/mt5-small
## csebuetnlp/mT5_multilingual_XLSum, 
## yihsuan/mt5_chinese_small
## IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese

CUDA_VISIBLE_DEVICES=0 accelerate launch train_summary.py \
--model_name_or_path google/mt5-small \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--learning_rate 5e-4 \
--num_beams 6 \
--weight_decay 0 \
--num_train_epochs 3 \
--lr_scheduler_type cosine_with_restarts \
--source_prefix "summarize: " \
--output_dir ./ckpt \
--with_tracking 
