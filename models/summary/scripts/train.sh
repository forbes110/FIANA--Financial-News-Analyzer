## do training: google/mt5-small
## csebuetnlp/mT5_multilingual_XLSum, 
## yihsuan/mt5_chinese_small
## IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese

# CUDA_VISIBLE_DEVICES=0 accelerate launch train_summary.py \
# --model_name_or_path IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese \
# --per_device_train_batch_size 2 \
# --per_device_eval_batch_size 2 \
# --gradient_accumulation_steps 8 \
# --learning_rate 5e-4 \
# --num_beams 6 \
# --weight_decay 0 \
# --num_train_epochs 3 \
# --lr_scheduler_type cosine_with_restarts \
# --source_prefix "summarize: " \
# --output_dir ./ckpt \
# --with_tracking 



CUDA_VISIBLE_DEVICES=0 accelerate launch train_summary.py \
--model_name_or_path yihsuan/mt5_chinese_small \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--learning_rate 5e-4 \
--train_file ./data/train_file.json \
--no_repeat_ngram_size 2 \
--num_beams 6 \
--weight_decay 5e-4 \
--num_train_epochs 3 \
--lr_scheduler_type cosine_with_restarts \
--source_prefix "summarize: " \
--output_dir ./ckpt \
--with_tracking 
