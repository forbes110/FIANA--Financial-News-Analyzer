'''
    --repetition_penalty 1.5 \
'''
CUDA_VISIBLE_DEVICES=0 accelerate launch eval_summary.py \
--model_name_or_path ./ckpt \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--no_repeat_ngram_size 2 \
--top_p 50 \
--do_sample True \
--source_prefix "summarize: " \
--output_dir ./ckpt \
--with_tracking 
