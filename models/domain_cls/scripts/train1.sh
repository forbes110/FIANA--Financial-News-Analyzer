## luhua/chinese_pretrain_mrc_macbert_large
python run_domain_cls.py \
    --do_train \
    --do_eval \
    --model_name_or_path luhua/chinese_pretrain_mrc_macbert_large \
    --max_seq_length 128 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine_with_restarts \
    --weight_decay 1e-5 \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --output_dir ./ckpt_save \
    --output_file ./result/pred_domnain_1.csv \
    --overwrite_output_dir