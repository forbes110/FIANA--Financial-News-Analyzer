python3 predict_summary.py \
--model_name_or_path ./ckpt \
--source_prefix "summarize: " \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 8 \
--num_beams 6 \
--test_file "${1}" \
--output_file "${2}" 