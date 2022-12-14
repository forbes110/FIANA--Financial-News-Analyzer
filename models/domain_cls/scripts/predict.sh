python run_domain_cls.py \
--do_predict \
--model_name_or_path ./ckpt_save \
--tokenizer_name TsinghuaAI/CPM-Generate \
--max_seq_length 128 \
--per_device_test_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 10 \
--test_file "${1}" \
--output_file "${2}" 