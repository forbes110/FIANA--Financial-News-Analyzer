## inference
python predict_summary.py \
--model_name_or_path ./qa_ckpt \
--source_prefix "summarize: " \
--test_file "${1}" \
--output_file "${2}" 
