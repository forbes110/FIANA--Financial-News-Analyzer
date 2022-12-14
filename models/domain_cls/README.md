## download
both task need to download first
```
bash download_bonus.sh
```

# Intent Classification

## train
```
python3.9 run_intent_cls.py \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./ckpt_save/intent
```




# Slot Tagging
## train
```
python3.9 run_slot_tag.py \
  --model_name_or_path Jean-Baptiste/camembert-ner \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 4 \
  --overwrite_output_dir \
  --output_dir ./ckpt_save/slot \
  --ignore_mismatched_sizes True
```
