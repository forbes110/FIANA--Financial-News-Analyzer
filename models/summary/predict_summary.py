import json, csv
import logging
import math
import os
import random
from pathlib import Path

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from filelock import FileLock
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers.utils import check_min_version, get_full_repo_name, is_offline_mode
from utils.config import parse_args_summary, summarization_name_mapping


logger = get_logger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def main():
    args = parse_args_summary()

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)


    ## Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    ## If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    ## Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()


    if args.dataset_name is not None:
        ## Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        ''' need to be modified '''
        data_files = {}
    
        if args.test_file is not None:
            data_files["test"] = args.test_file
        
        ## the file type
        extension = args.test_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    ## Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    ## Preprocessing the datasets.
    ## First we tokenize all the texts.
    column_names = raw_datasets["test"].column_names

    ## Get the column names for input/target.
    dataset_columns = ["maintext"]

    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )

    ## Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    test_dataset = processed_datasets["test"]
    ## for the preds show
    test_ids = [id for id in raw_datasets["test"]["id"]]

    ## Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {test_dataset[index]}.")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]

        ## rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

        return preds


    ''' need to be modified '''
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    ## Optimizer
    ## Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)

    ## Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(test_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    ## Prepare everything with our `accelerator`.
    model, optimizer, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, test_dataloader, lr_scheduler
    )

    ## We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(test_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    ## Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    ## Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    ## We need to initialize the trackers we use, and also store our configuration.
    ## The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        ## TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization_no_trainer", experiment_config)

    '''
        Start Predicting
    '''
    model.eval()
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length if args is not None else config.max_length,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample, 
        "top_k": args.top_k, 
        "top_p": args.top_p,
        "typical_p": args.typical_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    preds = []
    for epoch, batch in enumerate(tqdm(test_dataloader, total=len(test_dataloader), position=0, leave=True, ncols=100)):
        with torch.no_grad():
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            # generated_tokens = accelerator.gather_for_metrics(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            decoded_preds = postprocess_text(decoded_preds)
            preds.extend(decoded_preds)

    with open(args.output_file, "w", encoding='utf-8') as fp:
        for i, test_id in enumerate(test_ids):
            json.dump({"title":preds[i],"id":test_id}, fp)
            fp.write('\n')
        print(f"prediction file saved at {str(Path(args.output_file).resolve())}")


    if args.checkpointing_steps == "epoch":
        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)



if __name__ == "__main__":
    main()