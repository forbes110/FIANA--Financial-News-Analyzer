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
from transformers.utils import get_full_repo_name, is_offline_mode, send_example_telemetry
from utils.config import parse_args_summary
from transformers.optimization import Adafactor, AdafactorSchedule
from utils.metrics import get_rouge

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

    ## Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. 
    ## The information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization_no_trainer", args)

    ## init accelerator
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    ## need to add fp16
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, mixed_precision='fp16', **accelerator_log_kwargs)
    # accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)


    ## Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    ## set logger
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


    ## load dataset online or self-defined datasets
    if args.dataset_name is not None:
        ## Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

    else:
        data_files = {}
        if args.train_file is not None:
            # _train_file = load_jsonlines_file(args.train_file)
            # save_json(args._train_file, _train_file)
            data_files["train"] = args.train_file

        if args.validation_file is not None:
            # _validation_file = load_jsonlines_file(args.validation_file)
            # save_json(args._validation_file, _validation_file)
            data_files["validation"] = args.validation_file

        ## the file type
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)


    ## Load pretrained model and tokenizer, config is the same as model by default
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

    ## seq2seq model to load pretrained weights
    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    print(model)

    ## resize the embeddings only when necessary to avoid index errors. 
    ## if need to create a model from scratch on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""

    ## Preprocessing the datasets.
    ## tokenize all the texts first.
    column_names = raw_datasets["train"].column_names


    # dataset_columns = ["article", "highlights"]
    dataset_columns = ["maintext", "title"]


    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    ## set max_target_length or just max_len of all sentences for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]

        ## input "summary" + article
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)

        ## Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        ## If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        ## padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    eval_ids = [id for id in raw_datasets["validation"]["id"]]

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]



    ## log a few random samples from the training set to check
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    ## data collator set
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        ## rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

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

    ## AdamW
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    ## Adafactor
    # optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=args.learning_rate)

    ## Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
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
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    ## We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    ## Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    ## Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    ## We need to initialize the trackers we use, and also store our configuration.
    ## trackers init automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        ## TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("summarization_no_trainer", experiment_config)

    '''
        Metrics: use get_rouge in tw_rouge is better
    '''
    metric = evaluate.load("rouge")
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    ## Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    starting_epoch = 0

    ## Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:

            ## Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  ## Sorts folders by date modified, most recent checkpoint is the last

        ## Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    '''    
        Start training
    '''
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        print(f"\n<< Epoch {epoch+1} >>")
        print("-----Train-----")
        if args.with_tracking:
            total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, total=len(train_dataloader), position=0, leave=True, ncols=100)):
            
            ## We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                ## We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            ## Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
        print('total_loss:', total_loss)
        
    ## 415-581
        '''
            Start Evaluating
        '''
        model.eval()
        print("---Validation---")

        ## for get_rouge()
        predictions, references = [], []

        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        '''
            num_beams: beam size,
            top_k: top-k alg,
            top_p: top-p alg,
            temperature:
            penalty_alpha: 
            repetition_penalty: 
            do_sample: defualt None, else greedy
        '''
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

        for step, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader), position=0, leave=True, ncols=100)):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )

                labels = batch["labels"]
                if not args.pad_to_max_length:
                    ## If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()

                labels = labels.cpu().numpy()

                if args.ignore_pad_token_for_loss:

                    ## Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

                '''
                    need to save the preds and calculate the scores, maybe do the postprocess to clean the result.
                    metrics -> for not chinese
                    save predictions, references for get_rouge -> chinese
                ''' 
                # print("postprocess_text!")
                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                # print("decoded_preds one batch\n", decoded_preds[0])
                # print("decoded_labels one batch\n", decoded_labels[0])


                # print("add prediction and references!")
                ## Chinese
                # predictions.extend(['1' for a in decoded_labels])
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)

                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )

        pr_list = {
            "preds": predictions,
            "refs": references
        }
        ## save preds of validation set to check if need more postprocess
        with open(f"./cache/preds_e{epoch+1}.csv", "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'predictions', 'references'])

            ## bug here, need to save new p
            for i, eval_id in enumerate(eval_ids):
                writer.writerow([eval_id, pr_list['preds'][i], pr_list['refs'][i]])
            print(f"prediction of validation set saved in ./cache/preds{epoch}.csv!") 


        ## rouge score
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        logger.info(result)


        ## chinese calculated by get_rouge()
        rouge_score = get_rouge(predictions, references)
        result_tw = {
            "rouge-1": round(rouge_score['rouge-1']['f'] * 100, 4),
            "rouge-2": round(rouge_score['rouge-2']['f'] * 100, 4),
            "rouge-l": round(rouge_score["rouge-l"]["f"] * 100, 4)
        }
        logger.info(result_tw)
        print(result_tw)


        if args.with_tracking:

            result["train_loss"] = total_loss.item() / len(train_dataloader)
            result["epoch"] = epoch
            result["step"] = completed_steps
            accelerator.log(result, step=completed_steps)

            ## for chinese
            result_tw["train_loss"] = total_loss.item() / len(train_dataloader)
            result_tw["epoch"] = epoch
            result_tw["step"] = completed_steps
            accelerator.log(result_tw, step=completed_steps)

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

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

            all_results = {f"eval_{k}": v for k, v in result.items()}
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump(all_results, f)

            ## for chinese
            with open(os.path.join(args.output_dir, "all_results_tw.json"), "w") as f:
                json.dump(result_tw, f)


if __name__ == "__main__":
    main()