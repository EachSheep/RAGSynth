# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import colorlog
import random
import sys
import os
import pytz
import time
import copy
from datetime import datetime

import datasets
import torch
import transformers
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field
from datasets import Dataset
import json

from datasets import load_dataset
from peft import PeftConfig, get_peft_model
from tqdm.auto import tqdm
from accelerate.state import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.trainer_utils import get_last_checkpoint, EvalPrediction
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    DataCollatorForCompletionOnlyLM,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map
)

from alignment import (
    H4ArgumentParser, 
    DataArguments,
)
from huggingface_hub import list_repo_files
from huggingface_hub.utils._validators import HFValidationError

# from utils import get_datasets

tqdm.pandas()

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

logger = colorlog.getLogger(__name__)

def is_adapter_model(model_name_or_path: str, revision: str = "main") -> bool:
    try:
        # Try first if model on a Hub repo
        repo_files = list_repo_files(model_name_or_path, revision=revision)
    except (HFValidationError):
    # except (HFValidationError, RepositoryNotFoundError):
        # If not, check local repo
        repo_files = os.listdir(model_name_or_path)
    return "adapter_model.safetensors" in repo_files or "adapter_model.bin" in repo_files

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    train_data_paths: List[str] = field(default_factory=list, metadata={"help": "the training data directory"})

def get_sft_data(args) -> Dataset:
    all_datasets = []
    for cur_train_data_path in args.train_data_paths:
        tmp_dir = os.path.dirname(cur_train_data_path)
        cur_checkpoint_path = os.path.join(tmp_dir, "checkpoint.json")
        if os.path.exists(cur_checkpoint_path):
            with open(cur_checkpoint_path, "r") as f:
                cur_checkpoint = json.load(f)
            cur_data = cur_checkpoint['data']
        elif os.path.exists(cur_train_data_path):
            with open(cur_train_data_path, "r") as f:
                cur_data = json.load(f)
        else:
            raise ValueError(f"Data path {cur_train_data_path} does not exist.")
        all_datasets.extend(cur_data)
    # all_datasets = all_datasets[:10] 
    
    # 转换为SFT所需的messages格式
    ret_dict = {
        "messages": [
            [
                {"role": "user", "content": entry['prompt']},
                {"role": "assistant", "content": entry['chosen']}
            ]
            for entry in all_datasets
        ]
    }
    dataset = Dataset.from_dict(ret_dict)
    
    return dataset

if __name__ == "__main__":
    parser = H4ArgumentParser((DataArguments, ModelConfig, SFTConfig, ScriptArguments))
    data_args, model_args, training_args, args = parser.parse()
    
    training_args.output_dir = os.path.join(CUSTOM_CORPUS_HOME, training_args.output_dir)
    time_str_path = os.path.join(training_args.output_dir, "time_str.txt")

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        beijing_tz = pytz.timezone('Asia/Shanghai')
        time_str = datetime.now(beijing_tz).strftime('%Y_%b%d_%H-%M-%S')
        time_str_path = os.path.join(training_args.output_dir, "time_str.txt")
        if os.path.exists(time_str_path):
            os.remove(time_str_path)
        
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(time_str_path, "w") as f:
            f.write(time_str)
    else:
        time_str_path = os.path.join(training_args.output_dir, "time_str.txt")
    if training_args.local_rank != -1 and training_args.local_rank != 0:
        while not os.path.exists(time_str_path):
            time.sleep(1)
    with open(time_str_path, "r") as f:
        time_str = f.read().strip()
    training_args.output_dir = os.path.join(training_args.output_dir, time_str)
    if training_args.local_rank == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.logging_dir = os.path.join(training_args.output_dir, "runs")
    if training_args.local_rank == 0:
        os.makedirs(training_args.logging_dir, exist_ok=True)
    if training_args.local_rank == -1 or training_args.local_rank == 0:
        config_path = os.path.join(training_args.output_dir, "train_config.json")
        all_args = {
            "script_args": vars(args),
            "data_args": vars(data_args),
            "model_args": vars(model_args),
            "training_args": vars(training_args),
        }
        def custom_serializer(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            if isinstance(obj, set):
                return list(obj)
            # Add more custom serialization logic here for other types if needed
            return str(obj)  # Fallback: convert to string
        with open(config_path, "w") as f:
            json.dump(all_args, f, indent=2, ensure_ascii=False, default=custom_serializer)
    
    for i, cur_data_path in enumerate(args.train_data_paths):
        args.train_data_paths[i] = os.path.join(CUSTOM_CORPUS_HOME, cur_data_path)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    fmt_string = '%(log_color)s %(asctime)s - %(levelname)s - %(message)s'
    log_colors = {
            'DEBUG': 'white',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'purple'
            }
    log_level = training_args.get_process_log_level()
    colorlog.basicConfig(
        log_colors=log_colors, 
        format=fmt_string, 
        handlers=[colorlog.StreamHandler(sys.stdout)],
        level = log_level
    )
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")


    ################
    # Model & Tokenizer
    ################
    # MODEL
    logger.info("*** Loading pretrained model and tokenizer ***")

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(
            model_args.model_name_or_path, revision=model_args.model_revision
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = get_peft_model(base_model, peft_config)
        model_kwargs = None
    # ref_model = copy.deepcopy(model)
    # ref_model_kwargs = copy.deepcopy(model_kwargs)

    # if model_args.use_peft is True:
    #     ref_model = None
    #     ref_model_kwargs = None

    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True, 
        trust_remote_code=model_args.trust_remote_code
    )
    # tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"
    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side
    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = training_args.max_seq_length
    assert tokenizer.chat_template is not None, "Needs chat template!"

    if "llama-2" in model_args.model_name_or_path.lower():
        collator = DataCollatorForCompletionOnlyLM(instruction_template="[INST]", response_template="[/INST]", tokenizer=tokenizer)
    elif "llama3" in model_args.model_name_or_path.lower():
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
        collator = DataCollatorForCompletionOnlyLM(instruction_template="<|start_header_id|>user<|end_header_id|>\n\n", response_template="<|start_header_id|>assistant<|end_header_id|>\n\n", tokenizer=tokenizer)
    elif "qwen2" in model_args.model_name_or_path.lower() or "qwen2.5" in model_args.model_name_or_path.lower():
        collator = DataCollatorForCompletionOnlyLM(instruction_template="<|im_start|>system", response_template="<|im_start|>assistant", tokenizer=tokenizer)
    elif "mistral" in model_args.model_name_or_path.lower():
        collator = DataCollatorForCompletionOnlyLM(instruction_template="[INST]", response_template="[/INST]", tokenizer=tokenizer)
    elif "phi-3.5" in model_args.model_name_or_path.lower():
        collator = DataCollatorForCompletionOnlyLM(instruction_template="<|user|>\n", response_template="<|assistant|>\n", tokenizer=tokenizer)
        

    ################
    # Dataset
    ################
    logger.info("*** Loading datasets ***")
    train_dataset = get_sft_data(args)
    # train_dataset = train_dataset.filter(
    #     lambda x: len(tokenizer.encode(x["messages"][0]["content"])) + len(tokenizer.encode(x["messages"][1]["content"])) <= training_args.max_seq_length,
    #     num_proc=data_args.preprocessing_num_workers,
    # )
    len_train_dataset = len(train_dataset)

    # raw_datasets = get_datasets(
    #     data_args,
    #     splits=data_args.dataset_splits,
    #     configs=data_args.dataset_configs,
    #     columns_to_keep=["messages"],
    # )
    # logger.info(
    #     f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    # )
    # column_names = list(raw_datasets["train"].features)
    

    #####################
    # Apply chat template
    #####################    
    def formatting_chat_func(examples):
        output_texts = tokenizer.apply_chat_template(
            examples["messages"], 
            tokenize=True, 
            add_generation_prompt=False, 
            truncation=True, 
            padding=False,
            max_length=training_args.max_seq_length,
            return_dict=True)
        return output_texts
    
    # Pre-process the datasets only once per node. The remaining processes will use the cache.
    with PartialState().local_main_process_first():
        train_dataset = train_dataset.map(
            formatting_chat_func, 
            batched=True, 
            remove_columns=["messages"],
            num_proc=data_args.preprocessing_num_workers,
            desc="Formatting data with prompt template",
        )

        # train_dataset = raw_datasets["train"].map(
        #     formatting_chat_func, 
        #     batched=True, 
        #     remove_columns=column_names if training_args.remove_unused_columns else None,
        #     num_proc=None,
        #     batch_size=training_args.dataset_batch_size)
        
        # eval_dataset = raw_datasets["test"].map(
        #     formatting_chat_func, 
        #     batched=True, 
        #     remove_columns=column_names if training_args.remove_unused_columns else None,
        #     num_proc=None,
        #     batch_size=training_args.dataset_batch_size)

    #########################
    # Define evaluate metric
    #########################
    entropy_list = []
    def batch_compute_metrics(eval_pred: EvalPrediction, compute_result: bool):
        global entropy_list
        IGNORE_INDEX = -100
        shift_logits = eval_pred.predictions[..., :-1, :].contiguous()
        shift_labels = eval_pred.label_ids[..., 1:].contiguous()
        batch_size, seq_length, vocab_size = shift_logits.shape
        trainer.accelerator.print(f"batch_size, seq_length: {batch_size,seq_length}")
        # Flatten the tokens
        entropy = torch.nn.functional.cross_entropy(
            shift_logits.view(batch_size * seq_length, vocab_size), 
            shift_labels.view(batch_size * seq_length), 
            reduction='none'
        )
        # Append the flattened entropy for this batch
        entropy_list.append(entropy[torch.where(shift_labels.view(batch_size * seq_length) != IGNORE_INDEX)].cpu())
        if compute_result:
             # Concatenate all entropy values and compute the mean
            all_entropy = torch.cat(entropy_list, dim=0)
            mean_entropy = torch.mean(all_entropy, dim=-1)
            trainer.accelerator.print(mean_entropy)
            perplexity = torch.exp(mean_entropy)
            # empty cache
            entropy_list = []
            return {"perplexity": perplexity.item()}
        else:
            return {}
        
    
    class PrintMetricsCallback(TrainerCallback):
        def on_evaluate(self, args:TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            # 仅在主进程打印
            if state.is_world_process_zero:
                logs = state.log_history[-1]
                if "eval_perplexity" in logs:
                    logger.critical(f"Perplexity: {logs['eval_perplexity']}")

    logger.critical(training_args.max_seq_length)

    ################
    # Initialize the Trainer
    ################
    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = model_kwargs
    else:
        training_args.model_init_kwargs.update(model_kwargs)
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        compute_metrics=batch_compute_metrics,
        callbacks=[PrintMetricsCallback],
        data_collator=collator,
    )


    ###############
    # Training loop
    ###############
    logger.info("*** Training ***")
    checkpoint = None
    # Check for last checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = get_last_checkpoint(training_args.output_dir) if isinstance(training_args.resume_from_checkpoint, bool) else training_args.resume_from_checkpoint
        if checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {checkpoint=}.")
        else:
            logger.error(f"Failed to load last checkpoint at {checkpoint=}. Start training from scratch")
    
    # dl = trainer.get_train_dataloader()
    # for idx, batch in enumerate(dl):
    #     print(batch)  # Print or inspect the batch
    #     print("-------")
    #     print(tokenizer.decode(batch['input_ids'].tolist()[0]))
    #     print("-------")
    #     print(f"{len(batch['input_ids'].tolist()[0])=}")
    #     print("============")
    #     if idx == 10:
    #         break  # Break after one batch to avoid printing too much data

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


    ##########
    # Evaluate
    ##########
    torch.cuda.empty_cache()
    if training_args.do_eval:
        logger.info("*** Evaluating ***")
        metrics = trainer.evaluate()
        # metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    logger.info("*** Evaluating complete ***")
