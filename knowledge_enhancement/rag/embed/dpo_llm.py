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
"""
# regular:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

import copy
import random
import pytz
import time
import logging
import json
import os
from datetime import datetime
import sys
from contextlib import nullcontext
from copy import deepcopy
from typing import Optional, List

import colorlog
import datasets
import torch
import transformers
from accelerate.state import PartialState
from alignment import DataArguments, H4ArgumentParser
from dataclasses import dataclass, field
from datasets import Dataset
from peft import PeftConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from huggingface_hub import list_repo_files
from huggingface_hub.utils._validators import HFValidationError
# from utils import get_datasets, is_adapter_model

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

def get_data(
    args
) -> Dataset:

    all_datasets = []
    for i, cur_train_data_path in enumerate(args.train_data_paths):
        
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
        
    ret_dict = {
        "prompt": [cur_dict['prompt'] for cur_dict in all_datasets],
        "chosen": [cur_dict['chosen'] for cur_dict in all_datasets],
        "rejected": [cur_dict['rejected'] for cur_dict in all_datasets],
        "data_gen_type": [cur_dict['data_gen_type'] for cur_dict in all_datasets],
        "top_k_value": [cur_dict['top_k_value'] for cur_dict in all_datasets],
    }
    dataset = Dataset.from_dict(ret_dict)
    return dataset


if __name__ == "__main__":
    hf_parser = H4ArgumentParser((DataArguments, ModelConfig, DPOConfig, ScriptArguments))
    data_args, model_args, training_args, args = hf_parser.parse()

    training_args.output_dir = os.path.join(CUSTOM_CORPUS_HOME, training_args.output_dir)

    if training_args.local_rank == -1 or training_args.local_rank == 0:
        beijing_tz = pytz.timezone('Asia/Shanghai')
        time_str = datetime.now(beijing_tz).strftime('%Y_%b%d_%H-%M-%S')
        time_str_path = os.path.join(training_args.output_dir, "time_str.txt")
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
    fmt_string = "%(log_color)s %(asctime)s - %(levelname)s - %(message)s"
    log_colors = {
        "DEBUG": "white",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "purple",
    }
    log_level = training_args.get_process_log_level()
    colorlog.basicConfig(
        log_colors=log_colors,
        format=fmt_string,
        handlers=[colorlog.StreamHandler(sys.stdout)],
        level=log_level,
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

    ref_model = copy.deepcopy(model)
    ref_model_kwargs = copy.deepcopy(model_kwargs)

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    logging.info(f"Loading model {model_args.model_name_or_path}")

    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"
    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side
    # tokenizer.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    # Set reasonable default for models without max length
    if tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = training_args.max_length
    assert tokenizer.chat_template is not None, "Needs chat template!"

    if "llama-3" in model_args.model_name_or_path.lower():
        # For llama3 only
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
            "<|reserved_special_token_0|>"
        )

    ################
    # Dataset
    ################
    logger.info("*** Loading datasets ***")
    # raw_datasets = get_datasets(
    #     data_args,
    #     splits=data_args.dataset_splits,
    #     configs=data_args.dataset_configs,
    #     columns_to_keep=["chosen", "rejected", "prompt"],
    # )
    # logger.info(
    #     f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    # )
    # column_names = list(raw_datasets["train"].features)

    # #####################
    # # Apply chat template
    # #####################
    # def formatting_chat_func(example, tokenizer):
    #     prompt = [{"role": "user", "content": example["prompt"]}]
    #     example["prompt"] = tokenizer.apply_chat_template(
    #         prompt, tokenize=False, add_generation_prompt=False
    #     )
    #     return {
    #         "prompt": example["prompt"],
    #         "chosen": example["chosen"],
    #         "rejected": example["rejected"],
    #     }

    # with PartialState().main_process_first():
    #     raw_datasets = raw_datasets.map(
    #         formatting_chat_func,
    #         fn_kwargs={"tokenizer": tokenizer},
    #         num_proc=data_args.preprocessing_num_workers,
    #         remove_columns=(
    #             column_names if training_args.remove_unused_columns else None
    #         ),
    #         desc="Formatting comparisons with prompt template",
    #     )

    # # Log a few random samples from the training set:
    # if PartialState().is_main_process:
    #     for index in random.sample(range(len(raw_datasets["train"])), 3):
    #         logger.info(
    #             f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}"
    #         )
    #         logger.info(
    #             f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}"
    #         )
    #         logger.info(
    #             f"Refused sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}"
    #         )

    # train_dataset = raw_datasets["train"]
    # eval_dataset = raw_datasets["test"]

    train_dataset = get_data(args)
    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= training_args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= training_args.max_length,
        num_proc=data_args.preprocessing_num_workers,
    )
    # ramdom select 30000 samples
    # train_dataset = train_dataset.shuffle(seed=training_args.seed).select(range(30000))
    len_train_dataset = len(train_dataset)
    # eval_dataset = train_dataset.train_test_split(test_size=0.01, seed=training_args.seed)
    # random select 300 samples
    # eval_dataset = eval_dataset.shuffle(seed=training_args.seed).select(range(300))
    # len_eval_dataset = len(eval_dataset)

    ################
    # Instantiate DPO trainer
    ################
    if training_args.model_init_kwargs is None:
        training_args.model_init_kwargs = model_kwargs
        training_args.ref_model_init_kwargs = ref_model_kwargs
    else:
        training_args.model_init_kwargs.update(model_kwargs)
        training_args.ref_model_init_kwargs.update(ref_model_kwargs)
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Training ***")
    checkpoint = None
    # Check for last checkpoint
    if training_args.resume_from_checkpoint is not None:
        checkpoint = (
            get_last_checkpoint(training_args.output_dir)
            if isinstance(training_args.resume_from_checkpoint, bool)
            else training_args.resume_from_checkpoint
        )
        if checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {checkpoint=}.")
        else:
            logger.error(
                f"Failed to load last checkpoint at {checkpoint=}. Start training from scratch"
            )

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    # metrics["train_samples"] = len(raw_datasets["train"])
    metrics["train_samples"] = len_train_dataset
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
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
