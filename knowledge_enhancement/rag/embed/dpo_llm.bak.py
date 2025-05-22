# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# 0. imports
import os
import time
import pytz
from peft import get_peft_model
from datetime import datetime
import json
from dataclasses import dataclass, field
from typing import Optional, List
import logging
from transformers import logging as hf_logging

import torch
from accelerate import Accelerator
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed
)
from trl import DPOConfig, DPOTrainer

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    train_data_paths: List[str] = field(default_factory=list, metadata={"help": "the training data directory"})
    num_proc: int = field(default=24, metadata={"help": "the number of processes to use for data loading"})
    top_k_values: List[int] = field(default_factory=lambda: [3, 5], metadata={"help": "the top k values to consider"})
    only_gen_at_rephrased_poses: List[int] = field(default_factory=lambda: [4, 7], metadata={"help": "List of rephrased positions to evaluate."})
    run_name: Optional[str] = field(default="dpo_llama2", metadata={"help": "the run name"})

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="sft/results/final_checkpoint",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    use_lora: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use LoRA for parameter-efficient fine-tuning. "
                          "If False, performs full fine-tuning of the model."}
    )

    max_prompt_length: Optional[int] = field(default=10240, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=12288, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    report_to: Optional[str] = field(
        default="tensorboard",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )

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

def load_content(query_file_path):
    """
    Load contents from a JSON file.
    """
    with open(query_file_path, "r") as f:
        contents = json.load(f)
    return contents

if __name__ == "__main__":

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    args.output_dir = os.path.join(CUSTOM_CORPUS_HOME, args.output_dir)
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if rank == 0:
        logging.basicConfig(level=logging.WARNING)
        hf_logging.set_verbosity_warning()
        hf_logging.enable_default_handler()
        hf_logging.enable_explicit_format()
    else:
        logging.basicConfig(level=logging.ERROR)
        hf_logging.set_verbosity_error()
        hf_logging.disable_default_handler()
    
    if rank == 0:
        beijing_tz = pytz.timezone('Asia/Shanghai')
        time_str = datetime.now(beijing_tz).strftime('%Y_%b%d_%H-%M-%S')
        time_str_path = os.path.join(args.output_dir, "time_str.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(time_str_path, "w") as f:
            f.write(time_str)
    else:
        time_str_path = os.path.join(args.output_dir, "time_str.txt")
    if rank != 0:
        while not os.path.exists(time_str_path):
            time.sleep(1)
    with open(time_str_path, "r") as f:
        time_str = f.read().strip()
    args.output_dir = os.path.join(args.output_dir, time_str)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        config_path = os.path.join(args.output_dir, "train_config.json")
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    for i, cur_data_path in enumerate(args.train_data_paths):
        args.train_data_paths[i] = os.path.join(CUSTOM_CORPUS_HOME, cur_data_path)
    
    set_seed(args.seed)

    # 1. load a pretrained model
    torch_dtype = torch.float
    if args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        load_in_4bit=args.load_in_4bit,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    if args.use_lora:
        for param in model.parameters():
            param.requires_grad = False

    if args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    train_dataset = get_data(args)

    train_dataset = train_dataset.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= args.max_length
        and len(x["prompt"]) + len(x["rejected"]) <= args.max_length,
        num_proc=args.num_proc,
    )
    eval_dataset = train_dataset.train_test_split(test_size=0.1, seed=args.seed)

    # 4. initialize training arguments:
    training_args = DPOConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name=args.run_name,
        gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reentrant),
        seed=args.seed,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
    )

    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "out_proj",
                "fc_in",
                "fc_out",
                "wte",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None
    if rank == 0:
        model.print_trainable_parameters()
    
    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. train
    dpo_trainer.train()

    # 7. save
    output_dir = os.path.join(args.output_dir, "final_checkpoint")
    if args.use_lora:
        dpo_trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        dpo_trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
