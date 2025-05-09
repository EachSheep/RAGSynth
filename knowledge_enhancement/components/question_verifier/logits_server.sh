#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# export LOGITS_MODEL_NAME=/users/hy/Models/Qwen2.5-14B-Instruct
export LOGITS_MODEL_NAME=/users/hy/Models/Qwen2.5-7B-Instruct
# export LOGITS_MODEL_NAME=/users/hy/Models/Qwen2.5-3B-Instruct

python logits_server.py