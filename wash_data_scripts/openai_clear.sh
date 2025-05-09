#!/bin/bash

python src/openai_clear.py --folder_path docs/platform.openai.com \
    --suffix .clear \
    --base_url https://www.openai.com \
    --exclude /docs/api-reference

# 需要把/docs/api-reference全部替换为#