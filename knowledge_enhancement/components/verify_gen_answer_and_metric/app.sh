#!/bin/bash

QUERY_DOCS_TIER_X=docs_tier_1

INPUT_ROOT_DIR="knowledge_enhancement/data/chunk_results/${QUERY_DOCS_TIER_X}/docs.cyotek.com.filter/"
OUTPUT_ROOT_DIR="knowledge_enhancement/data/embed_check_res/${QUERY_DOCS_TIER_X}/docs.cyotek.com.filter/"

SAVE_INTERVAL=5
MAX_PROCESS_NUM=300
# ONLY_EVAL_AT_REPHRASED_POSES=(3)
# ONLY_EVAL_AT_REPHRASED_POSES_PART=(2)
# ONLY_EVAL_AT_REPHRASED_POSES_HYBRID=(6)

python ./app.py \
    --input_root_dir "$INPUT_ROOT_DIR" \
    --output_root_dir "$OUTPUT_ROOT_DIR" \
    --save_interval $SAVE_INTERVAL \
    --max_process_num $MAX_PROCESS_NUM

    # --only_eval_at_rephrased_poses "${ONLY_EVAL_AT_REPHRASED_POSES[@]}" \
    # --only_eval_at_rephrased_poses_part "${ONLY_EVAL_AT_REPHRASED_POSES_PART[@]}" \
    # --only_eval_at_rephrased_poses_hybrid "${ONLY_EVAL_AT_REPHRASED_POSES_HYBRID[@]}" \