#!/bin/bash

export CACHE_DIR="src/cache/"

# Default values
DOCS_TIER_X="docs_tier_1"

while getopts "t:" opt; do
  case $opt in
    t)
      case $OPTARG in
        "1")
          DOCS_TIER_X="docs_tier_1"
          ;;
        "2")
          DOCS_TIER_X="docs_tier_2"
          ;;
        "3")
          DOCS_TIER_X="docs_tier_3"
          ;;
        "4")
          DOCS_TIER_X="docs_tier_4"
          ;;
        "2_minus_1")
          DOCS_TIER_X="docs_tier_2_minus_1"
          ;;
        "3_minus_1")
          DOCS_TIER_X="docs_tier_3_minus_1"
          ;;
        "4_minus_1")
          DOCS_TIER_X="docs_tier_4_minus_1"
          ;;
        *)
          echo "Invalid tier option. Usage: $0 -t [1-4]"
          exit 1
          ;;
      esac
      ;;
    *)
      echo "Usage: $0 -t [1-4]"
      exit 1
      ;;
  esac
done

echo "DOCS_TIER_X=${DOCS_TIER_X}"

python data_statistic.py \
    --docs_tier_x $DOCS_TIER_X \
    --docs_path_cache_dir $CACHE_DIR
