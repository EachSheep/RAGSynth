#!/bin/bash  

# Check if the required arguments are provided as command-line arguments
while getopts "t:d:p:l:e:" opt; do
  case $opt in
    t) DOCS_TIER_X="$OPTARG" ;;
    d) DATASET_NAME="$OPTARG" ;;
    p) PROPOSE_TYPE="$OPTARG" ;;
    l) LLM="$OPTARG" ;;
    e) EMBED="$OPTARG" ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      echo "Usage: $0 -t <dataset_name> -p <docs_tier> -d <propose_type> [-l <llm>] [-e <embed>]"
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      echo "Usage: $0 -t <dataset_name> -p <docs_tier> -d <propose_type> [-l <llm>] [-e <embed>]"
      exit 1
      ;;
  esac
done

# Check if mandatory arguments are provided
if [ -z "$DOCS_TIER_X" ]; then
  echo "No docs tier provided."
  echo "Usage: $0 -t <dataset_name> -p <docs_tier> -d <propose_type> [-l <llm>] [-e <embed>]"
  exit 1
fi

if [ -z "$DATASET_NAME" ]; then
  echo "No dataset name provided."
  echo "Usage: $0 -t <dataset_name> -p <docs_tier> -d <propose_type> [-l <llm>] [-e <embed>]"
  exit 1
fi

if [ -z "$PROPOSE_TYPE" ]; then
  echo "No propose_type provided."
  echo "Usage: $0 -t <dataset_name> -p <docs_tier> -d <propose_type> [-l <llm>] [-e <embed>]"
  exit 1
fi

PREPROCESS_ENV_PATH="admission.stanford.edu.filter/admission.stanford.edu.filter.preprocess.env"

# Validate propose_type and set postprocess_env_path
case "$PROPOSE_TYPE" in  
  content)  
    echo "Valid propose_type: $PROPOSE_TYPE"
    POSTPROCESS_ENV_PATH="admission.stanford.edu.filter/admission.stanford.edu.filter.postprocess.content.env"
    
    # Execute the python script with the preprocess environment path for content
    python query_generation_test.py \
        --preprocess_env_path $PREPROCESS_ENV_PATH \
        ${LLM:+--llm $LLM} \
        ${EMBED:+--embed $EMBED} \
        --dataset_name $DATASET_NAME \
        --tier $DOCS_TIER_X
    ;;  
  entity_graph)
    echo "Valid propose_type: $PROPOSE_TYPE"
    POSTPROCESS_ENV_PATH="admission.stanford.edu.filter/admission.stanford.edu.filter.postprocess.entitygraph.env"
    ;;
  *)  
    echo "Invalid propose_type: $PROPOSE_TYPE"
    echo "Usage: $0 -t <dataset_name> -p <docs_tier> -d <propose_type> [-l <llm>] [-e <embed>]"
    echo "Where propose_type should be one of:"
    echo "  content"
    echo "  entity_graph"
    exit 1  
    ;;
esac  

# Execute the python script with the postprocess environment path
python query_generation_test.py \
    --postprocess_env_path $POSTPROCESS_ENV_PATH \
    ${LLM:+--llm $LLM} \
    ${EMBED:+--embed $EMBED} \
    --dataset_name $DATASET_NAME \
    --tier $DOCS_TIER_X

