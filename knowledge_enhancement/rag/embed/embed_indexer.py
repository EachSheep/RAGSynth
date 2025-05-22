import os
import json
import torch
import tiktoken
import argparse
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

from peft import PeftModel, PeftConfig
from transformers import AutoModel

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")


def load_model(embed_model_path, lora_adapter_path=None, max_seq_length=32768):
    
    """Load and configure the model."""

    if "snowflake-arctic-embed-m-v1.5" in embed_model_path or \
                "MedEmbed-small-v0.1" in embed_model_path:
        model = SentenceTransformer(embed_model_path, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16})
    elif "stella_en_400M_v5" in embed_model_path or \
                "gte-multilingual-base" in embed_model_path or \
                    "snowflake-arctic-embed-m-long" in embed_model_path or \
                        "rubert-tiny-turbo" in embed_model_path:
        model = SentenceTransformer(embed_model_path, trust_remote_code=True)
    else:
        raise ValueError(f"Model {embed_model_path} not supported.")

    model.max_seq_length = max_seq_length

    # If a LoRA adapter path is provided, load and merge it using PEFT
    if lora_adapter_path:
        try:
            print(f"Loading LoRA adapter from {lora_adapter_path}")
            # Access the underlying HuggingFace model
            # Adjust the attribute access based on your SentenceTransformer version
            base_model = model._first_module().auto_model  # Modify if necessary

            # Load the LoRA configuration
            peft_config = PeftConfig.from_pretrained(lora_adapter_path)

            # Load the LoRA model
            peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

            # Merge the LoRA weights into the base model
            peft_model = peft_model.merge_and_unload()

            # Replace the base model in SentenceTransformer with the merged model
            model._first_module().auto_model = peft_model

            print(f"Successfully merged LoRA adapter from {lora_adapter_path}")
        except Exception as e:
            print(f"Failed to load and merge LoRA adapter: {e}")
            raise e

    return model

def add_eos_func(input_examples, model):
  input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
  return input_examples

def save_embeddings_to_npy(embeddings, output_path):
    """
    Save embeddings as a .npy file.
    """
    np.save(output_path, embeddings)

def process_files(contexts_to_process, output_path, model, normalize_embeddings=False, add_eos=False, batch_size=2):
    """
    Encode contents and save all embeddings into a single .npy file, preserving directory structure, with batch processing.
    """
    if os.path.exists(output_path):
        print(f"Embeddings already exist at {output_path}. Skipping...")
    else:
        
        if add_eos:
            contexts_to_process = add_eos_func(contexts_to_process, model)
        print(f"Processing {len(contexts_to_process)} contexts to generate embeddings.")
        embeddings = model.encode(contexts_to_process, batch_size=batch_size, normalize_embeddings=normalize_embeddings, show_progress_bar=True)
        
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        np.save(output_path, np.array(embeddings))
        print(f"All embeddings saved to {output_path}")

if __name__ == "__main__":
    print("-" * 50)
    parser = argparse.ArgumentParser(description="Process files to generate embeddings and save them.")

    parser.add_argument("--input_path", type=str, help="Path to the input directory containing files to process.")
    parser.add_argument("--output_path", type=str, help="Path to the output directory to save embeddings.")

    parser.add_argument("--embed_model_path", type=str, help="Path where the model exists.")
    parser.add_argument("--normalize_embeddings", action="store_true", help="Normalize embeddings.")
    parser.add_argument("--add_eos", action="store_true", help="Add EOS token to each input example.")
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")

    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Path to the LoRA adapter to be merged.")
    args = parser.parse_args()
    
    print(f"Generating embeddings for model: {args.embed_model_path}, input_path: {args.input_path}, output_path: {args.output_path}")

    input_path = os.path.join(CUSTOM_CORPUS_HOME, args.input_path)
    output_path = os.path.join(CUSTOM_CORPUS_HOME, args.output_path)

    # Load the model and prepare the file content mapping
    token_encoder = tiktoken.encoding_for_model("gpt-4o")

    with open(input_path, "r", encoding='utf-8') as f:
        content_to_process = json.load(f)
    
    contexts_to_process = []
    for cur_chunk in content_to_process:
        contexts_to_process.append(cur_chunk["origin_context"])

    # Loop through directories and files
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_path):
        print(f"Embeddings already exist at {output_path}. Skipping.....")
    else:
        model = load_model(args.embed_model_path, args.lora_adapter_path, max_seq_length=args.max_seq_length)

        process_files(
            contexts_to_process,
            output_path,
            model,
            normalize_embeddings=args.normalize_embeddings,
            add_eos=args.add_eos,
            batch_size=args.batch_size
        )
    print("-" * 50)