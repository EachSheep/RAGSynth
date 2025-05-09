import os
import json
import torch
import argparse
import tiktoken
import re
import numpy as np
import logging
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_context(context):
    clue_counter = 1
    processed_parts = []

    # Regular expressions to match Markdown tables and images
    table_block_pattern = r'((?:^\|.*\|(?:\n|$))+)'
    image_pattern = r'(!\[.*?\]\(.*?\))'
    element_pattern = re.compile(f'{table_block_pattern}|{image_pattern}', re.MULTILINE)

    last_end = 0
    for match in element_pattern.finditer(context):
        start, end = match.start(), match.end()
        
        # Process the text before the matched element
        if start > last_end:
            text_block = context[last_end:start]
            processed_text, clue_counter = process_text_block(text_block, clue_counter)
            processed_parts.append(processed_text)
        
        # Process the matched element
        if match.group(1):  # Table processing
            table_block = match.group(1)
            processed_table, clue_counter = process_table_block(table_block, clue_counter)
            processed_parts.append(processed_table)
        else:  # Image processing
            image = match.group(2)
            processed_parts.append(f'{image} [Sen {clue_counter}]')
            clue_counter += 1
        
        last_end = end

    # Process the remaining text
    if last_end < len(context):
        text_block = context[last_end:]
        processed_text, clue_counter = process_text_block(text_block, clue_counter)
        processed_parts.append(processed_text)

    return '\n'.join(processed_parts)

def process_table_block(table_block, counter):
    processed_lines = []
    for line in table_block.strip().split('\n'):
        # Process escaped pipes and split cells
        cells = re.split(r'(?<!\\)\|', line.strip())
        cells = [cell.strip() for cell in cells if cell.strip()]
        processed_cells = []
        for cell in cells:
            # Process the text content of each cell
            cell_content, counter = process_text_block(cell, counter)
            processed_cells.append(cell_content.replace('\n', ' '))  # No line breaks within cells
        processed_line = '| ' + ' | '.join(processed_cells) + ' |'
        processed_lines.append(processed_line)
    return '\n'.join(processed_lines), counter

def process_text_block(text_block, counter):
    processed_lines = []
    for line in text_block.split('\n'):
        # Split sentences using regular expression (keep ending punctuation)
        sentences = re.split(r'(?<=[.!?])(?:\s+|$)', line.strip())
        processed_sentences = []
        for sent in sentences:
            if not sent:
                continue
            # Handle potential mis-splitting of abbreviations (simplified handling here)
            if sent.rstrip().endswith('.') or sent.rstrip().endswith('!') or sent.rstrip().endswith('?'):
                processed_sentences.append(f"{sent.rstrip()[:-1]} [Sen {counter}]{sent.rstrip()[-1]}")
            else:
                processed_sentences.append(f"{sent.rstrip()} [Sen {counter}]")
            counter += 1
        processed_lines.append(' '.join(processed_sentences))
    return '\n'.join(processed_lines), counter

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process files to generate embeddings and save them.")
    parser.add_argument("--docs_tier_x", type=str, help="which dataset to use, available options: docs_tier_1, docs_tier_2, docs_tier_3, docs_tier_4")
    parser.add_argument("--docs_path_cache_dir", type=str, help="the cache directory of files to load")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory to save embeddings.")
    args = parser.parse_args()
    
    output_dir = os.path.join(CUSTOM_CORPUS_HOME, args.output_dir)

    if args.docs_tier_x in [
        "docs_tier_1",
        "docs_tier_2",
        "docs_tier_2_minus_1",
        "docs_tier_3",
        "docs_tier_3_minus_1",
        "docs_tier_4",
        "docs_tier_4_minus_1"
    ]:
        pass
    else:
        raise ValueError("Invalid dataset provided. Please provide a valid dataset from docs_tier_1, docs_tier_2, docs_tier_3, docs_tier_4")

    docs_path_cache_path = os.path.join(CUSTOM_CORPUS_HOME, args.docs_path_cache_dir, args.docs_tier_x + ".json")
    with open(docs_path_cache_path, "r", encoding='utf-8') as f:
        tier_x_files = json.load(f)

    # Load the model and prepare the file content mapping
    token_encoder = tiktoken.encoding_for_model("gpt-4o")
    chunk_size = 1200
    overlap_size = 100
    chunks2filepath = {}  # Dictionary to store the mapping from chunk_name to related file paths and offsets
    chunk_contents = []

    # Loop through directories and files
    total_dirs = len(tier_x_files)
    for dir_index, (dir_name, file_path_list) in enumerate(tier_x_files.items(), start=1):
        logger.info(f"Processing directory {dir_index}/{total_dirs}: {dir_name}")
        # input("Press Enter to continue...")
        chunk_file_paths = {}  # Store the mapping from chunk_name to its related file and offsets
        total_files = len(file_path_list)

        # if "drug-instructions-alibaba" not in dir_name:
        #     continue

        for file_index, file_path in enumerate(tqdm(file_path_list, desc="Processing files", unit="file", total=total_files), start=1, dynamic_ncols=True): 
            # Process each file individually
            # logger.info(f"Processing file {file_index}/{total_files} in directory {dir_name}, remaining files: {total_files - file_index}")
            
            file_abs_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, "docs", dir_name, file_path))
            with open(file_abs_path, "r", encoding="utf-8") as f:
                cur_file_content = f.read()
            if "drug-instructions-alibaba" in file_abs_path:
                chunk_name = f"chunk_{0}"
                rel_file_path = os.path.join(dir_name, file_path)
                safe_file_name =rel_file_path.replace(os.path.sep, "_")
                id = f"{safe_file_name}_{chunk_name}"
                chunk_start_char_len = 0
                chunk_end_char_len = len(cur_file_content)
                chunk_content = cur_file_content
                context = process_context(chunk_content)
                chunk_contents.append(
                        {
                            "id": id,
                            "file_path": rel_file_path,
                            "start_char_pos": chunk_start_char_len,
                            "end_char_pos": chunk_end_char_len,
                            "origin_context": chunk_content,
                            "context": context
                        }
                    )
            else:
                # Encode file content to tokens
                all_tokens = token_encoder.encode(cur_file_content)
                contents_to_process = []

                # Split the file content into chunks
                for i, start_token_pos in enumerate(range(0, len(all_tokens), chunk_size - overlap_size)):
                    chunk_tokens = all_tokens[start_token_pos:start_token_pos + chunk_size]
                    chunk_content = token_encoder.decode(chunk_tokens)
                    context = process_context(chunk_content)

                    # Each chunk corresponds to a unique chunk name
                    chunk_name = f"chunk_{i}"
                    
                    # Save the chunk content
                    contents_to_process.append(chunk_content)
                    
                    # Calculate offset within the file
                    chunk_start_char = token_encoder.decode(all_tokens[:start_token_pos])
                    chunk_start_char_len = len(chunk_start_char)
                    chunk_end_char_len = chunk_start_char_len + len(chunk_content)

                    rel_file_path = os.path.join(dir_name, file_path)
                    safe_file_name =rel_file_path.replace(os.path.sep, "_")
                    id = f"{safe_file_name}_{chunk_name}"

                    # Save the chunk's related file path and offset information
                    chunk_file_paths[file_path] = [chunk_start_char_len, chunk_end_char_len]
                    chunk_contents.append(
                        {
                            "id": id,
                            "file_path": rel_file_path,
                            "start_char_pos": chunk_start_char_len,
                            "end_char_pos": chunk_end_char_len,
                            "origin_context": chunk_content,
                            "context": context
                        }
                    )

                    # Store the chunk's file paths and offset information in chunks2filepath
                    chunks2filepath[id] = chunk_file_paths
                    chunk_file_paths = {}

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save chunk-to-filepath mapping for this file
        with open(os.path.join(output_dir, f"{dir_name}.chunks2filepath.json"), "w", encoding="utf-8") as f:
            json.dump(chunks2filepath, f, indent=2, ensure_ascii=False)
        
        with open(os.path.join(output_dir, f"{dir_name}.chunk_contents.json"), "w", encoding="utf-8") as f:
            json.dump(chunk_contents, f, indent=2, ensure_ascii=False)
        
        chunk_contents = []
        chunks2filepath = {}