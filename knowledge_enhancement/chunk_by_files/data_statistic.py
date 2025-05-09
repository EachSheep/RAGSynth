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
    args = parser.parse_args()
    

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

    # Loop through directories and files
    total_dirs = len(tier_x_files)
    for dir_index, (dir_name, file_path_list) in enumerate(tier_x_files.items(), start=1):
        logger.info(f"Processing directory {dir_index}/{total_dirs}: {dir_name}")
        # input("Press Enter to continue...")
        total_files = len(file_path_list)

        total_tokens_len = 0
        totol_files = 0
        for file_index, file_path in enumerate(file_path_list, start=1):
            # Process each file individually
            logger.info(f"Processing file {file_index}/{total_files} in directory {dir_name}, remaining files: {total_files - file_index}")
            
            file_abs_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, "docs", dir_name, file_path))
            with open(file_abs_path, "r", encoding="utf-8") as f:
                cur_file_content = f.read()
                # Encode file content to tokens
                cur_tokens = token_encoder.encode(cur_file_content)
                cur_tokens_len = len(cur_tokens)
                total_tokens_len += cur_tokens_len
                totol_files += 1
            
        print(f"Average tokens length for {dir_name}: {total_tokens_len / totol_files} = {total_tokens_len} / {totol_files}")
        input("Press Enter to continue...")


                    

                    