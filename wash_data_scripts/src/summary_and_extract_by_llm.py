import os
import tiktoken
import argparse
import json
import logging
import re
import time
import shutil
from openai import OpenAI
from multiprocessing import Pool, Manager
from dotenv import load_dotenv
load_dotenv()

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

qwen_base_url = os.environ["QWEN_BASE_URL"]
qwen_api_key = os.environ["QWEN_API_KEY"]
model_name = "qwen2a5-72b-instruct"
client = OpenAI(base_url=f"{qwen_base_url}", api_key=qwen_api_key)

# api_key = loaded_config.get("OPENAI_API_KEY")
# model_name = "gpt-4o-mini"
# client = OpenAI(api_key=api_key)

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")
TOKEN_LIMIT = 8000  # Token limit for summaries and FAQ handling

# Function to extract the largest {} block from the text
def extract_largest_json_block(text):
    stack = []
    start = -1
    end = -1

    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                start = i  # Mark the start of the JSON block
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:  # If stack is empty, we found the matching closing }
                    end = i
                    break

    if start != -1 and end != -1:
        return text[start:end + 1]
    return None

def process_content(content, file_name):
    tokens = tokenizer.encode(content, disallowed_special=())
    truncated_content = content
    
    # If the content exceeds 8k tokens, truncate it
    if len(tokens) > TOKEN_LIMIT:
        truncated_tokens = tokens[:TOKEN_LIMIT]
        truncated_content = tokenizer.decode(truncated_tokens)

    system_prompt = """You are an assistant tasked with both summarizing long documents, extracting FAQs (if available), and tell me if this document is useful.  
First, provide a short summary of the document about what this document is about in Markdown format.  
If the document contains FAQ sections, rewrite them in clean Markdown format, preserving all questions and answers.  
Return the result as a JSON object with 'summary', 'faq', and 'valid_document' keys.  
If no FAQ section exists, return an empty string for the 'faq' key.

The result must be a JSON object with the following keys:

- `summary`: Contains the document's summary written in Markdown format. 
- `faq`: If there is an FAQ section in the document, this field should contain a list of dictionaries, where each dictionary contains the keys:
  - `question`: The question from the FAQ.
  - `answer`: The corresponding answer.  
If there is no FAQ section, return an empty list for this field.
- The `valid_document` field should be `true` if the document contains useful information. 
If the document contains only license, conversion artifacts, metadata, logos, login pages, navigation menus, footers, social media buttons, copyright notices, terms and conditions, incomplete sections, or other pages lacking useful content, set this to `false`.

Ensure the `summary` field contains a concise yet informative description of the document's main content. If applicable, the `faq` section should list each question and its corresponding answer in a JSON object format, where each entry is a dictionary.

```json
{
  "summary": "This document provides an overview of the company's annual performance, highlighting key achievements, financial data, and strategic goals for the upcoming year. It also outlines challenges faced in various sectors and discusses plans for future growth.",
  "faq": [
    {
      "question": "What is the company’s revenue for the year?",
      "answer": "The company reported a revenue of $5 billion for the fiscal year."
    },
    
    ...

    {
      "question": "What are the main challenges faced?",
      "answer": "The main challenges include supply chain disruptions and increased competition in the market."
    }
  ],
  "valid_document": true
}
```
"""
    
    user_prompt = f"""
Here is the content of the document:
{truncated_content}

DON't OUTPUT ANY CONTENT EXCEPT FOR THE ANSWER IN THE JSON FORMAT.
Note: I'm not asking you to generate FAQs, but to determine whether the document I give you contains an FAQ section. If it doesn't, you should return an empty list for this field.
"""
    
    try_times = 6
    cur_try_time = 0

    while cur_try_time < try_times:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            # Extract the response content
            result = response.choices[0].message.content

            # Extract the largest JSON block from the result
            largest_json_block = extract_largest_json_block(result)
            
            # Proceed if we found a JSON block
            if largest_json_block:
                try:
                    # Try to parse the extracted JSON block
                    parsed_result = json.loads(largest_json_block)
                    return parsed_result  # Return if parsing succeeds
                except json.JSONDecodeError as e:
                    logging.error(f"JSONDecodeError: Could not parse the extracted JSON block: {e}")
            return parsed_result  # Return if parsing succeeds
        
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON for file {file_name}, retrying...")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            cur_try_time += 1
            time.sleep(1)  # Wait before retrying
            
            if cur_try_time >= try_times:
                logging.error(f"Failed to process content after {try_times} JSONDecodeError retries")
                return None  # Return None if all retries failed due to JSONDecodeError
            
        except Exception as e:
            logging.error(f"Error during OpenAI API call: {e}")

            if "Internal Server Error" in str(e):
                logging.error(f"Internal Server Error for file {file_name}, skipping file.")
                return None  # Skip file if Internal Server Error occurs

            if any(keyword in str(e) for keyword in [
                "context_length_exceeded", 
                "Range of input length should be", 
                "Input validation error", 
                "repetitive patterns in your prompt", 
                "encode character", 
                "Expected a string with maximum length"
            ]):
                break  # Don't retry if it's an input-related issue
            
            cur_try_time += 1
            time.sleep(1)  # Wait before retrying

            if cur_try_time >= try_times:
                logging.error(f"Failed to process content after {try_times} attempts")
                return None  # Return None if all retries failed

    return None

def process_md_file(file_path, source_folder, target_folder):
    rel_dir = os.path.relpath(os.path.dirname(file_path), source_folder)
    output_dir = os.path.join(target_folder, rel_dir)
    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    summary_file_path = os.path.join(output_dir, f"{file_name}.summary.json")

    if os.path.exists(summary_file_path):
        logging.info(f"Summary file already exists: {summary_file_path}, skipping processing for {file_name}...")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    logging.info(f"Processing file: {file_name}")
    result = process_content(content, file_name)
    if result == None:
        logging.error(f"Failed to process file: {file_name}")
        return
    
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved summary for file: {file_name}")

def copy_non_md_file(file_path, source_folder, target_folder):
    rel_dir = os.path.relpath(os.path.dirname(file_path), source_folder)
    output_dir = os.path.join(target_folder, rel_dir)
    os.makedirs(output_dir, exist_ok=True)

    target_file_path = os.path.join(output_dir, os.path.basename(file_path))
    if not os.path.exists(target_file_path):
        shutil.copy(file_path, target_file_path)
        logging.info(f"Copied non-md file: {os.path.basename(file_path)}")
    else:
        logging.info(f"File already exists: {target_file_path}, skipping copy...")

# # Recursively traverse the folder and process .md files
# def process_files(source_folder, target_folder):
#     md_files = []  # To store only markdown files
#     non_md_files = []  # To store non-markdown files

#     # Traverse directories
#     for root, dirs, files in os.walk(source_folder):
#         for file in files:
#             file_path = os.path.join(root, file)
#             if file.endswith(".md") and "valid_document_index" not in file:
#                 md_files.append(file_path)  # Collect .md files
#             else:
#                 non_md_files.append(file_path)  # Collect non .md files

#     # Log the number of files
#     total_md_files = len(md_files)
#     total_non_md_files = len(non_md_files)
#     logging.info(f"Total .md files found: {total_md_files}")
#     logging.info(f"Total non-.md files found: {total_non_md_files}")

#     # Process .md files
#     for idx, file_path in enumerate(md_files, start=1):
#         # Extract relative paths and set output directory
#         rel_dir = os.path.relpath(os.path.dirname(file_path), source_folder)
#         output_dir = os.path.join(target_folder, rel_dir)
#         os.makedirs(output_dir, exist_ok=True)

#         # Extract file name without extension
#         file_name = os.path.splitext(os.path.basename(file_path))[0]
        
#         # Define summary file path
#         summary_file_path = os.path.join(output_dir, f"{file_name}.summary.json")
        
#         # Check if the summary file already exists
#         if os.path.exists(summary_file_path):
#             logging.info(f"Summary file already exists: {summary_file_path}, skipping processing for {file_name}...")
#             continue  # Skip this file if the summary already exists
        
#         # Read the file content
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         # Log the progress
#         logging.info(f"Processing file {idx}/{total_md_files}: {file_name}")
        
#         # Process content to get both summary and FAQ (if available)
#         result = process_content(content, file_name)
#         if result == None:
#             logging.error(f"Failed to process file: {file_name}")
#             continue
        
#         # Save summary if processing succeeded
#         with open(summary_file_path, 'w', encoding='utf-8') as f:
#             json.dump(result, f, indent=2, ensure_ascii=False)
#         logging.info(f"Saved summary for file: {file_name}")

#     # Process non-.md files by copying them directly
#     for idx, file_path in enumerate(non_md_files, start=1):
#         # Extract relative paths and set output directory
#         rel_dir = os.path.relpath(os.path.dirname(file_path), source_folder)
#         output_dir = os.path.join(target_folder, rel_dir)
#         os.makedirs(output_dir, exist_ok=True)

#         # Copy the file directly to the target directory
#         target_file_path = os.path.join(output_dir, os.path.basename(file_path))
#         if not os.path.exists(target_file_path):  # Check if the file already exists
#             shutil.copy(file_path, target_file_path)
#             logging.info(f"Copied non-md file {idx}/{total_non_md_files}: {os.path.basename(file_path)}")
#         else:
#             logging.info(f"File already exists: {target_file_path}, skipping copy...")

# Main function to handle argument parsing
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Clean and format files in a folder, output to a new folder.")
    parser.add_argument("--source_folder", type=str, help="Path to the input folder containing files to process.")
    parser.add_argument("--target_folder", type=str, help="Path to the output folder where cleaned files will be saved.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes to use for processing files.")
    
    # Parse arguments
    args = parser.parse_args()

    source_folder = os.path.join(CUSTOM_CORPUS_HOME, args.source_folder)
    target_folder = os.path.join(CUSTOM_CORPUS_HOME, args.target_folder)
    num_processes = args.num_processes

    # # Process the files in the given folder
    # process_files(source_folder, target_folder)

    # md_files = [os.path.join(root, file) 
    #             for root, _, files in os.walk(source_folder) 
    #             for file in files if file.endswith(".md") and "valid_document_index" not in file]
    # non_md_files = [os.path.join(root, file) 
    #                 for root, _, files in os.walk(source_folder) 
    #                 for file in files if not file.endswith(".md")]
    md_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(source_folder)
        for file in files if file.endswith(".md") and "valid_document_index" not in file and 
        not os.path.exists(os.path.join(target_folder, os.path.relpath(root, source_folder), f"{os.path.splitext(file)[0]}.summary.json"))
    ]
    non_md_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(source_folder)
        for file in files if not file.endswith(".md") and 
        not os.path.exists(os.path.join(target_folder, os.path.relpath(root, source_folder), file))
    ]

    logging.info(f"Total .md files found: {len(md_files)}")
    logging.info(f"Total non-.md files found: {len(non_md_files)}")
    input("Press Enter to continue...")

    with Pool(processes=num_processes) as pool:
        # Process .md files for summary and FAQ extraction
        pool.starmap(process_md_file, [(file, source_folder, target_folder) for file in md_files])

    with Pool(processes=num_processes) as pool:
        # Process non-.md files for direct copying
        pool.starmap(copy_non_md_file, [(file, source_folder, target_folder) for file in non_md_files])

# Entry point of the script
if __name__ == "__main__":
    main()
