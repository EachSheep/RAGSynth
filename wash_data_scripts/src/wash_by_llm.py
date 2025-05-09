import re
import os
import tiktoken
import argparse
import json
import logging
import time
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
TOKEN_LIMIT = 4000  # Maximum number of tokens per chunk

# Helper function to split content into paragraphs
def split_into_paragraphs(content):
    return re.split(r'\s*\n\s*\n\s*', content)  # Match any whitespace around double newlines

# def split_into_paragraphs(content):
#     return content.split("\n")  # Split paragraphs by double newlines

# Helper function to split paragraphs into sentences
def split_into_sentences(paragraph):
    import re
    # Use regular expression to split by sentence-ending punctuation marks
    return re.split(r'(?<=[.!?])\s+', paragraph)

# Function to split content into chunks, prioritizing paragraph integrity
def split_content(content):
    paragraphs = split_into_paragraphs(content)
    chunks = []
    current_chunk = []
    len_current_chunk = 0

    for paragraph in paragraphs:
        tokens = tokenizer.encode(paragraph, disallowed_special=())  # Disallow special tokens

        # If paragraph alone exceeds token limit, split by sentences
        if len(tokens) > TOKEN_LIMIT:
            sentences = split_into_sentences(paragraph)
            for sentence in sentences:
                sentence_tokens = tokenizer.encode(sentence)
                if len_current_chunk + len(sentence_tokens) > TOKEN_LIMIT:
                    chunks.append(". ".join(current_chunk))
                    current_chunk = []
                    len_current_chunk = 0
                current_chunk.append(sentence)
                len_current_chunk += len(sentence_tokens)
        else:
            # If adding this paragraph exceeds the limit, start a new chunk
            if len_current_chunk + len(tokens) > TOKEN_LIMIT:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                len_current_chunk = 0
            current_chunk.append(paragraph)
            len_current_chunk += len(tokens)

    # Append the last chunk if there is any content left
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    logging.info(f"Total API calls required for this file: {len(chunks)}")  # Added logging for total chunks

    return chunks

# Process a single file, used in the multiprocessing pool
def process_file(args):
    file_path, source_folder, target_folder, internal_server_error_files = args
    # Relative path setup and file paths
    rel_dir = os.path.relpath(os.path.dirname(file_path), source_folder)
    output_dir = os.path.join(target_folder, rel_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, os.path.basename(file_path))
    
    # Skip file if it already exists in target folder
    if os.path.exists(output_file_path):
        logging.info(f"File {output_file_path} already exists. Skipping.")
        return
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content into chunks
    token_chunks = split_content(content)
    total_chunks = len(token_chunks)
    cleaned_content = ""
    file_failed = False  # Flag to track if a file encounters server error

    # Log the start of file processing
    logging.info(f"Processing file: {file_path}, {total_chunks} chunks to process.")

    # Process each chunk of tokens
    for chunk in token_chunks:
        cleaned_content_chunk = call_openai_api(chunk)
        if cleaned_content_chunk == None:
            internal_server_error_files.append(file_path)  # Add to shared error log
            logging.error(f"Skipping file due to internal server error: {file_path}")
            file_failed = True
            break
        cleaned_content += cleaned_content_chunk

    if not file_failed and cleaned_content.strip():
        save_cleaned_content(output_file_path, cleaned_content)

# Multi-process function to process all files
def process_files(source_folder, target_folder, num_processes):
    all_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(source_folder)
        for file in files if file.endswith(".md")and "valid_document_index" not in file
    ]
    total_files = len(all_files)
    logging.info(f"Total files to process: {total_files}")

    new_files = []
    for file_path in all_files:
        # Relative path setup and file paths
        rel_dir = os.path.relpath(os.path.dirname(file_path), source_folder)
        output_dir = os.path.join(target_folder, rel_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        if not os.path.exists(output_file_path):
            new_files.append(file_path)
    all_files = new_files
    logging.info(f"Total new files to process: {len(all_files)}")
    input("Press Enter to continue...")
        
    # Use Manager to handle shared data across processes
    manager = Manager()
    internal_server_error_files = manager.list()

    # Define process pool and map files to processes
    with Pool(processes=num_processes) as pool:
        pool.map(
            process_file,
            [(file_path, source_folder, target_folder, internal_server_error_files) for file_path in all_files]
        )

    # Save files that encountered errors
    if internal_server_error_files:
        error_log_path = os.path.join(target_folder, "internal_server_error.txt")
        with open(error_log_path, 'w', encoding='utf-8') as error_log_file:
            for error_file in internal_server_error_files:
                error_log_file.write(f"{error_file}\n")

# # Recursively traverse the folder and process .md files
# def process_files(source_folder, target_folder):
#     total_files = sum([len(files) for _, _, files in os.walk(source_folder) if files])
#     file_counter = 0
#     internal_server_error_files = []

#     for root, dirs, files in os.walk(source_folder):
#         for file in files:
#             if file.endswith(".md")and "valid_document_index" not in file:
#                 file_counter += 1

#                 # Get the relative path to maintain folder structure
#                 rel_dir = os.path.relpath(root, source_folder)
#                 # Create corresponding directory in the output folder
#                 output_dir = os.path.join(target_folder, rel_dir)
#                 os.makedirs(output_dir, exist_ok=True)

#                 # Determine file paths
#                 file_path = os.path.join(root, file)
#                 output_file_path = os.path.join(output_dir, file)

#                 # Check if the output file already exists, if so, skip processing
#                 if os.path.exists(output_file_path):
#                     logging.info(f"File {output_file_path} already exists. Skipping.")
#                     continue  # Skip to the next file

#                 start_time = time.time()

#                 # Read the file content
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     content = f.read()

#                 logging.info(f"Processing file {file_counter}/{total_files}: {file_path}")

#                 # Split content into chunks intelligently
#                 token_chunks = split_content(content)

#                 # Track API call count for the current file
#                 total_chunks = len(token_chunks)
#                 current_chunk_num = 0

#                 # Process each chunk of tokens
#                 cleaned_content = ""
#                 file_failed = False  # Flag to track if a file encounters server error
#                 for chunk in token_chunks:
#                     current_chunk_num += 1
#                     logging.info(f"Making API call {current_chunk_num}/{total_chunks} for file {file_counter}/{total_files}")
#                     cleaned_content_chunk = call_openai_api(chunk)
#                     if cleaned_content_chunk == None:
#                         # If an internal server error occurred, mark the file and break
#                         internal_server_error_files.append(file_path)
#                         file_failed = True
#                         logging.error(f"Skipping file due to internal server error: {file_path}")
#                         break
#                     cleaned_content += cleaned_content_chunk

#                 if file_failed:
#                     continue  # Skip to the next file if this file failed

#                 # Save cleaned content if not empty or irrelevant
#                 if cleaned_content.strip():
#                     save_cleaned_content(output_file_path, cleaned_content)

#                 elapsed_time = time.time() - start_time
#                 logging.info(f"Completed file {file_counter}/{total_files}: {file_path} in {elapsed_time:.2f} seconds")

#     # Save the list of files that encountered internal server errors
#     if internal_server_error_files:
#         error_log_path = os.path.join(target_folder, "internal_server_error.txt")
#         with open(error_log_path, 'w', encoding='utf-8') as error_log_file:
#             for error_file in internal_server_error_files:
#                 error_log_file.write(f"{error_file}\n")


# Call OpenAI API to clean the content with retry mechanism
def call_openai_api(content_chunk):
    system_prompt = """
You are an assistant tasked with cleaning and formatting text converted from HTML or PDF documents. 
The goal is to remove unnecessary content like formatting errors, irrelevant metadata, conversion artifacts, advertisements, logos, 
login pages, navigation menus, footers, social media buttons, copyright notices, terms and conditions, empty lists, 
incomplete sections, and other pages lacking useful content, while keeping the text true to the original meaning. 
Do not change the meaning of any part of the document. 
Ensure the output follows clean Markdown formatting.
"""
    
    user_prompt = f"""Your task is to:
1. Remove all irrelevant or incomplete content mentioned.
2. Correct any formatting issues and output the text in proper Markdown format.
3. Keep the original meaning and content as accurate as possible.
    
Here is the text:

{content_chunk}
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
            return response.choices[0].message.content
        
        except Exception as e:
            logging.error(f"Error during OpenAI API call: {e}")
            if "Internal Server Error" in str(e):
                return None  # Return None for internal server errors to skip processing
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

# Save the cleaned content to a new Markdown file
def save_cleaned_content(output_file_path, cleaned_content):
    # Write the cleaned content to the new file in the output directory
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

# Main function to handle argument parsing
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Clean and format files in a folder, output to a new folder.")
    parser.add_argument("--source_folder", type=str, help="Path to the input folder containing files to process.")
    parser.add_argument("--target_folder", type=str, help="Path to the output folder where cleaned files will be saved.")
    parser.add_argument("--num_processes", type=int, default=4, help="Number of parallel processes to use for file processing.")

    # Parse arguments
    args = parser.parse_args()
    
    source_folder = os.path.join(CUSTOM_CORPUS_HOME, args.source_folder)
    target_folder = os.path.join(CUSTOM_CORPUS_HOME, args.target_folder)

    # Process the files in the given folder
    # process_files(source_folder, target_folder)
    process_files(source_folder, target_folder, args.num_processes)

# Entry point of the script
if __name__ == "__main__":
    main()
