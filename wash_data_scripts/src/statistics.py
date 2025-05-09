import os
import argparse
import tiktoken  # Ensure you have installed the tiktoken library from OpenAI

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

# (1) Generate the file tree structure and write it to a text file
def generate_file_tree(directory, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for dirpath, dirnames, filenames in os.walk(directory):
            # Determine the folder depth (level) and format accordingly
            level = dirpath.replace(directory, '').count(os.sep)
            indent = ' ' * 4 * level
            f.write(f'{indent}{os.path.basename(dirpath)}/\n')
            sub_indent = ' ' * 4 * (level + 1)
            for filename in filenames:
                f.write(f'{sub_indent}{filename}\n')

# (2) Recursively count all files and calculate tokens using tokenizer API
def count_files_and_tokens(directory):
    enc = tiktoken.get_encoding("cl100k_base")  # Use the appropriate encoding based on the model
    total_files = 0
    total_tokens = 0
    max_tokens = 0  # Variable to track the maximum token count
    max_tokens_file = ''  # File with the maximum tokens
    
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if filename.endswith('.md') and "valid_document_index" not in filename:
                total_files += 1
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tokens = len(enc.encode(content, disallowed_special=()))
                        total_tokens += tokens
                        
                        # Check if this file has the most tokens
                        if tokens > max_tokens:
                            max_tokens = tokens
                            max_tokens_file = file_path
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    # Calculate average tokens per file
    avg_tokens_per_file = total_tokens / total_files if total_files > 0 else 0
    return total_files, total_tokens, avg_tokens_per_file, max_tokens, max_tokens_file

# Main function to run both tasks
def main(directory):
    # (1) Generate file tree
    output_file = os.path.join(directory, 'dir_file_tree.txt')
    generate_file_tree(directory, output_file)
    
    # (2) Count files and tokens
    total_files, total_tokens, avg_tokens, max_tokens, max_tokens_file = count_files_and_tokens(directory)
    
    # Print the results
    print(f"Total number of files: {total_files}")
    print(f"Total number of tokens: {total_tokens}")
    print(f"Average tokens per file: {avg_tokens}")
    print(f"Maximum tokens in a single file: {max_tokens} (File: {max_tokens_file})")

# Entry point using argparse to handle source_folder as an argument
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Generate file tree and count tokens in files.")
    parser.add_argument("--source_folder", type=str, help="The path of the folder to process")
    
    args = parser.parse_args()
    source_folder = os.path.join(CUSTOM_CORPUS_HOME, args.source_folder)
    
    # Call the main function with the provided folder path
    main(source_folder)
