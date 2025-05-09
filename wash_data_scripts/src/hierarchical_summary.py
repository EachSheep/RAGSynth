import os
import json
import time
import tiktoken
import random
import argparse
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

tokenizer = tiktoken.get_encoding("cl100k_base")
TOKEN_LIMIT = 8000  # Assuming token limit is 8000

# Load OpenAI API configuration information
qwen_base_url = os.environ["QWEN_BASE_URL"]
qwen_api_key = os.environ["QWEN_API_KEY"]
model_name = "qwen2a5-72b-instruct"
client = OpenAI(base_url=f"{qwen_base_url}", api_key=qwen_api_key)

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

# Define function to process file content for summary
def process_content(content, file_name):
    tokens = tokenizer.encode(content, disallowed_special=())
    truncated_content = content
    
    # If the content exceeds the token limit, truncate it
    if len(tokens) > TOKEN_LIMIT:
        truncated_tokens = tokens[:TOKEN_LIMIT]
        truncated_content = tokenizer.decode(truncated_tokens)

    system_prompt = """
You are an assistant tasked with summarizing many documents according to the brief summary of them. 
Provide a short summary about what all these document is about in Markdown format.
Return the result as a JSON object with 'summary' keys.  
"""

    user_prompt = f"""
Here is the content of all these documents.
{truncated_content}

DON't OUTPUT ANY CONTENT EXCEPT FOR THE ANSWER IN THE JSON FORMAT.
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
            # Extract the returned content
            result = response.choices[0].message.content

            # Try to extract the largest JSON block
            largest_json_block = extract_largest_json_block(result)
            
            if largest_json_block:
                try:
                    parsed_result = json.loads(largest_json_block)
                    return parsed_result  # Return if successfully parsed
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}")
            return None
        
        except json.JSONDecodeError:
            print(f"Error decoding JSON for file {file_name}, retrying...")
            cur_try_time += 1
            time.sleep(1)  # Wait before retrying
            
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            cur_try_time += 1
            time.sleep(1)

    return None

# Define a function to recursively traverse directories and process summary.json files
def process_directory(directory):
    folder_structure = {}

    # First process subdirectories
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            subdir_summary, subdir_structure = process_directory(subdir_path)
            folder_structure[subdir] = {
                "type": "folder",
                "summary": subdir_summary,  # Include the summary of the subdir
                "children": subdir_structure  # Recursive structure of the subdir
            }

    # Now process files in the current directory
    file_summaries = []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        if file_name.endswith(".summary.json") and os.path.isfile(file_path):
            with open(file_path, "r") as f:
                try:
                    content = json.load(f)
                    summary = content.get("summary")
                    if summary:
                        folder_structure[file_name] = {
                            "type": "file",
                            "summary": summary  # Include the file summary
                        }
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
    
    # Aggregate summaries from files and subdirectories
    all_summaries = [v["summary"] for k, v in folder_structure.items() if "summary" in v]
    if all_summaries:
        aggregated_summary = process_content(json.dumps({"summaries": all_summaries}, ensure_ascii=False), directory)
        return aggregated_summary, folder_structure
    
    return None, folder_structure

# Main function to start the recursive process
def main(source_folder, target_folder):
    final_summary, combined_structure = process_directory(source_folder)
    
    folder_name = source_folder.replace("/", "_").replace("\\", "_").replace(":", "_")
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Save the final structure that includes both summaries and structure
    combined_structure["folder_summary"] = final_summary  # Attach folder's own summary
    dump_path = os.path.join(target_folder, f"{folder_name}.json")
    with open(dump_path, "w") as outfile:
        json.dump(combined_structure, outfile, indent=2, ensure_ascii=False)

    return combined_structure

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Recursively process folders and summarize contents.")
    parser.add_argument("--source_folder", type=str, help="Path to the input folder containing files to process.")
    parser.add_argument("--target_folder", type=str, help="Path to the output folder.")
    
    # Parse arguments
    args = parser.parse_args()

    source_folder = os.path.join(CUSTOM_CORPUS_HOME, args.source_folder)
    target_folder = os.path.join(CUSTOM_CORPUS_HOME, args.target_folder)

    final_result = main(source_folder, target_folder)
