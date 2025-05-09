import os
import re
import argparse

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

# Define the function to process the text
def process_text(content, base_url, exclude_prefixes):
    # 1. Replace multiple consecutive newlines with a single newline
    content = re.sub(r'\n+', '\n', content)
    
    # 2. Replace []() pattern if the URL inside () starts with '/' and not in the exclude list
    def replacement(match):
        text = match.group(1)
        url = match.group(2)
        
        # Check if the URL starts with any of the excluded prefixes
        if any(url.startswith(prefix) for prefix in exclude_prefixes):
            return f'[{text}]({url})'  # Keep the original URL unchanged
        else:
            return f'[{text}]({base_url}{url})'  # Add the base_url as prefix
    
    content = re.sub(r'\[([^\]]+)\]\((/[^)]+)\)', replacement, content) # text and image
    
    return content

# Define the function to extract all the URLs inside () that match the pattern []()
def extract_links(content):
    # Use a regular expression to find all patterns like []()
    # This will capture everything inside the parentheses ().
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    
    # Find all matches in the content
    matches = re.findall(pattern, content)
    
    # Extract and return only the parts inside the parentheses ()
    urls = [match[1] for match in matches]
    
    return urls

# Define the function to remove numbered lines from code blocks
def remove_numbered_lines(content):
    # Regular expression to match code blocks that contain consecutive line numbers
    # The pattern assumes the numbers start from 1 and are consecutive integers followed by a new line.
    pattern = r'(?:\d+\n)+'  # This pattern matches multiple lines consisting of just a number

    # Replace all occurrences of such numbered lines with an empty string
    cleaned_content = re.sub(pattern, '', content)

    return cleaned_content

# Define the function to recursively process files in the folder
def process_folder(folder_path, new_folder_suffix, base_url, exclude_prefixes=None):
    new_folder_path = folder_path + new_folder_suffix
    
    # Create the new folder if it doesn't exist
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    
    urls = []

    # Traverse all files and subfolders in the directory
    for root, dirs, files in os.walk(folder_path):
        # Build the relative path for the new folder
        relative_path = os.path.relpath(root, folder_path)
        target_folder = os.path.join(new_folder_path, relative_path)
        
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # Process each file
        for file_name in files:
            file_path = os.path.join(root, file_name)
            new_file_path = os.path.join(target_folder, file_name)
            
            # Process only text files, you can expand this to handle other types
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            content = remove_numbered_lines(content)
            
            # Extract all the URLs from the processed content
            urls.extend(extract_links(content))
            
            # Process the content of the file
            processed_content = process_text(content, base_url, exclude_prefixes)

            # Write the processed content to the new file in the new folder
            with open(new_file_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
    
    print(f'All files processed and saved to: {new_folder_path}')

    return urls

# Main function to handle command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Process text files in a folder and save to a new folder with updated content.")
    
    # Positional argument: folder path
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder to process")
    # Optional argument: new folder suffix
    parser.add_argument('--suffix', type=str, required=True, help="Suffix to append to the new folder")
    # Optional argument: base URL
    parser.add_argument('--base_url', type=str, required=True, help="Base URL to prefix for links")
    # Optional argument: exclude prefixes
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help="List of URL prefixes to exclude from adding the base URL")
    
    args = parser.parse_args()
    
    folder_path = os.path.join(CUSTOM_CORPUS_HOME, args.folder_path)

    # Call the folder processing function with provided arguments
    urls = process_folder(folder_path, args.suffix, args.base_url, args.exclude)

    urls = set(urls)  # Remove duplicates
    urls = sorted(urls)  # Sort the URLs alphabetically

    # Save the extracted URLs to a file
    with open('useless/openai_clear_urls.txt', 'w', encoding='utf-8') as f:
        for url in urls:
            f.write(f'{url}\n')

if __name__ == "__main__":
    main()
        