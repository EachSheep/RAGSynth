import os
import subprocess
import logging
from bs4 import BeautifulSoup  # For HTML to Markdown conversion
import markdownify  # For converting HTML content to Markdown
import re
import shutil
import argparse  # For command line argument parsing

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to strip specific extensions like .html.md, .htm.md, or .pdf.md from filenames
def strip_extension(file_name):
    if file_name.endswith('.html.md'):
        return file_name[:-3]  # Strip ".html.md"
    elif file_name.endswith('.htm.md'):
        return file_name[:-3]  # Strip ".htm.md"
    elif file_name.endswith('.pdf.md'):
        return file_name[:-3]  # Strip ".pdf.md"
    else:
        return file_name

# Define the function to check if a corresponding file exists in source_dir
def file_exists_for_url(url, current_file_path, source_dir):
    # If the URL starts with '/', treat it as an absolute path
    if url.startswith('/'):
        # Remove leading slash if present in the URL for absolute path handling
        relative_path = url.lstrip('/')
        full_dir_path = os.path.join(source_dir, os.path.dirname(relative_path))
        full_dir_path = os.path.abspath(full_dir_path)
    else:
        # For relative paths, use current file's directory as the base
        current_dir = os.path.dirname(current_file_path)
        relative_path = os.path.join(current_dir, url)
        full_dir_path = os.path.dirname(relative_path)
        full_dir_path = os.path.abspath(full_dir_path)

    # Check if the directory exists
    if not os.path.exists(full_dir_path):
        return False

    # Get the file name
    file_name = os.path.basename(relative_path)

    # Get all files in the directory
    files_in_dir = os.listdir(full_dir_path)

    # Compare the filename with each file in the directory, ignoring the extensions
    for file in files_in_dir:
        # Strip extension from the file in the directory
        file_without_ext = strip_extension(file)
        
        # If the filename without extension matches, return True
        if file_without_ext == file_name:
            return True

    # If no matching file is found, return False
    return False

# Define the function to process the text
def process_text(content, source_dir, current_file_path, base_url):
    # 1. Replace multiple consecutive newlines with a single newline
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # 2. Define the replacement function for normal links
    def replacement_link(match):
        text = match.group(1)
        url = match.group(2)
        
        # Check if the corresponding file exists
        if url.startswith("http"):
            return f'[{text}]({url})'
        elif file_exists_for_url(url, current_file_path, source_dir):
            if url.startswith('/'):
                relative_path = url.lstrip('/')  # Remove leading slash if present in the URL for absolute path handling
            else:
                relative_path = os.path.relpath(
                    os.path.join(os.path.dirname(current_file_path), url), source_dir)
            if relative_path.endswith('.pdf') or \
                relative_path.endswith('.html') or \
                    relative_path.endswith('.htm'):
                ret_str = f'[{text}](/{relative_path}.md)'
            else:
                ret_str = f'[{text}](/{relative_path})'

            while '//' in ret_str:
                ret_str = ret_str.replace('//', '/')
            ret_str = ret_str.replace('\\', '/')
            return ret_str
        else:
            if url.startswith('/'):
                # Remove leading slash if present in the URL for absolute path handling
                ret_str = f'[{text}]({base_url}/{url})'  # Add the base_url as prefix
                while '//' in ret_str:
                    ret_str = ret_str.replace('//', '/')
                return ret_str
            else:
                # For relative paths, use current file's directory as the base
                if url.startswith('/'):
                    relative_path = url.lstrip('/')  # Remove leading slash if present in the URL for absolute path handling
                else:
                    relative_path = os.path.relpath(
                        os.path.join(os.path.dirname(current_file_path), url), source_dir)
                ret_str = f'[{text}]({base_url}/{relative_path})'  # Add the base_url as prefix
                while '//' in ret_str:
                    ret_str = ret_str.replace('//', '/')
                ret_str = ret_str.replace('\\', '/')
                return ret_str
    
    # 3. Define the replacement function for image links
    def replacement_image(match):
        alt_text = match.group(1)
        url = match.group(2)
        
        if url.startswith("http"):
            return f'![{alt_text}]({url})'
        # Check if the URL starts with any of the excluded prefixes or if the corresponding file exists
        elif file_exists_for_url(url, current_file_path, source_dir):
            if url.startswith('/'):
                relative_path = url.lstrip('/')  # Remove leading slash if present in the URL for absolute path handling
            else:
                relative_path = os.path.relpath(
                    os.path.join(os.path.dirname(current_file_path), url), source_dir)
            ret_str = f'![{alt_text}](/{relative_path})'
            while '//' in ret_str:
                ret_str = ret_str.replace('//', '/')
            ret_str = ret_str.replace('\\', '/')
            return ret_str  # Keep the original URL unchanged
        else:
            if url.startswith('/'):
                # Remove leading slash if present in the URL for absolute path handling
                ret_str = f'![{alt_text}]({base_url}/{url})'
                while '//' in ret_str:
                    ret_str = ret_str.replace('//', '/')
                return ret_str
            else:
                # For relative paths, use current file's directory as the base
                if url.startswith('/'):
                    relative_path = url.lstrip('/')  # Remove leading slash if present in the URL for absolute path handling
                else:
                    relative_path = os.path.relpath(
                        os.path.join(os.path.dirname(current_file_path), url), source_dir)
                ret_str = f'![{alt_text}]({base_url}/{relative_path})'
                while '//' in ret_str:
                    ret_str = ret_str.replace('//', '/')
                ret_str = ret_str.replace('\\', '/')
                return ret_str
    
    # 4. Process image links first (those starting with '!')
    content = re.sub(r'!\[([^\]]+)\]\(([^)]+)\)', replacement_image, content)
    
    # 5. Process normal text links (exclude those starting with '!')
    content = re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)', replacement_link, content)
    
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

# Convert HTML/HTM to Markdown
def convert_html_to_markdown(html_content):
    return markdownify.markdownify(html_content, heading_style="ATX")

# Process HTML file
def extract_main_content_to_markdown_from_fandom(source_file_path):
    with open(source_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the <main> tag with class "page__main"
        main_content = soup.find('main', class_='page__main')
        
        # If the <main> tag is found, convert its content to Markdown
        if main_content:
            markdown_content = convert_html_to_markdown(str(main_content))
            return markdown_content
        else:
            return ""

def extract_main_content_to_markdown_from_drug(source_file_path):
    with open(source_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the <div> tag with id "content" and class "ddc-main-content"
        main_content = soup.find('div', id='content', class_='ddc-main-content')
        
        # If the <div> is found, convert its content to Markdown
        if main_content:
            markdown_content = convert_html_to_markdown(str(main_content))
            return markdown_content
        else:
            return ""

# Process HTML file
def extract_main_content_to_markdown_from_webcopy(source_file_path):
    with open(source_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the <div> with class "card card-lg"
        card_content = soup.find('div', class_='card card-lg')
        
        # Find the <h1> with class "page-title"
        page_title = soup.find('h1', class_='page-title')
        
        # Initialize markdown content list
        markdown_content = []

        # If the <h1> with class "page-title" is found, convert its content
        if page_title:
            markdown_content.append(convert_html_to_markdown(str(page_title)))
        
        # If the <div> with class "card card-lg" is found, convert its content
        if card_content:
            markdown_content.append(convert_html_to_markdown(str(card_content)))
        
        # Join the converted markdown parts
        return "\n\n".join(markdown_content) if markdown_content else ""

# Process HTML file
def extract_main_content_to_markdown_from_mayoclinic(source_file_path):
    with open(source_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Try to find the <main> or <article> tag with id "main-content"
        main_content = soup.find(['main', 'article'], id='main-content')
        
        # Initialize markdown content
        markdown_content = ""
        
        # If the <main> or <article> with id "main-content" is found, convert its content to Markdown
        if main_content:
            markdown_content = markdownify.markdownify(str(main_content), heading_style="ATX")
        else:
            # Fallback to find the <div> with specific classes
            fallback_div = soup.find('div', class_="aem-container cmp-container--centered cmp-container--ccg-body-wrapper cmp-column-control__container")
            if fallback_div:
                markdown_content = markdownify.markdownify(str(fallback_div), heading_style="ATX")
            else:
                markdown_content = ""
        
        return markdown_content

# Use Marker to convert PDF to Markdown via the marker_single command
def convert_pdf_to_markdown_with_marker(pdf_path, output_folder):
    try:    
        logging.info(f'Starting PDF conversion for {pdf_path}')
        # Build and run the marker_single command
        subprocess.run(['marker_single', pdf_path, output_folder], check=True)
        logging.info(f'Successfully converted PDF to Markdown: {pdf_path}')
    except subprocess.CalledProcessError as e:
        logging.error(f'Error converting {pdf_path}: {e}')
    

# Process a single file: either HTML/HTM or PDF
def process_file(source_file_path, target_file_path, source_dir, base_url):
    # If is a directory, skip
    if os.path.isdir(source_file_path):
        return

    file_extension = source_file_path.split('.')[-1].lower()
    
    # Handle HTML/HTM files
    if file_extension in ['html', 'htm']:
        if not re.search(r'js\.html?$', source_file_path):  # Exclude js.html and js.htm files
            try:
                logging.info(f'Processing HTML/HTM file: {source_file_path}')
                if "fandom" in source_dir:
                    markdown_content = extract_main_content_to_markdown_from_fandom(source_file_path)
                elif "cyotek" in source_dir:
                    markdown_content = extract_main_content_to_markdown_from_webcopy(source_file_path)
                elif "drugs" in source_dir:
                    markdown_content = extract_main_content_to_markdown_from_drug(source_file_path)
                elif "mayoclinic" in source_dir:
                    markdown_content = extract_main_content_to_markdown_from_mayoclinic(source_file_path)
                else:
                    with open(source_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                        soup = BeautifulSoup(html_content, 'html.parser')
                        markdown_content = convert_html_to_markdown(str(soup))
                
                markdown_content = remove_numbered_lines(markdown_content)
                markdown_content = process_text(markdown_content, source_dir, source_file_path, base_url)

                # Only create target directory if it doesn't exist and is needed
                target_dir = os.path.dirname(target_file_path)
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    logging.info(f'Created directory: {target_dir}')

                if markdown_content:
                    with open(target_file_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                logging.info(f'Successfully converted HTML/HTM to Markdown: {source_file_path}')
            except Exception as e:
                logging.error(f'Error processing HTML/HTM file {source_file_path}: {e}')

    # Handle PDF files
    elif file_extension == 'pdf':
        output_folder = os.path.dirname(target_file_path)  # Get the output folder
        output_name = os.path.basename(target_file_path)  # Get the output file name, .pdf.md
        # Only create target directory if it doesn't exist and is needed
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            logging.info(f'Created directory: {output_folder}')
        convert_pdf_to_markdown_with_marker(source_file_path, output_folder)
        md_path = os.path.join(output_folder, output_name[:-7], output_name[:-7] + '.md')
        with open(md_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        markdown_content = remove_numbered_lines(markdown_content)
        markdown_content = process_text(markdown_content, source_dir, source_file_path, base_url)
        with open(target_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        # Remove the intermediate file
        if os.path.isdir(os.path.join(output_folder, output_name[:-7])):
            shutil.rmtree(os.path.join(output_folder, output_name[:-7]))
        elif os.path.isfile(os.path.join(output_folder, output_name[:-7])):
            os.remove(os.path.join(output_folder, output_name[:-7]))
        else:
            print(f"{os.path.join(output_folder, output_name[:-7])} does not exist")
    elif file_extension == 'md':
        with open(source_file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        markdown_content = remove_numbered_lines(markdown_content)
        markdown_content = process_text(markdown_content, source_dir, source_file_path, base_url)
        with open(target_file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    else: # copy other files
        target_dir = os.path.dirname(target_file_path[:-3])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        shutil.copyfile(source_file_path, target_file_path[:-3]) # remove .md extension
    

# Traverse the folder and convert files
def traverse_and_convert(source_dir, target_dir, base_url):
    logging.info(f'Starting to traverse the directory: {source_dir}')
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(source_file_path, source_dir)
            target_file_path = os.path.join(target_dir, relative_path)

            # Change target file extension to .md for HTML and PDF files
            target_file_path += '.md'

            # Skip the file if it has already been converted
            if os.path.exists(target_file_path):
                logging.info(f'Skipping already converted file: {target_file_path}')
                continue
            
            logging.info(f'Starting to process file: {source_file_path}')
            process_file(source_file_path, target_file_path, source_dir, base_url)

# Main function
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert HTML/HTM and PDF files to Markdown.')
    parser.add_argument('--source_folder', type=str, help='The source folder containing HTML/HTM or PDF files.')
    parser.add_argument('--target_folder', type=str, help='The target folder of generated MD files.')
    parser.add_argument('--base_url', type=str, required=True, help="Base URL to prefix for links")
    args = parser.parse_args()

    source_folder = args.source_folder
    source_folder = os.path.join(CUSTOM_CORPUS_HOME, source_folder)
    target_folder = args.target_folder
    target_folder = os.path.join(CUSTOM_CORPUS_HOME, target_folder)
    base_url = args.base_url

    # Create the new folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    logging.info(f'Starting the conversion process from {source_folder} to {target_folder}')
    
    # Start processing
    traverse_and_convert(source_folder, target_folder, base_url)

    logging.info(f'Conversion process completed. Output stored in {target_folder}')

if __name__ == '__main__':
    main()
