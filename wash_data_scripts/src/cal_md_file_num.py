import os
import argparse

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME")
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable not set")

def count_md_files(source_folder):
    md_count = 0
    for root, _, files in os.walk(source_folder):
        md_count += sum(1 for file in files if file.endswith('.md') and "valid_document_index" not in file)
    return md_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count .md files in a source_folder.")
    parser.add_argument("--source_folder", type=str, help="Path to the target source_folder")
    args = parser.parse_args()

    source_folder = os.path.join(CUSTOM_CORPUS_HOME, args.source_folder)
    md_file_count = count_md_files(source_folder)
    print(f"Total number of .md files: {md_file_count}")
