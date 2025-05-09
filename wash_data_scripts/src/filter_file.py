import os
import json
import shutil
import logging
import argparse

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def process_valid_files(source_folder, summary_folder, target_folder, valid_index_file, faq_index_file):
    valid_files = []  # To store paths of valid files
    faq_files = []    # To store files containing FAQ along with their content
    invalid_files = [] # To store paths of invalid files
    total_files = 0 # To count total files with .summary.json extension

    # Traverse through the source folder looking for summary.json files
    for root, dirs, files in os.walk(summary_folder):
        for file in files:
            if file.endswith(".summary.json"):
                total_files += 1
                summary_file_path = os.path.join(root, file)
                with open(summary_file_path, 'r', encoding='utf-8') as f:
                    # print("summary_file_path:", summary_file_path)
                    summary_data = json.load(f)
                    
                    # Check if the document is valid
                    if summary_data.get("valid_document"):
                        # Determine the corresponding .md file path
                        original_md_file = summary_file_path.replace(".summary.json", ".md")
                        original_md_file = original_md_file.replace(summary_folder, source_folder)
                        if os.path.exists(original_md_file):
                            # Recreate the directory structure in the target folder
                            rel_path = os.path.relpath(original_md_file, source_folder)
                            target_file_path = os.path.join(target_folder, rel_path)
                            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

                            # Copy the valid .md file to the new target folder
                            shutil.copy2(original_md_file, target_file_path)
                            valid_files.append("/" + os.path.relpath(original_md_file, CUSTOM_CORPUS_HOME))
                    else:
                        # Collect valid file paths for reporting
                        original_md_file = summary_file_path.replace(".summary.json", ".md")
                        original_md_file = original_md_file.replace(summary_folder, source_folder)
                        invalid_files.append("/" + os.path.relpath(original_md_file, CUSTOM_CORPUS_HOME))
                        invalid_files.append("/" + os.path.relpath(summary_file_path, CUSTOM_CORPUS_HOME))

                    # Check if the document contains FAQ
                    faq = summary_data.get("faq", [])
                    if faq:
                        rel_path = os.path.relpath(summary_file_path.replace(".summary.json", ".md"), source_folder)
                        faq_files.append({
                            "file": rel_path,
                            "faq": faq
                        })

    # Write valid files index to a file
    with open(valid_index_file, 'w', encoding='utf-8') as f:
        # json.dump(valid_files, f, indent=2, ensure_ascii=False)
        for file in valid_files:
            f.write(f"[]({file})\n")

    # Write FAQ index to a file
    with open(faq_index_file, 'w', encoding='utf-8') as f:
        f.write("# FAQ Document Index\n\n")
        for faq_entry in faq_files:
            f.write(f"File: {faq_entry['file']}\n")
            for item in faq_entry["faq"]:
                f.write(f"Q: {item['question']}\nA: {item['answer']}\n")
            f.write("\n")
    
    # Log the invalid files information and the exclusion rate
    invalid_file_count = len(invalid_files) // 2
    exclusion_rate = invalid_file_count / total_files if total_files > 0 else 0
    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Valid files: {len(valid_files)}")
    logging.info(f"Invalid files: {invalid_file_count}")
    logging.info(f"Exclusion rate: {exclusion_rate:.2%}")

    # Write invalid files list to a separate file for review
    invalid_index_file = valid_index_file.replace("valid_document_index.md", "invalid_document_index.md")
    with open(invalid_index_file, 'w', encoding='utf-8') as f:
        # json.dump(invalid_files, f, indent=2, ensure_ascii=False)
        for file in invalid_files:
            f.write(f"[]({file})\n")

def copy_non_md_files(source_folder, target_folder):
    # Traverse through the source folder
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if not file.endswith(".md") and not file.endswith(".summary.json"):
                source_file_path = os.path.join(root, file)
                rel_path = os.path.relpath(source_file_path, source_folder)
                target_file_path = os.path.join(target_folder, rel_path)

                # Recreate the directory structure in the target folder
                os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

                # Copy the non-md file to the target folder
                shutil.copy2(source_file_path, target_file_path)


def main(source_folder, summary_folder, target_folder):
    # Define paths for index files
    valid_index_file = os.path.join(target_folder, "valid_document_index.md")
    faq_index_file = os.path.join(target_folder, "faq_document_index.txt")

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # Process valid .md files and generate indexes
    process_valid_files(source_folder, summary_folder, target_folder, valid_index_file, faq_index_file)

    # Copy non-md files while preserving directory structure
    copy_non_md_files(summary_folder, target_folder)

    logging.info(f"Processing completed. Valid documents and FAQ indexes saved in {target_folder}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Filter valid documents and generate indexes.")
    argparser.add_argument("--source_folder", type=str, help="The path of the source folder")
    argparser.add_argument("--summary_folder", type=str, help="The path of the summary folder")
    argparser.add_argument("--target_folder", type=str, help="The path of the target folder")
    args = argparser.parse_args()

    source_folder = os.path.join(CUSTOM_CORPUS_HOME, args.source_folder)
    summary_folder = os.path.join(CUSTOM_CORPUS_HOME, args.summary_folder)
    target_folder = os.path.join(CUSTOM_CORPUS_HOME, args.target_folder)

    logging.info(f"Source folder: {source_folder}")
    logging.info(f"Summary folder: {summary_folder}")
    logging.info(f"Target folder: {target_folder}")

    main(source_folder, summary_folder, target_folder)
