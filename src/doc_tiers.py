import os
import random
import tiktoken
import json

# Retrieve the CUSTOM_CORPUS_HOME environment variable
CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME")
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable not set")

doc_home = os.path.join(CUSTOM_CORPUS_HOME, "docs")

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# The next data level will include the data from the previous level
docs_tier_1 = {
    "admission.stanford.edu.filter": [
        "./ + all"
    ],
    "cmu.edu.admission.filter": [
        "./ + all"
    ],
    "berkeley.edu.admission.filter": [
        "./ + all"
    ],
    "www.notion.so.help.filter": [
        "help/guides + all"
    ],
    "docs.cyotek.com.filter": [
        "cyowcopy + all"
    ],
    "eldenring.fandom.com.wiki.filter": [
        "./ + random_200"
    ],
    "zelda.fandom.com.wiki.filter": [
        "./ + random_200"
    ],
    "hearthstone.fandom.com.wiki.filter": [
        "./ + random_200"
    ],
    "www.mayoclinic.org.filter": [
        "medical-professionals + all"
    ],
    "www.drugs.com.filter": [
        "drug-class + random_200"
    ],
    "drug-instructions-alibaba": [
        "./ + random_200"
    ]
}

docs_tier_2 = {
    "admission.stanford.edu.filter": [
        "./ + all"
    ],
    "cmu.edu.admission.filter": [
        "./ + all"
    ],
    "berkeley.edu.admission.filter": [
        "./ + all"
    ],
    "www.notion.so.help.filter": [
        "./ + all"
    ],
    "docs.cyotek.com.filter": [
        "./ + all"
    ],
    "eldenring.fandom.com.wiki.filter": [
        "./ + random_600"
    ],
    "zelda.fandom.com.wiki.filter": [
        "./ + random_600"
    ],
    "hearthstone.fandom.com.wiki.filter": [
        "./ + random_600"
    ],
    "www.mayoclinic.org.filter": [
        "medical-professionals + all",
        "diseases-conditions + random_344"
    ],
    "www.drugs.com.filter": [
        "drug-class + all",
        "alpha + random_119"
    ],
    "drug-instructions-alibaba": [
        "./ + random_600"
    ]
}

docs_tier_3 = {
    "admission.stanford.edu.filter": [
        "./ + all"
    ],
    "cmu.edu.admission.filter": [
        "./ + all"
    ],
    "berkeley.edu.admission.filter": [
        "./ + all"
    ],
    "www.notion.so.help.filter": [
        "./ + all"
    ],
    "docs.cyotek.com.filter": [
        "./ + all"
    ],
    "eldenring.fandom.com.wiki.filter": [
        "./ + random_3000"
    ],
    "zelda.fandom.com.wiki.filter": [
        "./ + random_3000"
    ],
    "hearthstone.fandom.com.wiki.filter": [
        "./ + random_3000"
    ],
    "www.mayoclinic.org.filter": [
        "./ + all"
    ],
    "www.drugs.com.filter": [
        "drug-class + all",
        "alpha + all",
        "pro + random_1865"
    ],
    "drug-instructions-alibaba": [
        "./ + random_3000"
    ]
}

docs_tier_4 = {
    "admission.stanford.edu.filter": [
        "./ + all"
    ],
    "cmu.edu.admission.filter": [
        "./ + all"
    ],
    "berkeley.edu.admission.filter": [
        "./ + all"
    ],
    "www.notion.so.help.filter": [
        "./ + all"
    ],
    "docs.cyotek.com.filter": [
        "./ + all"
    ],
    "eldenring.fandom.com.wiki.filter": [
        "./ + all"
    ],
    "zelda.fandom.com.wiki.filter": [
        "./ + all"
    ],
    "hearthstone.fandom.com.wiki.filter": [
        "./ + all"
    ],
    "www.mayoclinic.org.filter": [
        "./ + all"
    ],
    "www.drugs.com.filter": [
        "./ + all",
    ],
    "drug-instructions-alibaba": [
        "./ + all"
    ]
}

# Function to get token count for a file's content
def get_token_count(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tokens = tokenizer.encode(content, disallowed_special=())
        return len(tokens)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

# Function to get files from the directories as specified in the directory structure
def get_files_from_directories(directory_structure, last_result=None):
    result = {}

    for folder_name, paths in directory_structure.items():
        collected_files = []

        for path_rule in paths:
            path, rule = path_rule.split(" + ")
            # Construct the full path to the target folder
            full_path = os.path.join(doc_home, folder_name, path.strip("./"))
            
            if not os.path.isdir(full_path):
                print(f"Path {full_path} does not exist, skipping...")
                continue

            # Use os.walk to recursively retrieve all files from subdirectories
            all_files = [os.path.join(root, file) for root, _, files in os.walk(full_path) for file in files if file.endswith(".md") and "valid_document_index" not in file]

            if rule == "all":
                collected_files.extend(all_files)
            elif rule.startswith("random_"):
                sample_size = int(rule.split("_")[1])

                random.seed(42)
                sampled_files = random.sample(all_files, min(sample_size, len(all_files)))
                collected_files.extend(sampled_files)
                
            elif rule.startswith("length_"):
                sample_size = int(rule.split("_")[1])

                # Calculate token length for each file
                file_lengths = [(file, get_token_count(file)) for file in all_files]
                # Sort files by length from longest to shortest
                sorted_files = sorted(file_lengths, key=lambda x: x[1], reverse=True)
                # Select the top files based on length
                top_files = [file for file, _ in sorted_files[:sample_size]]
                collected_files.extend(top_files)

        for idx, file in enumerate(collected_files):
            rel_path = os.path.relpath(file, os.path.join(doc_home, folder_name))
            collected_files[idx] = rel_path

        if last_result is None:
            result[folder_name] = collected_files
        else:
            result[folder_name] = last_result[folder_name] + [file for file in collected_files if file not in last_result[folder_name]]

    return result

if __name__ == "__main__":

    os.makedirs("cache", exist_ok=True)

    print(f"Generating document tier {1}...")
    if os.path.exists("cache/docs_tier_1.json"):
        with open("cache/docs_tier_1.json", "r", encoding='utf-8') as f:
            result_1 = json.load(f)
    else:
        result_1 = get_files_from_directories(docs_tier_1)
        # with open("cache/docs_tier_1.json", "w", encoding='utf-8') as f:
        #     json.dump(result_1, f, indent=2, ensure_ascii=False)
        # write_str = ""
        # for folder, files in result_1.items():
        #     write_str += f"# {folder}: {len(files)}\n\n"
        #     for file in files:
        #         file_path = os.path.join("docs", folder, file)
        #         file_path = "/" + file_path
        #         write_str += f"[]({file_path})\n"
        #     write_str += "\n\n"
        # with open("cache/docs_tier_1.md", "w", encoding='utf-8') as f:
        #     f.write(write_str)

    print(f"Generating document tier {2}...")
    if os.path.exists("cache/docs_tier_2.json"):
        with open("cache/docs_tier_2.json", "r", encoding='utf-8') as f:
            result_2 = json.load(f)
    else:
        result_2 = get_files_from_directories(docs_tier_2, result_1)
        with open("cache/docs_tier_2.json", "w", encoding='utf-8') as f:
            json.dump(result_2, f, indent=2, ensure_ascii=False)
        result_2_minus_1 = {}
        for folder, files in result_2.items():
            result_2_minus_1[folder] = [file for file in files if file not in result_1[folder]]
        with open("cache/docs_tier_2_minus_1.json", "w", encoding='utf-8') as f:
            json.dump(result_2_minus_1, f, indent=2, ensure_ascii=False)
        # write_str = ""
        # for folder, files in result_2.items():
        #     write_str += f"# {folder}: {len(files)}\n\n"
        #     for file in files:
        #         file_path = os.path.join("docs", folder, file)
        #         file_path = "/" + file_path
        #         write_str += f"[]({file_path})\n"
        #     write_str += "\n\n"
        # with open("cache/docs_tier_2.md", "w", encoding='utf-8') as f:
        #     f.write(write_str)
    
    print(f"Generating document tier {3}...")
    if os.path.exists("cache/docs_tier_3.json"):
        with open("cache/docs_tier_3.json", "r", encoding='utf-8') as f:
            result_3 = json.load(f)
    else:
        result_3 = get_files_from_directories(docs_tier_3, result_2)
        with open("cache/docs_tier_3.json", "w", encoding='utf-8') as f:
            json.dump(result_3, f, indent=2, ensure_ascii=False)
        result_3_minus_1 = {}
        for folder, files in result_3.items():
            result_3_minus_1[folder] = [file for file in files if file not in result_1[folder]]
        with open("cache/docs_tier_3_minus_1.json", "w", encoding='utf-8') as f:
            json.dump(result_3_minus_1, f, indent=2, ensure_ascii=False)
        # write_str = ""
        # for folder, files in result_3.items():
        #     write_str += f"# {folder}: {len(files)}\n\n"
        #     for file in files:
        #         file_path = os.path.join("docs", folder, file)
        #         file_path = "/" + file_path
        #         write_str += f"[]({file_path})\n"
        #     write_str += "\n\n"
        # with open("cache/docs_tier_3.md", "w", encoding='utf-8') as f:
        #     f.write(write_str)

    print(f"Generating document tier {4}...")
    if os.path.exists("cache/docs_tier_4.json"):
        with open("cache/docs_tier_4.json", "r", encoding='utf-8') as f:
            result_4 = json.load(f)
    else:
        result_4 = get_files_from_directories(docs_tier_4, result_3)
        with open("cache/docs_tier_4.json", "w", encoding='utf-8') as f:
            json.dump(result_4, f, indent=2, ensure_ascii=False)
        result_4_minus_1 = {}
        for folder, files in result_4.items():
            result_4_minus_1[folder] = [file for file in files if file not in result_1[folder]]
        with open("cache/docs_tier_4_minus_1.json", "w", encoding='utf-8') as f:
            json.dump(result_4_minus_1, f, indent=2, ensure_ascii=False)
        # write_str = ""
        # for folder, files in result_4.items():
        #     write_str += f"# {folder}: {len(files)}\n\n"
        #     for file in files:
        #         file_path = os.path.join("docs", folder, file)
        #         file_path = "/" + file_path
        #         write_str += f"[]({file_path})\n"
        #     write_str += "\n\n"
        # with open("cache/docs_tier_4.md", "w", encoding='utf-8') as f:
        #     f.write(write_str)
    
    # # Print the result
    # for folder, files in result.items():
    #     print(f"{folder}: {len(files)}")
