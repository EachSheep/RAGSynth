import os
import json
import math
import re
import numpy as np
import argparse
from tqdm import tqdm
from openai import OpenAI
from scipy.stats import entropy
from collections import defaultdict
from scipy.special import rel_entr
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag.utils.request_openai_utils import OpenAIModel

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

API_KEY = os.getenv("API_KEY", "None")
BASE_URL = os.getenv("BASE_URL", None)
if BASE_URL == None:
    CLIENT = OpenAI(api_key=API_KEY)
else:
    CLIENT = OpenAI(base_url=f"{BASE_URL}", api_key=API_KEY)

MODEL_NAME = os.getenv("MODEL_NAME", None)
if MODEL_NAME == None:
    raise EnvironmentError("MODEL_NAME environment variable is not set")
STOP_WORDS="------"
MAX_NEW_TOKENS="None"
openai_model = OpenAIModel(MODEL_NAME, STOP_WORDS, MAX_NEW_TOKENS)
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

def extract_reasoning_and_score(text):
    """
    Extracts 'Reasoning' and 'Score' from the input text.
    Converts 'Score' to int and keeps 'Reasoning' as str.
    If extraction fails or format is incorrect, returns None.
    """
    try:
        # Use regular expressions to find 'Reasoning' and 'Score'
        reasoning_match = re.search(r'Reasoning\s*:\s*(.*)', text, re.IGNORECASE)
        score_match = re.search(r'Score\s*:\s*(.*)', text, re.IGNORECASE)

        # Check if both 'Reasoning' and 'Score' were found
        if not reasoning_match and not score_match:
            return None, None  # Return None to indicate format error

        # Extract the reasoning and score string
        reasoning = reasoning_match.group(1).strip()
        score_str = score_match.group(1).strip()

        # Convert 'Score' to int
        score = int(score_str)

        return reasoning, score

    except ValueError:
        # Score is not a valid integer
        return None, None  # Return None to indicate format error

    except Exception:
        # Catch any other exceptions
        return None, None  # Return None to indicate format error


def score_relevance(client, answer, question):
    """
    Scores the RAG's answer based on the Relevance criterion.
    """
    cur_prompt = f"""
You are to evaluate the following answer based on the Relevance criterion.

**Relevance Scoring Guide:**
Evaluates how well the answer addresses the question, ensuring content alignment.

- **1 (Very Low Relevance):**
  - The answer is completely unrelated to the question.
  - The content diverges from the topic without providing useful information.
- **2 (Low Relevance):**
  - The answer has weak relevance to the question, only partially addressing it.
  - It fails to directly answer the question or is vague.
- **3 (Moderate Relevance):**
  - The answer partially addresses the question but lacks completeness or accuracy.
  - It includes some off-topic content.
- **4 (High Relevance):**
  - The answer directly addresses and resolves the question.
  - The content is highly aligned with the question's topic.
- **5 (Very High Relevance):**
  - The answer fully and accurately responds to the question with comprehensive content.
  - It is entirely focused on the topic with no deviation.

**Instructions:**
Please provide a score from 1 to 5 according to the Relevance Scoring Guide above, and explain your reasoning.

------
You should output in the following format. Do not output any other content.

Reasoning: The reasoning for this score
Score: A num in [1,2,3,4,5]

------
You are to evaluate the following answer based on the Relevance criterion.

**Question:**
{question}

**RAG Answer:**
{answer}
------
Begin!

"""

    generator_response, _ = openai_model.generate(client, cur_prompt, TEMPERATURE)
    return generator_response


# def score_inferability(client, answer, clues, question):
#     """
#     Scores the RAG's answer based on the Inferability criterion.
#     """
#     cur_prompt = f"""
# You are to evaluate the following answer based on the Inferability criterion.

# **Inferability Scoring Guide:**
# Evaluates if the answer can be reasonably inferred from the provided clues and the value of each clue.

# - **1 (Very Low Inferability):**
#   - The answer cannot be derived from the clues provided.
#   - The answer is correct but does not utilize the clues.
# - **2 (Low Inferability):**
#   - The relationship between the answer and clues is weak, with unclear reasoning.
#   - Clues contribute little to the answer.
# - **3 (Moderate Inferability):**
#   - The answer can be partially inferred from clues but requires additional information.
#   - Clues have some influence but are not the main basis.
# - **4 (High Inferability):**
#   - The answer is primarily inferred from the clues with clear reasoning.
#   - Clues significantly contribute to the answer.
# - **5 (Very High Inferability):**
#   - The answer is entirely derived from the clues with a direct reasoning process.
#   - All clues are crucial for forming the answer.

# **Instructions:**
# Please provide a score from 1 to 5 according to the Inferability Scoring Guide above, and explain your reasoning.
# ------
# You should output in the following format. Do not output any other content.

# Reasoning: The reasoning for this score
# Score: A num in [1,2,3,4,5]

# ------
# You are to evaluate the following answer based on the Inferability criterion.

# **Question:**
# {question}

# **RAG Answer:**
# {answer}

# **Clues:**
# {clues}

# ------
# Begin!

# """

#     generator_response, _ = openai_model.generate(client, cur_prompt, TEMPERATURE)
#     return generator_response

# def score_practicality(client, answer, question):
#     """
#     Scores the question based on the Practicality criterion.
#     """
#     cur_prompt = f"""
# You are to evaluate the following question based on the Practicality criterion.

# **Practicality Scoring Guide:**
# Evaluates the real-world significance of the question.

# - **1 (Very Low Practicality):**
#   - The question is completely unrelated to the target audience's needs and interests, lacking any practical value.
#   - The question is overly abstract or outdated.
# - **2 (Low Practicality):**
#   - The question has weak relevance to the target audience's needs, with limited practical value.
#   - It may be too theoretical with little real-world application.
# - **3 (Moderate Practicality):**
#   - The question is somewhat relevant to the audience's needs but lacks prominence.
#   - It has some practical value but is too general.
# - **4 (High Practicality):**
#   - The question clearly aligns with common needs and interests of the target audience.
#   - It has practical applications and can engage the audience.
# - **5 (Very High Practicality):**
#   - The question is highly relevant to core needs and interests of the audience.
#   - It offers significant practical value and directly helps the audience.

# **Instructions:**
# Please provide a score from 1 to 5 according to the Practicality Scoring Guide above, and explain your reasoning.

# ------
# You should output in the following format. Do not output any other content.

# Reasoning: The reasoning for this score
# Score: A num in [1,2,3,4,5]

# ------
# You are to evaluate the following question based on the Practicality criterion.

# **Question:**
# {question}

# **RAG Answer:**
# {answer}
# ------
# Begin!

# """

#     generator_response, _ = openai_model.generate(client, cur_prompt, TEMPERATURE)
#     return generator_response

def score_semantic_similarity(client, clues, question):
    """
    Scores the semantic similarity between the question and the clues based on the Semantic Similarity criterion.
    """
    cur_prompt = f"""
You are to evaluate the semantic similarity between the following question and clues based on the Semantic Similarity criterion.

**Semantic Similarity Scoring Guide:**
Evaluates the degree of similarity in meaning or theme between the question and the clues.

- **1 (Very Low Similarity):**
  - The question and clues are completely unrelated in theme or meaning.
- **2 (Low Similarity):**
  - The question and clues have weak thematic connections.
- **3 (Moderate Similarity):**
  - The question and clues share some thematic elements, but the connection is average.
- **4 (High Similarity):**
  - The question and clues are highly related in theme and meaning.
- **5 (Very High Similarity):**
  - The question and clues are fully aligned in theme and meaning, directly corresponding.

**Instructions:**
Please provide a score from 1 to 5 according to the Semantic Similarity Scoring Guide above, and explain your reasoning.
------
You should output in the following format. Do not output any other content.

Reasoning: The reasoning for this score
Score: A num in [1,2,3,4,5]

------
You are to evaluate the semantic similarity between the following question and clues based on the Semantic Similarity criterion.

**Question:**
{question}

**Clues:**
{clues}

------
Begin!

"""

    generator_response, _ = openai_model.generate(client, cur_prompt, TEMPERATURE)
    return generator_response

def list_to_numbered_string(string_list):
    """
    Convert a list of strings into a numbered string.

    :param string_list: list of str, the list of strings to be converted
    :return: str, the resulting numbered string
    """
    numbered_string = ""
    for index, string in enumerate(string_list, start=1):
        numbered_string += f"{index}. {string}\n"
    return numbered_string.strip()

def expand_numbers_and_ranges(numbers_and_ranges):
    expanded_numbers = []
    for item in numbers_and_ranges:
        if '-' in item:  # It's a range like 'xx1-xx2'
            start, end = map(int, item.split('-'))
            if start > end:
                start, end = end, start
            expanded_numbers.extend(range(start, end + 1))
        else:  # It's a single number
            expanded_numbers.append(int(item))
    expanded_numbers = list(sorted(list(set(expanded_numbers))))
    return expanded_numbers

def process_file_content(input_path, output_path, save_interval, max_workers, answer_evaluator_max_gen_times):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        with open(input_path, 'r') as f:
            data = json.load(f)

    if answer_evaluator_max_gen_times == -1:
        answer_evaluator_max_gen_times = len(data)

    all_num, new_gen_num = 0, 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_data = {}
        for cur_dict in data[:answer_evaluator_max_gen_times]:
            
            """calculate generation metrics"""
            chunk_id = cur_dict['id'] # admission.stanford.edu.filter_index.htm.md_chunk_0
            objective_facts = cur_dict['objective-facts']
            objective_fact_id_2_objective_prompt = {idx: fact for idx, fact in enumerate(objective_facts, start=1)}

            if 'proposed-questions' not in cur_dict:
                continue
            proposed_questions = cur_dict['proposed-questions']
            if_already_generated = False
            for question_type, question_dict in proposed_questions.items():

                question = question_dict['question']

                if "objective-facts" in question_dict:
                    objective_fact_clue_ids = re.findall(r'\d+-\d+|\d+', question_dict['objective-facts'].strip())
                    objective_fact_clue_ids = expand_numbers_and_ranges(objective_fact_clue_ids)
                else:
                    objective_fact_clue_ids = []
                
                clues = [objective_fact_id_2_objective_prompt[int(clue_id)] for clue_id in objective_fact_clue_ids if int(clue_id) in objective_fact_id_2_objective_prompt]
                clues_str = list_to_numbered_string(clues)

                answer_keys = [key for key in question_dict.keys() if 'answer' in key and 'score' not in key and 'reason' not in key]

                for answer_key in answer_keys:
                    answer = question_dict[answer_key]

                    relevance_score_key_name = f"{answer_key}-relevance-score"

                    # Relevance Score
                    if relevance_score_key_name not in question_dict or question_dict[relevance_score_key_name] == None:
                        future = executor.submit(score_relevance, CLIENT, answer, question)
                        futures_to_data[future] = (question_dict, answer_key, 'relevance')

            if if_already_generated:
                continue

        all_num = len(futures_to_data)
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
            question_dict, answer_key, score_type = futures_to_data[future]
            try:
                score_response = future.result(timeout=10*60)
                reason, score = extract_reasoning_and_score(score_response)
                score_key_name = f"{answer_key}-{score_type}-score"
                reason_key_name = f"{answer_key}-{score_type}-reason"

                question_dict[reason_key_name] = reason
                question_dict[score_key_name] = score

                new_gen_num += 1
                if (new_gen_num + 1) % save_interval == 0:
                    print(f"Saving results to {output_path}")
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

            except Exception as e:
                print(f"Error processing {score_type} for answer_key {answer_key}: {e}")
                continue

    if new_gen_num or not os.path.exists(output_path):
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

    return new_gen_num, all_num

def process_file_entity_graph(input_path, output_path, save_interval, max_workers, answer_evaluator_max_gen_times):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        with open(input_path, 'r') as f:
            data = json.load(f)

    if answer_evaluator_max_gen_times == -1:
        answer_evaluator_max_gen_times = len(data)

    all_num, new_gen_num = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_data = {}

        for entity_id, entity_dict in list(data.items())[:answer_evaluator_max_gen_times]:
            
            proposed_questions = entity_dict['proposed-questions']
            
            objective_relationships = entity_dict['selected-relationships']['objective-relationships']
            objective_fact_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}
            
            """calculate generation metrics"""
            if_already_generated = False
            for question_type, question_dict in proposed_questions.items():
                question = question_dict['question']

                objective_relationship_ids = re.findall(r'\d+-\d+|\d+', question_dict['objective-relationship-id'].strip())
                objective_relationship_ids = expand_numbers_and_ranges(objective_relationship_ids)

                answer = question_dict['answer']
                
                # First, identify which relationships correspond to the correct answers, and then locate the relevant documents based on those relationships.
                real_related_relationships = [objective_fact_id_2_objective_relationship[int(clue_id)] for clue_id in objective_relationship_ids if int(clue_id) in objective_fact_id_2_objective_relationship]

                answer_keys = [key for key in question_dict.keys() if 'answer' in key and 'score' not in key and 'reason' not in key]
                for answer_key in answer_keys:
                    answer = question_dict[answer_key]

                    relevance_score_key_name = f"{answer_key}-relevance-score"

                    # Relevance Score
                    if relevance_score_key_name not in question_dict or question_dict[relevance_score_key_name] == None:
                        future = executor.submit(score_relevance, CLIENT, answer, question)
                        futures_to_data[future] = (question_dict, answer_key, 'relevance')

            if if_already_generated:
                continue

        all_num = len(futures_to_data)
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
            question_dict, answer_key, score_type = futures_to_data[future]
            try:
                score_response = future.result(timeout=10*60)
                reason, score = extract_reasoning_and_score(score_response)
                score_key_name = f"{answer_key}-{score_type}-score"
                reason_key_name = f"{answer_key}-{score_type}-reason"

                question_dict[reason_key_name] = reason
                question_dict[score_key_name] = score

                new_gen_num += 1
                if (new_gen_num + 1) % save_interval == 0:
                    print(f"Saving results to {output_path}")
                    with open(output_path, 'w') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

            except Exception as e:
                print(f"Error processing {score_type} for answer_key {answer_key}: {e}")
                continue
    if new_gen_num or not os.path.exists(output_path):
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")
    
    return new_gen_num, all_num
    
def answer_evaluator(input_path, output_path, save_interval, max_workers, answer_evaluator_max_gen_times):
    file_name = os.path.basename(input_path)
    relative_path = os.path.relpath(input_path, CUSTOM_CORPUS_HOME)
    print(f"Processing file {relative_path}")

    if "content" in file_name:
        return process_file_content(input_path, output_path, save_interval, max_workers, answer_evaluator_max_gen_times)
    elif "entity_graph" in file_name:
        return process_file_entity_graph(input_path, output_path, save_interval, max_workers, answer_evaluator_max_gen_times)
    else:
        raise ValueError(f"Unknown file type: {file_name}")
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Calculate metrics for answer.")
    parser.add_argument('--input_path', type=str, help="Input file containing query results.")
    parser.add_argument('--output_path', type=str, help="Output file to save the results.")
    parser.add_argument('--save_interval', type=int, help="The interval at which to save the results.")
    parser.add_argument('--max_workers', type=int, default=8, help="Maximum number of concurrent requests.")
    args = parser.parse_args()

    args.input_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.input_path))
    args.output_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.output_path))

    answer_evaluator(args.input_path, args.output_path, args.save_interval, args.max_workers)