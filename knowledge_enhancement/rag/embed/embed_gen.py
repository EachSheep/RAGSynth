import os
import re
import json
import argparse
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag.utils import (
    is_val_in_top_k,
    are_all_elements_in_list,
    expand_numbers_and_ranges,
    cal_percentage_of_elements_in_list
)
from rag.utils import extract_doc_to_sen
from rag.utils.request_openai_utils import OpenAIModel

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

API_KEY = os.getenv("API_KEY", "None")
MODEL_NAME = os.getenv("MODEL_NAME", None)
if MODEL_NAME == None:
    raise EnvironmentError("MODEL_NAME environment variable is not set")
BASE_URL = os.getenv("BASE_URL", None)
if BASE_URL == None:
    CLIENT = OpenAI(api_key=API_KEY)
else:
    CLIENT = OpenAI(base_url=f"{BASE_URL}", api_key=API_KEY)

openai_model = OpenAIModel(MODEL_NAME)

def dict_to_docided_string(string_dict):
    """
    Convert a list of strings into a docided string.

    :param string_list: list of str, the list of strings to be converted
    :return: str, the resulting numbered string
    """
    numbered_string = ""
    for index, (doc_id, doc_content) in enumerate(string_dict.items()):
        numbered_string += f"""{index}. <doc>
    <doc-name>{doc_id}</doc-name>
    <detailed-desc>{doc_content}</detailed-desc>
</doc>
"""
    return numbered_string.strip()

def extract_answer_and_reason(text):
    """
    Extracts the content between <answer></answer> and <reason></reason> tags from the provided text.

    Parameters:
    text (str): The text from which to extract the sections.

    Returns:
    tuple: A tuple containing the 'answer' and 'reason' as strings.
    """
    # Define regular expressions to match the <answer> and <reason> sections.
    answer_pattern = r"<answer>(.*?)</answer>"
    reason_pattern = r"<reason>(.*?)</reason>"

    # Search for the patterns in the text.
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    reason_match = re.search(reason_pattern, text, re.DOTALL)

    # Extract the matched text or return None if no match is found.
    answer = answer_match.group(1).strip() if answer_match else None
    reason = reason_match.group(1).strip() if reason_match else None

    if not answer and not reason:
        return text, None
    elif not answer:
        return reason, None
    elif not reason:
        return answer, None

    return answer, reason

def process_single_question(args):
    """
    Processes a single question dictionary and generates the answer and reason using the OpenAI model.

    Parameters:
    args (tuple): A tuple containing all necessary arguments:
        - proposed_question_dict (dict): The dictionary containing question information.
        - cur_prompt (str): The prompt.
        - dir_model_name (str): The name of the model.
        - openai_model (OpenAIModel): The OpenAIModel instance.

    Returns:
    dict: Updated proposed_question_dict with generated answer and reason.
    """
    (proposed_question_dict, cur_top_k_value, cur_prompt, dir_model_name, openai_model, max_tokens) = args

    answer_key_name = dir_model_name + "-top_k_value-" + str(cur_top_k_value) + "-answer"
    reason_key_name = dir_model_name + "-top_k_value-" + str(cur_top_k_value) + "-reason"
    if_already_generated = False
    if answer_key_name in proposed_question_dict and proposed_question_dict[answer_key_name]:
        if_already_generated = True
        return proposed_question_dict, if_already_generated  # Already generated

    # print("cur_prompt:", cur_prompt)
    # input("Press Enter to continue...")

    generator_response, _ = openai_model.generate(CLIENT, cur_prompt, max_tokens=max_tokens)
    answer, reason = extract_answer_and_reason(generator_response)
    proposed_question_dict[answer_key_name] = answer
    proposed_question_dict[reason_key_name] = reason

    return proposed_question_dict, if_already_generated

def gen_answer_for_top_k_documents(proposed_question_dict, prompt_template, question, corpusid_2_context, executor, futures_to_data, args):

    for cur_top_k_value in args.top_k_values:
        
        answer_key_name = args.dir_model_name + "-top_k_value-" + str(cur_top_k_value) + "-answer"
        reason_key_name = args.dir_model_name + "-top_k_value-" + str(cur_top_k_value) + "-reason"
        if answer_key_name in proposed_question_dict and proposed_question_dict[answer_key_name]:
            continue
        if 'top_k_documents' not in proposed_question_dict:
            continue
        top_k_documents = proposed_question_dict['top_k_documents']
        if not top_k_documents:
            continue
        top_k_corpusid_2_context = {doc: corpusid_2_context[doc] for doc in top_k_documents[:cur_top_k_value] if doc in corpusid_2_context}
        clue_str = dict_to_docided_string(top_k_corpusid_2_context)
        cur_prompt = prompt_template.replace('[[QUESTION]]', question)
        cur_prompt = cur_prompt.replace('[[CLUES]]', clue_str)
        
        future = executor.submit(process_single_question, (proposed_question_dict, cur_top_k_value, cur_prompt, args.dir_model_name, openai_model, args.max_tokens))
        futures_to_data[future] = proposed_question_dict

def process_file_content(args, data, corpusid_2_context, prompt_template):

    if args.max_process_num == -1:
        max_process_num = len(data)
    else:
        max_process_num = min(args.max_process_num, len(data))
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures_to_data = {}
        for cur_dict in tqdm(data[:max_process_num], desc="Processing Contents", total=max_process_num, dynamic_ncols=True):
            if 'proposed-questions' not in cur_dict:
                continue
            proposed_questions = cur_dict['proposed-questions']

            chunk_id = cur_dict['id'] # admission.stanford.edu.filter_index.htm.md_chunk_0
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                if 'top_k_documents' not in proposed_question_dict:
                    continue
                top_k_documents = proposed_question_dict['top_k_documents']
                if not top_k_documents:
                    continue

                tmp_result = is_val_in_top_k(top_k_documents, chunk_id, args.top_k_values)
                for cur_top_k_value in args.top_k_values:
                    all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                    proposed_question_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]
                
                original_question = proposed_question_dict['question']
                needed_corpusids = [chunk_id]
                needed_corpusid_2_context = {chunk_id: corpusid_2_context[chunk_id]}

                if not args.not_gen_for_original:
                    gen_answer_for_top_k_documents(proposed_question_dict, prompt_template, original_question, corpusid_2_context, executor, futures_to_data, args)
    

                rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions, start=1):
                    if "top_k_documents" not in rephrased_question_dict:
                        continue
                    tmp_result = is_val_in_top_k(rephrased_question_dict['top_k_documents'], chunk_id, args.top_k_values)
                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]

                tmp_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                rephrased_questions = []
                for only_gen_at_rephrased_pos in args.only_gen_at_rephrased_poses:
                    if only_gen_at_rephrased_pos < len(tmp_rephrased_questions):
                        rephrased_questions.append(tmp_rephrased_questions[only_gen_at_rephrased_pos])
                for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions, start=1):
                    if 'reordered-question' in rephrased_question_dict:
                        rephrased_question_str = rephrased_question_dict['reordered-question']
                    else:
                        rephrased_question_str = rephrased_question_dict['result']
                    gen_answer_for_top_k_documents(rephrased_question_dict, prompt_template, rephrased_question_str, corpusid_2_context, executor, futures_to_data, args)

                rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                for rephrased_question_part_type, rephrased_question_part_dict in enumerate(rephrased_questions_part, start=1):
                    if "top_k_documents" not in rephrased_question_part_dict:
                        continue
                    top_k_documents = rephrased_question_part_dict['top_k_documents']
                    if not top_k_documents:
                        continue
                    tmp_result = is_val_in_top_k(top_k_documents, chunk_id, args.top_k_values)
                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_part_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]

                tmp_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                rephrased_questions_part = []
                for only_gen_at_rephrased_pos_part in args.only_gen_at_rephrased_poses_part:
                    if only_gen_at_rephrased_pos_part < len(tmp_rephrased_questions_part):
                        rephrased_questions_part.append(tmp_rephrased_questions_part[only_gen_at_rephrased_pos_part])
                for rephrased_question_part_type, rephrased_question_part_dict in enumerate(rephrased_questions_part, start=1):
                    if 'reordered-question' in rephrased_question_part_dict:
                        rephrased_question_part_str = rephrased_question_part_dict['reordered-question']
                    else:
                        rephrased_question_part_str = rephrased_question_part_dict['result']
                    gen_answer_for_top_k_documents(rephrased_question_part_dict, prompt_template, rephrased_question_part_str, corpusid_2_context, executor, futures_to_data, args)
                
                rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid, start=1):
                    if "top_k_documents" not in rephrased_question_hybrid_dict:
                        continue
                    tmp_result = is_val_in_top_k(rephrased_question_hybrid_dict['top_k_documents'], chunk_id, args.top_k_values)
                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_hybrid_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]

                tmp_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                rephrased_questions_hybrid = []
                for only_gen_at_rephrased_pos_hybrid in args.only_gen_at_rephrased_poses_hybrid:
                    if only_gen_at_rephrased_pos_hybrid < len(tmp_rephrased_questions_hybrid):
                        rephrased_questions_hybrid.append(tmp_rephrased_questions_hybrid[only_gen_at_rephrased_pos_hybrid])
                for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid, start=1):
                    if 'reordered-question' in rephrased_question_hybrid_dict:
                        rephrased_question_hybrid_str = rephrased_question_hybrid_dict['reordered-question']
                    else:
                        rephrased_question_hybrid_str = rephrased_question_hybrid_dict['result']
                    gen_answer_for_top_k_documents(rephrased_question_hybrid_dict, prompt_template, rephrased_question_hybrid_str, corpusid_2_context, executor, futures_to_data, args)

        new_gen_num = 0
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Questions", dynamic_ncols=True):
            try:
                proposed_question_dict, if_already_generated = future.result(timeout=5*60)
                
                if not if_already_generated:
                    new_gen_num += 1
                    if new_gen_num % args.save_interval == 0:
                        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                        print(f"Saving results to {os.path.relpath(args.output_path, CUSTOM_CORPUS_HOME)}")
                        with open(args.output_path, 'w') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"Processed {new_gen_num} new items.")
            except Exception as e:
                print(f"Error processing item: {e}")

    if new_gen_num or not os.path.exists(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        print(f"Saving results to {os.path.relpath(args.output_path, CUSTOM_CORPUS_HOME)}")
        with open(args.output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {new_gen_num} new items.")


def process_file_entity_graph(args, data, corpusid_2_context, prompt_template):

    if args.max_process_num == -1:
        max_process_num = len(data)
    else:
        max_process_num = min(args.max_process_num, len(data))
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures_to_data = {}
                        
        for entity_dict in tqdm(list(data.values())[:max_process_num], desc="Processing Entities", total=max_process_num, dynamic_ncols=True):
            if 'proposed-questions' not in entity_dict:
                continue
            
            objective_relationships = entity_dict['selected-relationships']['objective-relationships']
            objective_relationship_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}

            proposed_questions = entity_dict['proposed-questions']
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                if 'top_k_documents' not in proposed_question_dict:
                    continue
                top_k_documents = proposed_question_dict['top_k_documents']
                if not top_k_documents:
                    continue

                # needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'].strip())
                # needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                # needed_related_relationships = [
                #     objective_relationship_id_2_objective_relationship[
                #         int(clue_id)] for clue_id in needed_objective_relationship_ids if clue_id and int(clue_id) in objective_relationship_id_2_objective_relationship
                #     ]
                # needed_corpusids = list(set([cur_related_relationship['id'] for cur_related_relationship in needed_related_relationships if 'id' in cur_related_relationship]))
                positive = proposed_question_dict["positive"]
                needed_corpusid_2_sens = extract_doc_to_sen(positive)
                needed_corpusids = list(needed_corpusid_2_sens.keys())
                if not needed_corpusids:
                    continue

                # precision
                for cur_top_k_value in args.top_k_values:
                    all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                    proposed_question_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                    percentage_of_elements_in_real_answer = cal_percentage_of_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                    proposed_question_dict[f"percentage_in_top_{cur_top_k_value}"] = percentage_of_elements_in_real_answer

                original_question = proposed_question_dict['question']
                if not args.not_gen_for_original:
                    gen_answer_for_top_k_documents(proposed_question_dict, prompt_template, original_question, corpusid_2_context, executor, futures_to_data, args)

                rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions, start=1):
                    if "top_k_documents" not in rephrased_question_dict:
                        continue
                    top_k_documents = proposed_question_dict['top_k_documents']
                    if not top_k_documents:
                        continue
                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])

                tmp_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                rephrased_questions = []
                for only_gen_at_rephrased_pos in args.only_gen_at_rephrased_poses:
                    if only_gen_at_rephrased_pos < len(tmp_rephrased_questions):
                        rephrased_questions.append(tmp_rephrased_questions[only_gen_at_rephrased_pos])
                for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions, start=1):
                    if 'reordered-question' in rephrased_question_dict:
                        rephrased_question_str = rephrased_question_dict['reordered-question']
                    else:
                        rephrased_question_str = rephrased_question_dict['result']
                    gen_answer_for_top_k_documents(rephrased_question_dict, prompt_template, rephrased_question_str, corpusid_2_context, executor, futures_to_data, args)
                
                rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                for rephrased_question_part_type, rephrased_question_part_dict in enumerate(rephrased_questions_part, start=1):
                    if "top_k_documents" not in rephrased_question_part_dict:
                        continue
                    top_k_documents = proposed_question_dict['top_k_documents']
                    if not top_k_documents:
                        continue
                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_part_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])

                tmp_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                rephrased_questions_part = []
                for only_gen_at_rephrased_pos_part in args.only_gen_at_rephrased_poses_part:
                    if only_gen_at_rephrased_pos_part < len(tmp_rephrased_questions_part):
                        rephrased_questions_part.append(tmp_rephrased_questions_part[only_gen_at_rephrased_pos_part])
                for rephrased_question_part_type, rephrased_question_part_dict in enumerate(rephrased_questions_part, start=1):
                    if 'reordered-question' in rephrased_question_part_dict:
                        rephrased_question_part_str = rephrased_question_part_dict['reordered-question']
                    else:
                        rephrased_question_part_str = rephrased_question_part_dict['result']
                    gen_answer_for_top_k_documents(rephrased_question_part_dict, prompt_template, rephrased_question_part_str, corpusid_2_context, executor, futures_to_data, args)
                
                rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid, start=1):
                    if "top_k_documents" not in rephrased_question_hybrid_dict:
                        continue
                    top_k_documents = proposed_question_dict['top_k_documents']
                    if not top_k_documents:
                        continue
                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_hybrid_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])

                tmp_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                rephrased_questions_hybrid = []
                for only_gen_at_rephrased_pos_hybrid in args.only_gen_at_rephrased_poses_hybrid:
                    if only_gen_at_rephrased_pos_hybrid < len(tmp_rephrased_questions_hybrid):
                        rephrased_questions_hybrid.append(tmp_rephrased_questions_hybrid[only_gen_at_rephrased_pos_hybrid])
                for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid, start=1):
                    if 'reordered-question' in rephrased_question_hybrid_dict:
                        rephrased_question_hybrid_str = rephrased_question_hybrid_dict['reordered-question']
                    else:
                        rephrased_question_hybrid_str = rephrased_question_hybrid_dict['result']
                    gen_answer_for_top_k_documents(rephrased_question_hybrid_dict, prompt_template, rephrased_question_hybrid_str, corpusid_2_context, executor, futures_to_data, args)
        
        new_gen_num = 0
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Questions", dynamic_ncols=True):
            try:
                proposed_question_dict, if_already_generated = future.result(timeout=5*60)
                if not if_already_generated:
                    new_gen_num += 1
                    if new_gen_num % args.save_interval == 0:
                        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                        print(f"Saving results to {os.path.relpath(args.output_path, CUSTOM_CORPUS_HOME)}")
                        with open(args.output_path, 'w') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"Processed {new_gen_num} new items.")
            except Exception as e:
                print(f"Error processing item: {e}")
        
    if new_gen_num or not os.path.exists(args.output_path):
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        print(f"Saving results to {os.path.relpath(args.output_path, CUSTOM_CORPUS_HOME)}")
        with open(args.output_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {new_gen_num} new items.")

def main(args):

    relpath = os.path.relpath(args.input_path, CUSTOM_CORPUS_HOME)
    print(f"Processing file {relpath}")

    if os.path.exists(args.output_path):
        with open(args.output_path, 'r') as f:
            data = json.load(f)
    else:
        with open(args.input_path, 'r') as f:
            data = json.load(f)
    
    with open(args.chunk_path, 'r') as f:
        chunks_data = json.load(f)
    corpusid_2_context = {cur_dict['id']: cur_dict['context'] for cur_dict in chunks_data}

    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()
    if args.question_type in ["content"]:
        process_file_content(args, data, corpusid_2_context, prompt_template)
    elif args.question_type in ["entity_graph"]:
        process_file_entity_graph(args, data, corpusid_2_context, prompt_template)
    else:
        raise ValueError(f"Unknown question type: {args.question_type}")

if __name__ == '__main__':
    print("-" * 50)
    parser = argparse.ArgumentParser(description="RAG generation using General RAG Models.")
    parser.add_argument('--not_gen_for_original', action='store_true', help="Generate for original questions.")
    parser.add_argument('--only_gen_at_rephrased_poses', type=int, nargs='+', default=[], help="List of rephrased positions to evaluate.")
    parser.add_argument('--only_gen_at_rephrased_poses_part', type=int, nargs='+', default=[], help="List of rephrased positions part to evaluate.")
    parser.add_argument('--only_gen_at_rephrased_poses_hybrid', type=int, nargs='+', default=[6], help="List of rephrased positions hybrid to evaluate.")
    parser.add_argument('--question_type', type=str, choices=["content", "entity_graph"], help="Type of question to process.")
    parser.add_argument('--chunk_path', type=str, default=None, help="Path to the chunk file.")
    parser.add_argument('--input_path', type=str, help="Input file path.")
    parser.add_argument('--output_path', type=str, help="Path to save the result.")
    parser.add_argument('--top_k_values', type=int, nargs='+', default=[3, 5], help="List of top_k values for which precision will be calculated.")
    parser.add_argument('--prompt_path', type=str, help="Path to the prompt file.")
    parser.add_argument('--save_interval', type=int, default=100, help="The interval at which to save the results.")
    parser.add_argument('--max_workers', type=int, default=4, help="Maximum number of parallel workers.")
    parser.add_argument('--max_process_num', type=int, default=-1, help="Maximum number of process data.")
    parser.add_argument('--dir_model_name', type=str, required=True, help="The model name.")
    parser.add_argument('--max_tokens', type=int, default=None, help="The model name.")
    
    args = parser.parse_args()

    args.chunk_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.chunk_path))
    args.input_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.input_path))
    args.output_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.output_path))
    args.prompt_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.prompt_path))

    # cp file at chunk_path to output_path
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    chunk_output_path = os.path.join(output_dir, os.path.basename(args.chunk_path))
    if not os.path.exists(chunk_output_path):
        os.system(f"cp {args.chunk_path} {chunk_output_path}")

    main(args)
    print("-" * 50)