import re
import os
import random
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import (
    OpenAIModel,
    CUSTOM_CORPUS_HOME,
    CLIENT,
    MODEL_NAME
)

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

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


def parse_transformations(text):
    # Regex pattern to match each transformation block
    pattern = re.compile(r'''
        ^\s*(\d+)\.\s*<transformed-action>(.*?)</transformed-action>\s*  # Capture the transformation type
        <transformed-explanation>(.*?)</transformed-explanation>\s*  # Capture the explanation
        <transformed-question>(.*?)</transformed-question>\s*        # Capture the result
        <transformed-answer>(.*?)</transformed-answer>               # Capture the answer
        ''', re.MULTILINE | re.DOTALL | re.VERBOSE)
    
    transformations = []
    
    # Find all matches in the text
    matches = pattern.findall(text)
    
    for match in matches:
        transformation = {
            'transformation': match[1].strip(),
            'explanation': match[2].strip(),
            'result': match[3].strip(),
            'answer': match[4].strip()
        }
        transformations.append(transformation)
    
    return transformations

class RephraseGeneratorHybrid:
    def __init__(self, save_interval=20):
        self.CLIENT = CLIENT
        self.MODEL_NAME = MODEL_NAME

        self.REPHRASE_GENERATOR_HYBRID_INPUT_PATH, self.REPHRASE_GENERATOR_HYBRID_PROMPT_PATH, self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH = None, None, None
        if os.getenv("REPHRASE_GENERATOR_HYBRID_CONTENT_INPUT_PATH", None) != None:
            self.REPHRASE_GENERATOR_HYBRID_INPUT_PATH = os.getenv("REPHRASE_GENERATOR_HYBRID_CONTENT_INPUT_PATH")
            self.REPHRASE_GENERATOR_HYBRID_PROMPT_PATH = os.getenv("REPHRASE_GENERATOR_HYBRID_CONTENT_PROMPT_PATH")
            self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH = os.getenv("REPHRASE_GENERATOR_HYBRID_CONTENT_OUTPUT_PATH")
            self.REPHRASE_GENERATOR_HYBRID_GENERATED_TYPE = 'content'
        elif os.getenv("REPHRASE_GENERATOR_HYBRID_ENTITYGRAPH_INPUT_PATH", None) != None:
            self.REPHRASE_GENERATOR_HYBRID_INPUT_PATH = os.getenv("REPHRASE_GENERATOR_HYBRID_ENTITYGRAPH_INPUT_PATH")
            self.REPHRASE_GENERATOR_HYBRID_PROMPT_PATH = os.getenv("REPHRASE_GENERATOR_HYBRID_ENTITYGRAPH_PROMPT_PATH")
            self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH = os.getenv("REPHRASE_GENERATOR_HYBRID_ENTITYGRAPH_OUTPUT_PATH")
            self.REPHRASE_GENERATOR_HYBRID_GENERATED_TYPE = 'entity_graph'
        else:
            raise EnvironmentError("Environment variable 'REPHRASE_GENERATOR_HYBRID_CONTENT_INPUT_PATH' or 'REPHRASE_GENERATOR_HYBRID_ENTITYGRAPH_INPUT_PATH' is not set.")
        
        self.REPHRASE_GENERATOR_HYBRID_INPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.REPHRASE_GENERATOR_HYBRID_INPUT_PATH)
        self.REPHRASE_GENERATOR_HYBRID_PROMPT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.REPHRASE_GENERATOR_HYBRID_PROMPT_PATH)
        self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH)

        self.REPHRASE_GENERATOR_STOP_WORDS = os.getenv("REPHRASE_GENERATOR_STOP_WORDS", None)
        self.REPHRASE_GENERATOR_MAX_NEW_TOKENS = os.getenv("REPHRASE_GENERATOR_MAX_NEW_TOKENS", None)
        self.REPHRASE_GENERATOR_NUM_WORKERS = int(os.getenv("REPHRASE_GENERATOR_NUM_WORKERS", 4))
        self.REPHRASE_GENERATOR_MAX_GEN_TIMES = int(os.getenv("REPHRASE_GENERATOR_MAX_GEN_TIMES", 300))

        if os.path.exists(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH):
            with open(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH, "r", encoding="utf-8") as f:
                self.inputs = json.load(f)
            print(f"Loaded rephrase generator {len(self.inputs)} examples from {os.path.relpath(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.")
        else:
            with open(self.REPHRASE_GENERATOR_HYBRID_INPUT_PATH, "r", encoding="utf-8") as f:
                self.inputs = json.load(f)
            print(f"Loaded rephrase generator {len(self.inputs)} examples from {os.path.relpath(self.REPHRASE_GENERATOR_HYBRID_INPUT_PATH, CUSTOM_CORPUS_HOME)}.")
        
        if self.REPHRASE_GENERATOR_MAX_GEN_TIMES == -1:
            self.REPHRASE_GENERATOR_MAX_GEN_TIMES = len(self.inputs)

        self.openai_model = OpenAIModel(MODEL_NAME, self.REPHRASE_GENERATOR_STOP_WORDS, self.REPHRASE_GENERATOR_MAX_NEW_TOKENS)
        self.save_interval = save_interval

    def run(self):

        with open(self.REPHRASE_GENERATOR_HYBRID_PROMPT_PATH, "r", encoding="utf-8") as f:
            rephrase_generator_prompt = f.read()
        print(f"Loaded rephrase generator prompt from {os.path.relpath(self.REPHRASE_GENERATOR_HYBRID_PROMPT_PATH, CUSTOM_CORPUS_HOME)}.")

        all_num, success_num = 0, 0
        tasks = []

        with ThreadPoolExecutor(max_workers=self.REPHRASE_GENERATOR_NUM_WORKERS) as executor:
            if self.REPHRASE_GENERATOR_HYBRID_GENERATED_TYPE in ['content']:
                for i, cur_input in enumerate(self.inputs[:self.REPHRASE_GENERATOR_MAX_GEN_TIMES]):

                    questions = cur_input['proposed-questions']
                    objective_facts = cur_input['objective-facts']

                    for proposed_question_type, proposed_question_dict in questions.items():
                        if 'rephrased-questions-hybrid' in proposed_question_dict and proposed_question_dict['rephrased-questions-hybrid']:
                            continue

                        needed_objective_fact_ids = proposed_question_dict['objective-facts']
                        needed_objective_fact_ids = re.findall(r'\d+-\d+|\d+', needed_objective_fact_ids)
                        needed_objective_fact_ids = expand_numbers_and_ranges(needed_objective_fact_ids)
                        needed_objective_factid_2_fact = {idx: objective_facts[idx-1] for idx in needed_objective_fact_ids if idx <= len(objective_facts)}

                        context = "Given clues:\n"
                        for idx, clue in needed_objective_factid_2_fact.items():
                            context += f"{idx}. {clue}\n"
                        context += "\n"
                        context += f"Original Question: {proposed_question_dict['question']}\n"
                        context += f"Answer: {proposed_question_dict['answer']}\n"
                        context += "\n"

                        cur_rephrase_generator_prompt = rephrase_generator_prompt.replace('[[CONTEXT]]', context)
                        future = executor.submit(self.openai_model.generate, self.CLIENT, cur_rephrase_generator_prompt, TEMPERATURE)
                        tasks.append((future, proposed_question_dict))

            elif self.REPHRASE_GENERATOR_HYBRID_GENERATED_TYPE in ['entity_graph']:
                for i, cur_input in list(self.inputs.items())[:self.REPHRASE_GENERATOR_MAX_GEN_TIMES]:

                    questions = cur_input['proposed-questions']
                    
                    objective_relationship_prompts = cur_input['selected-relationships']['objective-relationship-prompts']

                    for proposed_question_type, proposed_question_dict in questions.items():
                        if 'rephrased-questions-hybrid' in proposed_question_dict and proposed_question_dict['rephrased-questions-hybrid']:
                            continue

                        needed_objective_relationship_ids = proposed_question_dict['objective-relationship-id']
                        needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', needed_objective_relationship_ids)
                        needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                        needed_objective_relationship_id_2_prompt = {idx: objective_relationship_prompts[idx-1] for idx in needed_objective_relationship_ids if idx <= len(objective_relationship_prompts)}

                        context = "Given clues:\n"
                        for idx, clue in needed_objective_relationship_id_2_prompt.items():
                            context += f"{idx}. {clue}\n"
                        context += "\n"
                        context += f"Original Question: {proposed_question_dict['question']}\n"
                        context += f"Answer: {proposed_question_dict['answer']}\n"
                        context += "\n"

                        cur_rephrase_generator_prompt = rephrase_generator_prompt.replace('[[CONTEXT]]', context)
                        future = executor.submit(self.openai_model.generate, self.CLIENT, cur_rephrase_generator_prompt, TEMPERATURE)
                        tasks.append((future, proposed_question_dict))
            else:
                raise ValueError(f"Invalid value for 'REPHRASE_GENERATOR_HYBRID_GENERATED_TYPE': {self.REPHRASE_GENERATOR_HYBRID_GENERATED_TYPE}")

            all_num = len(tasks)
            for future_info in tqdm(as_completed([t[0] for t in tasks]), total=len(tasks), desc="Generating"):
                future = future_info
                idx = [t[0] for t in tasks].index(future)
                if idx == -1:
                    raise ValueError("Invalid index.")
                proposed_question_dict = tasks[idx][1]
                try:
                    rephrase_generator_response, _ = future.result(timeout=10*60)
                    rephrased_questions = parse_transformations(rephrase_generator_response)
                    if rephrased_questions:
                        proposed_question_dict['rephrased-questions-hybrid'] = rephrased_questions
                        success_num += 1
                        if success_num % self.save_interval == 0:
                            dir_path = os.path.dirname(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH)
                            os.makedirs(dir_path, exist_ok=True)
                            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
                            with open(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                                json.dump(self.inputs, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"Error processing question: {e}")

        if success_num or not os.path.exists(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH):
            dir_path = os.path.dirname(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH)
            os.makedirs(dir_path, exist_ok=True)
            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
            with open(self.REPHRASE_GENERATOR_HYBRID_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                json.dump(self.inputs, f, indent=2, ensure_ascii=False)
        
        return success_num, all_num