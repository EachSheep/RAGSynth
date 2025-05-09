import re
import os
import copy
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .. import (
    OpenAIModel,
    CUSTOM_CORPUS_HOME,
    CLIENT,
    MODEL_NAME
)

TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

def extract_objective_facts(text):
    """
    Extracts objective facts and their referenced sentence numbers.

    Parameters:
        text (str): The input text content.

    Returns:
        tuple: A tuple containing two lists.
            - objective_facts: A list of detailed descriptions of the objective facts.
            - sen_numbers: A list of sentence numbers as a formatted string corresponding to each objective fact.
    """
    # Regex pattern to match <detailed-desc> and <sentences-used> blocks
    pattern = r'<detailed-desc>(.*?)</detailed-desc>\s*<sentences-used>\[Sen\s*([^\]]+)\]</sentences-used>'
    
    # Use re.findall to extract all matches
    matches = re.findall(pattern, text, re.DOTALL)
    
    objective_facts = []
    sen_numbers = []

    for desc, sensors in matches:
        # Append detailed description to the objective_facts list
        objective_facts.append(desc.strip())
        
        # Extract all numbers using regex
        numbers = [int(num) for num in re.findall(r'\d+', sensors)]
        # Sort numbers to ensure the ranges are correctly identified
        numbers.sort()
        
        # Process the numbers to detect ranges
        formatted_sens = []
        i = 0
        while i < len(numbers):
            start = numbers[i]
            while i < len(numbers) - 1 and numbers[i] + 1 == numbers[i + 1]:
                i += 1
            end = numbers[i]
            if start == end:
                formatted_sens.append(f"{start}")
            else:
                formatted_sens.append(f"{start}-{end}")
            i += 1
        
        # Create the formatted string
        sen_string = f"{','.join(formatted_sens)}"
        sen_numbers.append(sen_string)
    
    return objective_facts, sen_numbers

class FactExtractor:
    def __init__(self, save_interval=20):
        self.CLIENT = CLIENT
        self.MODEL_NAME = MODEL_NAME

        self.FACT_EXTRACTOR_INPUT_PATH, self.FACT_EXTRACTOR_PROMPT_PATH, self.FACT_EXTRACTOR_OUTPUT_PATH = None, None, None
        if os.getenv("FACT_EXTRACTOR_INPUT_PATH", None) != None:
            self.FACT_EXTRACTOR_INPUT_PATH = os.getenv("FACT_EXTRACTOR_INPUT_PATH")
            self.FACT_EXTRACTOR_PROMPT_PATH = os.getenv("FACT_EXTRACTOR_PROMPT_PATH")
            self.FACT_EXTRACTOR_OUTPUT_PATH = os.getenv("FACT_EXTRACTOR_OUTPUT_PATH")
        else:
            raise EnvironmentError("Environment variable 'FACT_EXTRACTOR_INPUT_PATH' is not set.")

        self.FACT_EXTRACTOR_INPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.FACT_EXTRACTOR_INPUT_PATH)
        self.FACT_EXTRACTOR_PROMPT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.FACT_EXTRACTOR_PROMPT_PATH)
        self.FACT_EXTRACTOR_OUTPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.FACT_EXTRACTOR_OUTPUT_PATH)
        
        self.FACT_EXTRACTOR_STOP_WORDS = os.getenv("FACT_EXTRACTOR_STOP_WORDS", None)
        self.FACT_EXTRACTOR_NUM_WORKERS = int(os.getenv("FACT_EXTRACTOR_NUM_WORKERS", 4))
        FACT_EXTRACTOR_MAX_NEW_TOKENS = os.getenv("FACT_EXTRACTOR_MAX_NEW_TOKENS", None)

        self.openai_model = OpenAIModel(MODEL_NAME, self.FACT_EXTRACTOR_STOP_WORDS, FACT_EXTRACTOR_MAX_NEW_TOKENS)
        self.save_interval = save_interval

    def process_input(self, cur_input, fact_extractor_prompt, i):
        try:
            context = cur_input['context']
            cur_fact_extractor_prompt = fact_extractor_prompt.replace('[[CONTEXT]]', context)
            fact_extractor_response, _ = self.openai_model.generate(self.CLIENT, cur_fact_extractor_prompt, TEMPERATURE)
            objective_facts, sens = extract_objective_facts(fact_extractor_response)
            result = {
                **cur_input,
                'objective-facts': objective_facts,
                'sens': sens
            }
            return result, i
        except Exception as e:
            print(f"An error occurred while processing input {cur_input.get('id', 'unknown id')}: {e}")
            return None, None  # or you can return an error result

    def run(self):
        if os.path.exists(self.FACT_EXTRACTOR_OUTPUT_PATH):
            with open(self.FACT_EXTRACTOR_OUTPUT_PATH, "r", encoding="utf-8") as f:
                inputs = json.load(f)
            print(f"Loaded {len(inputs)} fact extractor examples from {os.path.relpath(self.FACT_EXTRACTOR_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.")
        else:
            with open(self.FACT_EXTRACTOR_INPUT_PATH, "r", encoding="utf-8") as f:
                inputs = json.load(f)
            print(f"Loaded {len(inputs)} fact extractor examples from {os.path.relpath(self.FACT_EXTRACTOR_INPUT_PATH, CUSTOM_CORPUS_HOME)}.")
        
        base_dir = os.path.dirname(self.FACT_EXTRACTOR_OUTPUT_PATH)
        os.makedirs(base_dir, exist_ok=True)
        chunk_output_path = os.path.join(base_dir, os.path.basename(self.FACT_EXTRACTOR_INPUT_PATH))
        if not os.path.exists(chunk_output_path):
            os.system(f"cp {self.FACT_EXTRACTOR_INPUT_PATH} {chunk_output_path}")
            print(f"Copied {os.path.relpath(self.FACT_EXTRACTOR_INPUT_PATH, CUSTOM_CORPUS_HOME)} to {os.path.relpath(chunk_output_path, CUSTOM_CORPUS_HOME)}.")

        with open(self.FACT_EXTRACTOR_PROMPT_PATH, "r", encoding="utf-8") as f:
            fact_extractor_prompt = f.read()
        print(f"Loaded fact extractor prompt from {os.path.relpath(self.FACT_EXTRACTOR_PROMPT_PATH, CUSTOM_CORPUS_HOME)}.")

        all_num, success_num = 0, 0
        with ThreadPoolExecutor(max_workers=self.FACT_EXTRACTOR_NUM_WORKERS) as executor:
            futures = []
            for i, cur_input in enumerate(inputs):
                if 'objective-facts' not in cur_input:
                    futures.append(executor.submit(self.process_input, cur_input, fact_extractor_prompt, i))

            all_num = len(futures)
            for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
                result, i = future.result(timeout=10*60)
                if result != None:
                    inputs[i] = result
                    success_num += 1
                    if success_num % self.save_interval == 0:
                        dir_path = os.path.dirname(self.FACT_EXTRACTOR_OUTPUT_PATH)
                        os.makedirs(dir_path, exist_ok=True)
                        print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.FACT_EXTRACTOR_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
                        with open(self.FACT_EXTRACTOR_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                            json.dump(inputs, f, indent=2, ensure_ascii=False)

        if success_num or not os.path.exists(self.FACT_EXTRACTOR_OUTPUT_PATH):
            dir_path = os.path.dirname(self.FACT_EXTRACTOR_OUTPUT_PATH)
            os.makedirs(dir_path, exist_ok=True)
            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.FACT_EXTRACTOR_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
            with open(self.FACT_EXTRACTOR_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                json.dump(inputs, f, indent=2, ensure_ascii=False)
            
        return success_num, all_num