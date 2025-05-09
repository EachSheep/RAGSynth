import os
import re
import json
import requests
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
import numpy as np

from .. import (
    OpenAIModel,
    CUSTOM_CORPUS_HOME,
    CLIENT,
    MODEL_NAME
)

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

class QuestionVerifier():
    def __init__(self, save_interval):
        self.CLIENT = CLIENT
        self.MODEL_NAME = MODEL_NAME

        self.QUESTION_VERIFIER_INPUT_PATH, self.QUESTION_VERIFIER_PROMPT_PATH, self.QUESTION_VERIFIER_OUTPUT_PATH = None, None, None
        if os.getenv("QUESTION_VERIFIER_CONTENT_INPUT_PATH", None) != None:
            self.QUESTION_VERIFIER_INPUT_PATH = os.getenv("QUESTION_VERIFIER_CONTENT_INPUT_PATH")
            self.QUESTION_VERIFIER_PROMPT_PATH = os.getenv("QUESTION_VERIFIER_CONTENT_PROMPT_PATH")
            self.QUESTION_VERIFIER_OUTPUT_PATH = os.getenv("QUESTION_VERIFIER_CONTENT_OUTPUT_PATH")
            self.QUESTION_VERIFIER_GENERATED_TYPE = 'content'
        elif os.getenv("QUESTION_VERIFIER_ENTITYGRAPH_INPUT_PATH", None) != None:
            self.QUESTION_VERIFIER_INPUT_PATH = os.getenv("QUESTION_VERIFIER_ENTITYGRAPH_INPUT_PATH")
            self.QUESTION_VERIFIER_PROMPT_PATH = os.getenv("QUESTION_VERIFIER_ENTITYGRAPH_PROMPT_PATH")
            self.QUESTION_VERIFIER_OUTPUT_PATH = os.getenv("QUESTION_VERIFIER_ENTITYGRAPH_OUTPUT_PATH")
            self.QUESTION_VERIFIER_GENERATED_TYPE = 'entity_graph'
        else:
            raise EnvironmentError("Environment variable 'QUESTION_VERIFIER_CONTENT_INPUT_PATH' or 'QUESTION_VERIFIER_ENTITYGRAPH_INPUT_PATH' is not set.")

        self.QUESTION_VERIFIER_INPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.QUESTION_VERIFIER_INPUT_PATH)
        self.QUESTION_VERIFIER_PROMPT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.QUESTION_VERIFIER_PROMPT_PATH)
        self.QUESTION_VERIFIER_OUTPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.QUESTION_VERIFIER_OUTPUT_PATH)

        self.QUESTION_VERIFIER_MAX_GEN_TIMES = int(os.getenv("QUESTION_VERIFIER_MAX_GEN_TIMES", 300))

        if os.path.exists(self.QUESTION_VERIFIER_OUTPUT_PATH):
            with open(self.QUESTION_VERIFIER_OUTPUT_PATH, "r", encoding="utf-8") as f:
                self.inputs = json.load(f)
            print(f"Loaded rephrase generator {len(self.inputs)} examples from {os.path.relpath(self.QUESTION_VERIFIER_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.")
        else:
            with open(self.QUESTION_VERIFIER_INPUT_PATH, "r", encoding="utf-8") as f:
                self.inputs = json.load(f)
            print(f"Loaded rephrase generator {len(self.inputs)} examples from {os.path.relpath(self.QUESTION_VERIFIER_INPUT_PATH, CUSTOM_CORPUS_HOME)}.")

        if self.QUESTION_VERIFIER_MAX_GEN_TIMES == -1:
            self.QUESTION_VERIFIER_MAX_GEN_TIMES = len(self.inputs)

        self.LOGITS_SERVER_ADDRESS = os.getenv("LOGITS_SERVER_ADDRESS", None)
        if self.LOGITS_SERVER_ADDRESS == None:
            raise ValueError("LOGITS_SERVER_ADDRESS environment variable is not set")
        
        self.LOGITS_MODEL_NAME = os.getenv("LOGITS_MODEL_NAME", None)
        if self.LOGITS_MODEL_NAME == None:
            raise ValueError("LOGITS_MODEL_NAME environment variable is not set")

        self.save_interval = save_interval
    
    def run(self):
        with open(self.QUESTION_VERIFIER_PROMPT_PATH, "r", encoding="utf-8") as f:
            question_verifier_prompt = f.read()
        print(f"Loaded question verifier prompt from {os.path.relpath(self.QUESTION_VERIFIER_PROMPT_PATH, CUSTOM_CORPUS_HOME)}.")

        if self.QUESTION_VERIFIER_GENERATED_TYPE in ['content']:
            
            success_num = 0
            for i, cur_input in tqdm(enumerate(self.inputs), desc="Processing", total=len(self.inputs), dynamic_ncols=True):

                proposed_questions = cur_input['proposed-questions']
                objective_facts = cur_input['objective-facts']
                
                clue_seqs = {}
                if_already_processed = True
                for proposed_question_type, proposed_question_dict in proposed_questions.items():
                    base_prob_key = self.LOGITS_MODEL_NAME + '-base-prob'
                    if base_prob_key not in proposed_question_dict or proposed_question_dict[base_prob_key] == None:
                        if_already_processed = False

                if if_already_processed:
                    continue

                for proposed_question_type, proposed_question_dict in proposed_questions.items():
                    question_clue_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-facts'].strip())
                    question_clue_ids = expand_numbers_and_ranges(question_clue_ids)
                    needed_clues = [objective_facts[int(clue_id)-1] for clue_id in question_clue_ids]

                    base_prompt = "Given clues:\n"
                    for idx, cur_clue in enumerate(needed_clues, start=1):
                        base_prompt += f"{idx}. {cur_clue}\n"
                    base_prompt += "\n"
                    base_prompt += f"Question: {proposed_question_dict['question']}\n"
                    base_prompt = question_verifier_prompt.replace("[[CONTEXT]]", base_prompt)

                    prompts = []
                    for clue_to_remove in range(len(needed_clues)):
                        modified_prompt = "Given clues:\n"
                        meet_or_not = False
                        for idx, cur_clue in enumerate(needed_clues, start=1):
                            if not meet_or_not and idx - 1 != clue_to_remove:  # Skip the clue to be removed
                                modified_prompt += f"{idx}. {cur_clue}\n"
                            elif meet_or_not and idx - 1 != clue_to_remove:
                                modified_prompt += f"{idx - 1}. {cur_clue}\n"
                            elif idx - 1 == clue_to_remove:
                                meet_or_not = True
                        modified_prompt += "\n"
                        modified_prompt += f"Question: {proposed_question_dict['question']}\n"
                        modified_prompt = question_verifier_prompt.replace("[[CONTEXT]]", modified_prompt)
                        prompts.append(modified_prompt)
                    
                    base_prob_key = self.LOGITS_MODEL_NAME + '-base-prob'
                    new_probs_key = self.LOGITS_MODEL_NAME + '-new-probs'
                    try:
                        base_prob, new_probs, differences = self.evaluate_prompts(base_prompt, prompts, proposed_question_dict['answer'], method='kl')
                        proposed_question_dict[base_prob_key] = base_prob
                        proposed_question_dict[new_probs_key] = new_probs
                        clue_seqs[proposed_question_type] = proposed_question_dict
                    except Exception as e:
                        print(f"Error: {e}")
                        proposed_question_dict[base_prob_key] = None
                        proposed_question_dict[new_probs_key] = None
                        clue_seqs[proposed_question_type] = proposed_question_dict
                        continue

                self.inputs[i]['proposed-questions'] = clue_seqs
                success_num += 1

                if success_num % self.save_interval == 0:
                    print(f'Saving {success_num} outputs to {os.path.relpath(self.QUESTION_VERIFIER_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
                    with open(self.QUESTION_VERIFIER_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                        json.dump(self.inputs, f, indent=2, ensure_ascii=False)

            if success_num or not os.path.exists(self.QUESTION_VERIFIER_OUTPUT_PATH):
                print(f'Saving {success_num} outputs to {os.path.relpath(self.QUESTION_VERIFIER_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
                with open(self.QUESTION_VERIFIER_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                    json.dump(self.inputs, f, indent=2, ensure_ascii=False)

        elif self.QUESTION_VERIFIER_GENERATED_TYPE in ['entity_graph']:

            success_num = 0
            for cur_entity_id, cur_entity_item in tqdm(self.inputs.items(), desc="Processing", total=len(self.inputs), dynamic_ncols=True):
                proposed_questions = cur_entity_item['proposed-questions']
                objective_relationship_prompts = cur_entity_item['selected-relationships']['objective-relationship-prompts']
                
                clue_seqs = {}
                if_already_processed = True
                for proposed_question_type, proposed_question_dict in proposed_questions.items():
                    base_prob_key = self.LOGITS_MODEL_NAME + '-base-prob'
                    new_probs_key = self.LOGITS_MODEL_NAME + '-new-probs'
                    if base_prob_key not in proposed_question_dict or proposed_question_dict[base_prob_key] == None:
                        if_already_processed = False
                
                if if_already_processed:
                    continue

                for proposed_question_type, proposed_question_dict in proposed_questions.items():
                    objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'].strip())
                    objective_relationship_ids = expand_numbers_and_ranges(objective_relationship_ids)
                    needed_clues = [objective_relationship_prompts[int(clue_id)-1] for clue_id in objective_relationship_ids]
                    
                    base_prompt = "Given clues:\n"
                    for idx, cur_clue in enumerate(needed_clues, start=1):
                        base_prompt += f"{idx}. {cur_clue}\n"
                    base_prompt += "\n"
                    base_prompt += f"Question: {proposed_question_dict['question']}\n"
                    base_prompt = question_verifier_prompt.replace("[[CONTEXT]]", base_prompt)

                    prompts = []
                    for clue_to_remove in range(len(needed_clues)):
                        modified_prompt = "Given clues:\n"
                        meet_or_not = False
                        for idx, cur_clue in enumerate(needed_clues, start=1):
                            if not meet_or_not and idx - 1 != clue_to_remove:
                                modified_prompt += f"{idx}. {cur_clue}\n"
                            elif meet_or_not and idx - 1 != clue_to_remove:
                                modified_prompt += f"{idx - 1}. {cur_clue}\n"
                            elif idx - 1 == clue_to_remove:
                                meet_or_not = True
                        modified_prompt += "\n"
                        modified_prompt += f"Question: {proposed_question_dict['question']}\n"
                        modified_prompt = question_verifier_prompt.replace("[[CONTEXT]]", modified_prompt)
                        prompts.append(modified_prompt)
                    base_prob_key = self.LOGITS_MODEL_NAME + '-base-prob'
                    new_probs_key = self.LOGITS_MODEL_NAME + '-new-probs'
                    try:    
                        base_prob, new_probs, differences = self.evaluate_prompts(base_prompt, prompts, proposed_question_dict['answer'], method='kl')
                        proposed_question_dict[base_prob_key] = base_prob
                        proposed_question_dict[new_probs_key] = new_probs
                        clue_seqs[proposed_question_type] = proposed_question_dict
                    except Exception as e:
                        print(f"Error: {e}")
                        proposed_question_dict[base_prob_key] = None
                        proposed_question_dict[new_probs_key] = None
                        clue_seqs[proposed_question_type] = proposed_question_dict
                        continue

                self.inputs[cur_entity_id]['proposed-questions'] = clue_seqs
                success_num += 1
                
                if success_num % self.save_interval == 0:
                    print(f'Saving {success_num} outputs to {os.path.relpath(self.QUESTION_VERIFIER_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
                    with open(self.QUESTION_VERIFIER_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                        json.dump(self.inputs, f, indent=2, ensure_ascii=False)
            
            if success_num or not os.path.exists(self.QUESTION_VERIFIER_OUTPUT_PATH):
                print(f'Saving {success_num} outputs to {os.path.relpath(self.QUESTION_VERIFIER_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
                with open(self.QUESTION_VERIFIER_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                    json.dump(self.inputs, f, indent=2, ensure_ascii=False)

    def _get_logprobs(self, prompts, answers):
        payload = {
            "prompts": prompts,
            "answers": answers
        }
        
        try:
            response = requests.post(self.LOGITS_SERVER_ADDRESS, json=payload)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()
            
            return data["probabilities"], data["answer_input_ids"]
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")


    def _calculate_distribution_difference(self, base_probs, new_probs, method='js'):
        if method == 'js':
            return jensenshannon(base_probs, new_probs)
        elif method == 'kl':
            # Ensure no zero probabilities for KL divergence
            base_probs += 1e-10
            new_probs += 1e-10
            return np.sum(kl_div(base_probs, new_probs))
        else:
            raise ValueError("Unsupported method. Use 'js' or 'kl'.")

    def evaluate_prompts(self, base_prompt, prompts, answer, method='js'):
        base_prob, base_answer_input_ids = self._get_logprobs([base_prompt], [answer])
        base_prob_array = np.array(base_prob[0])
        
        new_probs, new_answer_input_ids = self._get_logprobs(prompts, [answer] * len(prompts))
        differences = []
        for new_prob, new_answer_input_id in zip(new_probs, new_answer_input_ids):
            new_prob = np.array(new_prob)
            new_answer_input_id = np.array(new_answer_input_id)
            difference = self._calculate_distribution_difference(base_prob_array, new_prob, method)
            differences.append(difference)
        
        return base_prob[0], new_probs, differences
