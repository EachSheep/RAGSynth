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
    """
    Parses the transformed questions and their metadata from the provided text.

    Args:
        text (str): The input text containing the transformed questions.

    Returns:
        list of dict: A list of dictionaries, each containing the question number,
                      transformed question, and the same-meaning-with-origin value.
    """
    # Regex pattern to match each transformation block
    pattern = re.compile(r'''
        \s*<question-\d+>                               # Match the question number and opening tag
        \s*<transformed-question>(.*?)</transformed-question> # Capture the transformed question
        \s*<same-meaning-with-origin>(True|False)</same-meaning-with-origin> # Capture the boolean value
        \s*</question-\d+>                                   # Match the closing tag
        ''', re.DOTALL | re.VERBOSE)

    transformations = []

    # Find all matches in the text
    for match in pattern.finditer(text):
        transformed_question = match.group(1).strip()
        same_meaning = match.group(2).strip() == 'True'
        
        # Extract the question number from the matched string
        question_tag = re.search(r'<question-(\d+)>', match.group(0))
        question_number = int(question_tag.group(1)) if question_tag else None

        transformations.append({
            'question_number': question_number,
            'transformed_question': transformed_question,
            'same_meaning_with_origin': same_meaning
        })

    return transformations

class SentenceOrderChanger:
    def __init__(self, save_interval=20):
        self.CLIENT = CLIENT
        self.MODEL_NAME = MODEL_NAME

        self.SENTENCE_ORDER_CHANGER_INPUT_PATH, self.SENTENCE_ORDER_CHANGER_PROMPT_PATH, self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH = None, None, None
        if os.getenv("SENTENCE_ORDER_CHANGER_CONTENT_INPUT_PATH", None) != None:
            self.SENTENCE_ORDER_CHANGER_INPUT_PATH = os.getenv("SENTENCE_ORDER_CHANGER_CONTENT_INPUT_PATH")
            self.SENTENCE_ORDER_CHANGER_PROMPT_PATH = os.getenv("SENTENCE_ORDER_CHANGER_CONTENT_PROMPT_PATH")
            self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH = os.getenv("SENTENCE_ORDER_CHANGER_CONTENT_OUTPUT_PATH")
            self.SENTENCE_ORDER_CHANGER_GENERATED_TYPE = 'content'
        elif os.getenv("SENTENCE_ORDER_CHANGER_ENTITYGRAPH_INPUT_PATH", None) != None:
            self.SENTENCE_ORDER_CHANGER_INPUT_PATH = os.getenv("SENTENCE_ORDER_CHANGER_ENTITYGRAPH_INPUT_PATH")
            self.SENTENCE_ORDER_CHANGER_PROMPT_PATH = os.getenv("SENTENCE_ORDER_CHANGER_ENTITYGRAPH_PROMPT_PATH")
            self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH = os.getenv("SENTENCE_ORDER_CHANGER_ENTITYGRAPH_OUTPUT_PATH")
            self.SENTENCE_ORDER_CHANGER_GENERATED_TYPE = 'entity_graph'
        else:
            raise EnvironmentError("Environment variable 'SENTENCE_ORDER_CHANGER_CONTENT_INPUT_PATH' or 'SENTENCE_ORDER_CHANGER_ENTITYGRAPH_INPUT_PATH' is not set.")
        
        self.SENTENCE_ORDER_CHANGER_INPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.SENTENCE_ORDER_CHANGER_INPUT_PATH)
        self.SENTENCE_ORDER_CHANGER_PROMPT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.SENTENCE_ORDER_CHANGER_PROMPT_PATH)
        self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH)

        self.SENTENCE_ORDER_CHANGER_STOP_WORDS = os.getenv("SENTENCE_ORDER_CHANGER_STOP_WORDS", None)
        self.SENTENCE_ORDER_CHANGER_MAX_NEW_TOKENS = os.getenv("SENTENCE_ORDER_CHANGER_MAX_NEW_TOKENS", None)
        self.SENTENCE_ORDER_CHANGER_NUM_WORKERS = int(os.getenv("SENTENCE_ORDER_CHANGER_NUM_WORKERS", 4))
        self.SENTENCE_ORDER_CHANGER_MAX_GEN_TIMES = int(os.getenv("SENTENCE_ORDER_CHANGER_MAX_GEN_TIMES", 300))

        if os.path.exists(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH):
            with open(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH, "r", encoding="utf-8") as f:
                self.inputs = json.load(f)
            print(f"Loaded sentence order changer {len(self.inputs)} examples from {os.path.relpath(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.")
        else:
            with open(self.SENTENCE_ORDER_CHANGER_INPUT_PATH, "r", encoding="utf-8") as f:
                self.inputs = json.load(f)
            print(f"Loaded sentence order changer {len(self.inputs)} examples from {os.path.relpath(self.SENTENCE_ORDER_CHANGER_INPUT_PATH, CUSTOM_CORPUS_HOME)}.")

        if self.SENTENCE_ORDER_CHANGER_MAX_GEN_TIMES == -1:
            self.SENTENCE_ORDER_CHANGER_MAX_GEN_TIMES = len(self.inputs)

        self.openai_model = OpenAIModel(MODEL_NAME, self.SENTENCE_ORDER_CHANGER_STOP_WORDS, self.SENTENCE_ORDER_CHANGER_MAX_NEW_TOKENS)
        self.save_interval = save_interval

    def run(self):

        with open(self.SENTENCE_ORDER_CHANGER_PROMPT_PATH, "r", encoding="utf-8") as f:
            sentence_changer_prompt = f.read()
        print(f"Loaded sentence order changer prompt from {os.path.relpath(self.SENTENCE_ORDER_CHANGER_PROMPT_PATH, CUSTOM_CORPUS_HOME)}.")

        all_num, success_num = 0, 0
        tasks = []

        with ThreadPoolExecutor(max_workers=self.SENTENCE_ORDER_CHANGER_NUM_WORKERS) as executor:
            if self.SENTENCE_ORDER_CHANGER_GENERATED_TYPE in ['content']:
                for i, cur_input in enumerate(self.inputs[:self.SENTENCE_ORDER_CHANGER_MAX_GEN_TIMES]):

                    questions = cur_input['proposed-questions']
                    objective_facts = cur_input['objective-facts']

                    for proposed_question_type, proposed_question_dict in questions.items():

                        context = "Given clues:\n"
                        for idx, clue in enumerate(objective_facts, start=1):
                            context += f"{idx}. {clue}\n"
                        context += "\n"
                        context += f"Questions and Answers: \n"
                        if 'rephrased-questions-part' in proposed_question_dict:
                            rephrased_questions_part = proposed_question_dict['rephrased-questions-part']
                            already_processed = False
                            for j, cur_rephrased_question_part in enumerate(rephrased_questions_part, start=1):
                                if 'reordered-question' in cur_rephrased_question_part:
                                    already_processed = True
                                    break
                                context += f"<question-{j}>{cur_rephrased_question_part['result']}</question-{j}>\n"
                                context += f"<answer-{j}>{cur_rephrased_question_part['answer']}</answer-{j}>\n"
                            if not already_processed:
                                context += "\n"
                                cur_sentence_changer_prompt = sentence_changer_prompt.replace('[[CONTEXT]]', context)
                                future = executor.submit(self.openai_model.generate, self.CLIENT, cur_sentence_changer_prompt, TEMPERATURE)
                                tasks.append((future, rephrased_questions_part))

                        context = "Given clues:\n"
                        for idx, clue in enumerate(objective_facts, start=1):
                            context += f"{idx}. {clue}\n"
                        context += "\n"
                        context += f"Questions and Answers: \n"
                        if 'rephrased-questions-hybrid' in proposed_question_dict:
                            rephrased_questions_hybrid = proposed_question_dict['rephrased-questions-hybrid']
                            already_processed = False
                            for j, cur_rephrased_question_hybrid in enumerate(rephrased_questions_hybrid, start=1):
                                if 'reordered-question' in cur_rephrased_question_hybrid:
                                    already_processed = True
                                    break
                                context += f"<question-{j}>{cur_rephrased_question_hybrid['result']}</question-{j}>\n"
                                context += f"<answer-{j}>{cur_rephrased_question_hybrid['answer']}</answer-{j}>\n"
                            if not already_processed:
                                context += "\n"
                                cur_sentence_changer_prompt = sentence_changer_prompt.replace('[[CONTEXT]]', context)
                                future = executor.submit(self.openai_model.generate, self.CLIENT, cur_sentence_changer_prompt, TEMPERATURE)
                                tasks.append((future, rephrased_questions_hybrid))

            elif self.SENTENCE_ORDER_CHANGER_GENERATED_TYPE in ['entity_graph']:
                for i, cur_input in tqdm(list(self.inputs.items())[:self.SENTENCE_ORDER_CHANGER_MAX_GEN_TIMES], desc="Processing", total=len(self.inputs), dynamic_ncols=True):

                    questions = cur_input['proposed-questions']

                    objective_relationship_prompts = cur_input['selected-relationships']['objective-relationship-prompts']

                    for proposed_question_type, proposed_question_dict in questions.items():

                        context = "Given clues:\n"
                        for idx, clue in enumerate(objective_relationship_prompts, start=1):
                            context += f"{idx}. {clue}\n"
                        context += "\n"
                        context += f"Questions and Answers: \n"
                        if 'rephrased-questions-part' in proposed_question_dict:
                            rephrased_questions_part = proposed_question_dict['rephrased-questions-part']
                            already_processed = False
                            for j, cur_rephrased_question_part in enumerate(rephrased_questions_part, start=1):
                                if 'reordered-question' in cur_rephrased_question_part:
                                    already_processed = True
                                    break
                                context += f"<question-{j}>{cur_rephrased_question_part['result']}</question-{j}>\n"
                                context += f"<answer-{j}>{cur_rephrased_question_part['answer']}</answer-{j}>\n"
                            if not already_processed:
                                context += "\n"
                                cur_sentence_changer_prompt = sentence_changer_prompt.replace('[[CONTEXT]]', context)
                                future = executor.submit(self.openai_model.generate, self.CLIENT, cur_sentence_changer_prompt, TEMPERATURE)
                                tasks.append((future, rephrased_questions_part))


                        context = "Given clues:\n"
                        for idx, clue in enumerate(objective_relationship_prompts, start=1):
                            context += f"{idx}. {clue}\n"
                        context += "\n"
                        context += f"Questions and Answers: \n"
                        if 'rephrased-questions-hybrid' in proposed_question_dict:
                            rephrased_questions_hybrid = proposed_question_dict['rephrased-questions-hybrid']
                            already_processed = False
                            for j, cur_rephrased_question_hybrid in enumerate(rephrased_questions_hybrid, start=1):
                                if 'reordered-question' in cur_rephrased_question_hybrid:
                                    already_processed = True
                                    break
                                context += f"<question-{j}>{cur_rephrased_question_hybrid['result']}</question-{j}>\n"
                                context += f"<answer-{j}>{cur_rephrased_question_hybrid['answer']}</answer-{j}>\n"
                            if not already_processed:
                                context += "\n"
                                cur_sentence_changer_prompt = sentence_changer_prompt.replace('[[CONTEXT]]', context)
                                future = executor.submit(self.openai_model.generate, self.CLIENT, cur_sentence_changer_prompt, TEMPERATURE)
                                tasks.append((future, rephrased_questions_hybrid))
            else:
                raise ValueError(f"Invalid value for 'SENTENCE_ORDER_CHANGER_GENERATED_TYPE': {self.SENTENCE_ORDER_CHANGER_GENERATED_TYPE}")

            all_num = len(tasks)
            for future_info in tqdm(as_completed([t[0] for t in tasks]), total=len(tasks), desc="Generating", dynamic_ncols=True):
                future = future_info
                idx = [t[0] for t in tasks].index(future)
                if idx == -1:
                    raise ValueError("Invalid index.")
                rephrased_questions = tasks[idx][1]

                sentence_changer_response, _ = future.result(timeout=10*60)
                reordered_questions = parse_transformations(sentence_changer_response)
                reordered_question_index2dict = {
                    reordered_question['question_number']: reordered_question
                    for reordered_question in reordered_questions
                }

                for j, cur_rephrased_question_part in enumerate(rephrased_questions, start=1):
                    if j not in reordered_question_index2dict:
                        continue
                    cur_rephrased_question_part['reordered-question'] = reordered_question_index2dict[j]['transformed_question']
                    cur_rephrased_question_part['reordered-same-meaning-with-origin'] = reordered_question_index2dict[j]['same_meaning_with_origin']
                
                success_num += 1
                if success_num % self.save_interval == 0:
                    dir_path = os.path.dirname(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH)
                    os.makedirs(dir_path, exist_ok=True)
                    print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
                    with open(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                        json.dump(self.inputs, f, indent=2, ensure_ascii=False)

        if success_num or not os.path.exists(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH):
            dir_path = os.path.dirname(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH)
            os.makedirs(dir_path, exist_ok=True)
            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
            with open(self.SENTENCE_ORDER_CHANGER_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                json.dump(self.inputs, f, indent=2, ensure_ascii=False)
        
        return success_num, all_num