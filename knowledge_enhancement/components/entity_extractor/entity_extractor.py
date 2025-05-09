import re
import os
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

# Function to extract entities and relationships
def extract_entity_from_output(input_text):
    entity_pattern = r'\("entity"\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\)'
    relationship_pattern = r'\("relationship"\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\{tuple_delimiter\}(.*?)\)'
    
    # Replace placeholders with actual delimiters
    tuple_delimiter = '<tuple_delimiter>'
    record_delimiter = '<record_delimiter>'
    completion_delimiter = '<completion_delimiter>'
    
    # Parse entities
    entities = re.findall(entity_pattern, input_text)
    parsed_entities = [
        {
            "entity_name": match[0],
            "entity_description": match[1]
        }
        for match in entities
    ]
    
    # Parse relationships
    relationships = re.findall(relationship_pattern, input_text)
    parsed_relationships = []
    for match in relationships:
        # Extract sentence numbers from the sentences_used field
        sentences_raw = match[3]

        # Remove square brackets if present and split by comma
        sentences_clean = sentences_raw.strip("[]")
        
        numbers_and_ranges = re.findall(r'\d+-\d+|\d+', sentences_clean)

        # Create a formatted string for sentences_used
        formatted_sentence_numbers = ','.join(numbers_and_ranges)
        formatted_sentence_numbers = f'{formatted_sentence_numbers}'
        
        parsed_relationships.append({
            "source_entity_name": match[0].strip(),
            "target_entity_name": match[1].strip(),
            "relationship_description": match[2].strip(),
            "sentences_used": formatted_sentence_numbers
        })
    
    # Validate output format
    is_complete = completion_delimiter in input_text
    
    return {
        "entities": parsed_entities,
        "relationships": parsed_relationships,
        "is_complete": is_complete
    }

class EntityExtractor:
    def __init__(self, save_interval=20):
        self.CLIENT = CLIENT
        self.MODEL_NAME = MODEL_NAME

        self.ENTITY_EXTRACTOR_INPUT_PATH, self.ENTITY_EXTRACTOR_PROMPT_PATH, self.ENTITY_EXTRACTOR_OUTPUT_PATH = None, None, None
        if os.getenv("ENTITY_EXTRACTOR_INPUT_PATH", None) != None:
            self.ENTITY_EXTRACTOR_INPUT_PATH = os.getenv("ENTITY_EXTRACTOR_INPUT_PATH")
            self.ENTITY_EXTRACTOR_PROMPT_PATH = os.getenv("ENTITY_EXTRACTOR_PROMPT_PATH")
            self.ENTITY_EXTRACTOR_OUTPUT_PATH = os.getenv("ENTITY_EXTRACTOR_OUTPUT_PATH")
        else:
            raise ValueError("Environment variable 'ENTITY_EXTRACTOR_INPUT_PATH' is not set.")

        self.ENTITY_EXTRACTOR_INPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_EXTRACTOR_INPUT_PATH)
        self.ENTITY_EXTRACTOR_PROMPT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_EXTRACTOR_PROMPT_PATH)
        self.ENTITY_EXTRACTOR_OUTPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_EXTRACTOR_OUTPUT_PATH)
        
        self.ENTITY_EXTRACTOR_STOP_WORDS = os.getenv("ENTITY_EXTRACTOR_STOP_WORDS", None)
        self.ENTITY_EXTRACTOR_NUM_WORKERS = int(os.getenv("ENTITY_EXTRACTOR_NUM_WORKERS", 4))
        self.ENTITY_EXTRACTOR_MAX_NEW_TOKENS = os.getenv("ENTITY_EXTRACTOR_MAX_NEW_TOKENS", None)

        self.openai_model = OpenAIModel(
            MODEL_NAME,
            self.ENTITY_EXTRACTOR_STOP_WORDS,
            self.ENTITY_EXTRACTOR_MAX_NEW_TOKENS
        )
        self.save_interval = save_interval

    def process_input(self, cur_input, entity_extractor_prompt, i):
        try:
            context = cur_input['context']
            cur_entity_extractor_prompt = entity_extractor_prompt.replace('[[CONTEXT]]', context)
            entity_extractor_response, _ = self.openai_model.generate(self.CLIENT, cur_entity_extractor_prompt, TEMPERATURE)
            extract_entity = extract_entity_from_output(entity_extractor_response)
            entities, relationships, is_complete = (
                extract_entity['entities'],
                extract_entity['relationships'],
                extract_entity['is_complete']
            )
            
            filtered_entities = []
            for entity in entities:
                if "entity_name" in entity and "entity_description" in entity:
                    filtered_entities.append(entity)

            entity_names = [entity['entity_name'] for entity in filtered_entities]
            filtered_relationships = []
            for relationship in relationships:
                source_entity_name = relationship['source_entity_name']
                target_entity_name = relationship['target_entity_name']
                if source_entity_name in entity_names and target_entity_name in entity_names:
                    filtered_relationships.append(relationship)

            result = {
                **cur_input,
                'entity': filtered_entities,
                'relationship': filtered_relationships
            }
            return result, i
        except Exception as e:
            print(f"Error processing input {cur_input['id']}: {e}")
            return None, None

    def run(self):
        if os.path.exists(self.ENTITY_EXTRACTOR_OUTPUT_PATH):
            with open(self.ENTITY_EXTRACTOR_OUTPUT_PATH, "r", encoding="utf-8") as f:
                inputs = json.load(f)
            print(f"Loaded {len(inputs)} entity extractor examples from {os.path.relpath(self.ENTITY_EXTRACTOR_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.")
        else:
            with open(self.ENTITY_EXTRACTOR_INPUT_PATH, "r", encoding="utf-8") as f:
                inputs = json.load(f)
            print(f"Loaded {len(inputs)} entity extractor examples from {os.path.relpath(self.ENTITY_EXTRACTOR_INPUT_PATH, CUSTOM_CORPUS_HOME)}.")

        with open(self.ENTITY_EXTRACTOR_PROMPT_PATH, "r", encoding="utf-8") as f:
            entity_extractor_prompt = f.read()
        print(f"Loaded entity extractor prompt from {os.path.relpath(self.ENTITY_EXTRACTOR_PROMPT_PATH, CUSTOM_CORPUS_HOME)}.")

        all_num, success_num = 0, 0
        with ThreadPoolExecutor(max_workers=self.ENTITY_EXTRACTOR_NUM_WORKERS) as executor:
            future_to_input = []
            for i, cur_input in enumerate(inputs):
                if "entity" not in cur_input:
                    future = executor.submit(self.process_input, cur_input, entity_extractor_prompt, i)
                    future_to_input.append(future)

            all_num = len(future_to_input)
            for future in tqdm(as_completed(future_to_input), total=len(future_to_input), dynamic_ncols=True):
                result, i = future.result(timeout=10*60)
                if result != None:
                    inputs[i] = result
                    success_num += 1
                    if success_num % self.save_interval == 0:
                        dir_path = os.path.dirname(self.ENTITY_EXTRACTOR_OUTPUT_PATH)
                        os.makedirs(dir_path, exist_ok=True)
                        print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.ENTITY_EXTRACTOR_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
                        with open(self.ENTITY_EXTRACTOR_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                            json.dump(inputs, f, indent=2, ensure_ascii=False)

        if success_num or not os.path.exists(self.ENTITY_EXTRACTOR_OUTPUT_PATH):
            dir_path = os.path.dirname(self.ENTITY_EXTRACTOR_OUTPUT_PATH)
            os.makedirs(dir_path, exist_ok=True)
            print(f'Saving {success_num}/{all_num} outputs to {os.path.relpath(self.ENTITY_EXTRACTOR_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.')
            with open(self.ENTITY_EXTRACTOR_OUTPUT_PATH, 'w', encoding="utf-8") as f:
                json.dump(inputs, f, indent=2, ensure_ascii=False)
        
        return success_num, all_num
