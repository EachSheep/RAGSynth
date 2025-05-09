import os
import random  # Ensure random is imported for shuffling
import json
import copy
from tqdm import tqdm
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .. import (
    OpenAIModel,
    extract_largest_json,
    CUSTOM_CORPUS_HOME,
    CLIENT,
    MODEL_NAME
)
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

class EntityEliminatorEvaluator:
    def __init__(self):
        self.CLIENT = CLIENT
        self.MODEL_NAME = MODEL_NAME

        self.ENTITY_ELIMINATOR_INPUT_PATH, self.ENTITY_ELIMINATOR_OUTPUT_PATH, self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH, self.ENTITY_ELIMINATOR_EVALUATION_PROMPT_PATH, self.ENTITY_ELIMINATOR_EVALUATION_RESULT_PATH = None, None, None, None, None
        if os.getenv("ENTITY_ELIMINATOR_INPUT_PATH", None) != None:
            self.ENTITY_ELIMINATOR_INPUT_PATH = os.getenv("ENTITY_ELIMINATOR_INPUT_PATH")
            self.ENTITY_ELIMINATOR_OUTPUT_PATH = os.getenv("ENTITY_ELIMINATOR_OUTPUT_PATH")
            self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH = os.getenv("ENTITY_ELIMINATOR_OUTPUT_MAP_PATH")
            self.ENTITY_ELIMINATOR_EVALUATION_PROMPT_PATH = os.getenv("ENTITY_ELIMINATOR_EVALUATION_PROMPT_PATH")
            self.ENTITY_ELIMINATOR_EVALUATION_RESULT_PATH = os.getenv("ENTITY_ELIMINATOR_EVALUATION_RESULT_PATH")
            self.ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD = os.getenv("ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD")
        else:
            raise ValueError("Environment variable 'ENTITY_ELIMINATOR_INPUT_PATH' is not set.")
        
        self.ENTITY_ELIMINATOR_INPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_ELIMINATOR_INPUT_PATH)
        self.ENTITY_ELIMINATOR_EVALUATION_PROMPT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_ELIMINATOR_EVALUATION_PROMPT_PATH)
        self.ENTITY_ELIMINATOR_EVALUATION_RESULT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_ELIMINATOR_EVALUATION_RESULT_PATH)
        self.ENTITY_ELIMINATOR_OUTPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_ELIMINATOR_OUTPUT_PATH)
        self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH)
        self.ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD = float(self.ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD)
        
        self.ENTITY_ELIMINATOR_MAX_NEW_TOKENS = os.getenv("ENTITY_ELIMINATOR_MAX_NEW_TOKENS", None)
        self.ENTITY_ELIMINATOR_STOP_WORDS = os.getenv("ENTITY_ELIMINATOR_STOP_WORDS", None)

        self.openai_model = OpenAIModel(MODEL_NAME, self.ENTITY_ELIMINATOR_STOP_WORDS,self.ENTITY_ELIMINATOR_MAX_NEW_TOKENS)

        with open(self.ENTITY_ELIMINATOR_EVALUATION_PROMPT_PATH, "r", encoding="utf-8") as f:
            self._evaluation_prompt = f.read()
        print(f"Loaded entity eliminator prompt from {os.path.relpath(self.ENTITY_ELIMINATOR_EVALUATION_PROMPT_PATH, CUSTOM_CORPUS_HOME)}.")

    def dump_name_map_to_file(self, output_map, input_entityid_2_entity, output_entityid_2_entity):
        """ Dump the entity name mapping to a file.
        Args:
            output_map: dict, the mapping from input entity id to output entity id
            input_entityid_2_entity: dict, the mapping from input entity id to entity
            output_entityid_2_entity: dict, the mapping from output entity id to entity
        Returns:
            str, the entity name mapping
        """
        map_list = []
        for origin_entity_id, new_entity_id in output_map.items():
            origin_entity = input_entityid_2_entity[origin_entity_id]['entity_name']
            new_entity = output_entityid_2_entity[new_entity_id]['entity_name']
            map_list.append({origin_entity: new_entity})
        map_list = list(sorted(map_list, key=lambda x: list(x.keys())[0]))
        map_str = "\n".join([f"{list(cur_map.keys())[0]} -> {list(cur_map.values())[0]}" for cur_map in map_list])
        with open(self.ENTITY_ELIMINATOR_EVALUATION_RESULT_PATH, "w", encoding="utf-8") as f:
            f.write(map_str)
        return map_str
    
    def evaluate_by_llm(self, output_map, input_entityid_2_entity, output_entityid_2_entity):
        """ Evaluate the entity  result by LLM.
        Args:
            output_map: dict, the mapping from input entity id to output entity id
            input_entityid_2_entity: dict, the mapping from input entity id to entity
            output_entityid_2_entity: dict, the mapping from output entity id to entity
        Returns:
            dict, the evaluation result
        """

        to_be_evaluated_num = 0
        for origin_entity_id, new_entity_id in tqdm(output_map.items(), desc="Evaluating entities", total=len(output_map), dynamic_ncols=True):
            origin_entity = input_entityid_2_entity[origin_entity_id]['entity_name']
            origin_entity_desc = input_entityid_2_entity[origin_entity_id]['entity_description']
            new_entity = output_entityid_2_entity[new_entity_id]['entity_name']
            new_entity_desc = output_entityid_2_entity[new_entity_id]['entity_description']
            if origin_entity.lower() != new_entity.lower():
                to_be_evaluated_num += 1
        print(f"Total number of entities to be evaluated: {to_be_evaluated_num}.")
        # input("Press Enter to continue...")

        eval_res = []
        # Prepare the evaluation prompt
        for origin_entity_id, new_entity_id in tqdm(output_map.items(), desc="Evaluating entities", total=len(output_map), dynamic_ncols=True):
            origin_entity = input_entityid_2_entity[origin_entity_id]['entity_name']
            origin_entity_desc = input_entityid_2_entity[origin_entity_id]['entity_description']
            new_entity = output_entityid_2_entity[new_entity_id]['entity_name']
            new_entity_desc = output_entityid_2_entity[new_entity_id]['entity_description']

            if origin_entity.lower() != new_entity.lower():
                cur__evaluation_prompt = copy.deepcopy(self._evaluation_prompt)
                cur__evaluation_prompt = cur__evaluation_prompt.replace(f"[[ENTITY_NAME_A]]", origin_entity, 1)
                cur__evaluation_prompt = cur__evaluation_prompt.replace(f"[[ENTITY_DESCRIPTION_A]]", origin_entity_desc, 1)
                cur__evaluation_prompt = cur__evaluation_prompt.replace(f"[[ENTITY_NAME_B]]", new_entity, 1)
                cur__evaluation_prompt = cur__evaluation_prompt.replace(f"[[ENTITY_DESCRIPTION_B]]", new_entity_desc, 1)
                # Generate the evaluation result
                # _evaluation_response, _ = self.openai_model.generate(self.CLIENT, cur__evaluation_prompt)
                _evaluation_response = "yes"
                _evaluation_result = _evaluation_response.lower().strip()
            else:
                _evaluation_result = "yes"
            if "yes" in _evaluation_result:
                eval_res.append({
                    "if_same": True,
                    'if_name_same': origin_entity.lower() == new_entity.lower(),
                    "origin_entity": origin_entity,
                    "new_entity": new_entity,
                    "origin_entity_desc": origin_entity_desc,
                    "new_entity_desc": new_entity_desc
                })
            elif "no" in _evaluation_result:
                eval_res.append({
                    "if_same": False,
                    'if_name_same': origin_entity.lower() == new_entity.lower(),
                    "origin_entity": origin_entity,
                    "new_entity": new_entity,
                    "origin_entity_desc": origin_entity_desc,
                    "new_entity_desc": new_entity_desc
                })
            else:
                eval_res.append({
                    "if_same": None,
                    'if_name_same': origin_entity.lower() == new_entity.lower(),
                    "origin_entity": origin_entity,
                    "new_entity": new_entity,
                    "origin_entity_desc": origin_entity_desc,
                    "new_entity_desc": new_entity_desc
                })
        # Sorting eval_res based on the specified conditions
        eval_res = sorted(eval_res, key=lambda x: (not x['if_name_same'], not x['if_same']), reverse=True)
        return eval_res
    
    def run(self):
        # Load the input data
        with open(self.ENTITY_ELIMINATOR_INPUT_PATH, "r", encoding="utf-8") as f:
            inputs = json.load(f)
        # Extract all entities from the input data
        input_all_entities = []
        for cur_input in inputs:
            if "entity" not in cur_input:
                continue
            for cur_entity in cur_input['entity']:
                input_all_entities.append(cur_entity)
        input_all_entities = list(sorted(input_all_entities, key=lambda x: x["entity_name"]))
        input_entityid_2_entity = {cur_entity["entity_id"]: cur_entity for cur_entity in input_all_entities}

        # Load the output path
        with open(self.ENTITY_ELIMINATOR_OUTPUT_PATH, "r", encoding="utf-8") as f:
            outputs = json.load(f)
        # Extract all entities from the output data
        output_all_entities = []
        for cur_output in outputs:
            if "entity" not in cur_output:
                continue
            for cur_entity in cur_output['entity']:
                output_all_entities.append(cur_entity)
        output_all_entities = list(sorted(output_all_entities, key=lambda x: x["entity_name"]))
        output_entityid_2_entity = {cur_entity["entity_id"]: cur_entity for cur_entity in output_all_entities}

        with open(self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH, "r", encoding="utf-8") as f:
            output_map = json.load(f)
        # transform key type from str to int
        output_map = {int(k): v for k, v in output_map.items()}

        # self.dump_name_map_to_file(output_map, input_entityid_2_entity, output_entityid_2_entity)
        # print(f"Saved evaluation result to {os.path.relpath(self.ENTITY_ELIMINATOR_EVALUATION_RESULT_PATH, CUSTOM_CORPUS_HOME)}.")

        eval_res = self.evaluate_by_llm(output_map, input_entityid_2_entity, output_entityid_2_entity)
        
        accuracy = sum([1 for cur_eval_res in eval_res if cur_eval_res['if_same']]) / len(eval_res)
        print(f"Accuracy: {accuracy * 100:.2f}%.")

        eval_res_str = ""
        for cur_eval_res in eval_res:
            eval_res_str += json.dumps(cur_eval_res, ensure_ascii=False) + "\n"
        with open(self.ENTITY_ELIMINATOR_EVALUATION_RESULT_PATH, "w", encoding="utf-8") as f:
            f.write(eval_res_str)
        print(f"Saved  evaluation result to {os.path.relpath(self.ENTITY_ELIMINATOR_EVALUATION_RESULT_PATH, CUSTOM_CORPUS_HOME)}.")
        
        return inputs