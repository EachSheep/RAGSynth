import os
import json

from .. import CUSTOM_CORPUS_HOME

class AddEntityId:
    def __init__(self):
        
        self.ADD_ENTITY_ID_INPUT_PATH, self.ADD_ENTITY_ID_OUTPUT_PATH = None, None
        if os.getenv("ADD_ENTITY_ID_INPUT_PATH", None) != None:
            self.ADD_ENTITY_ID_INPUT_PATH = os.getenv("ADD_ENTITY_ID_INPUT_PATH")
            self.ADD_ENTITY_ID_OUTPUT_PATH = os.getenv("ADD_ENTITY_ID_OUTPUT_PATH")
        else:
            raise EnvironmentError("Environment variable 'ADD_ENTITY_ID_INPUT_PATH' is not set.")
            
        self.ADD_ENTITY_ID_INPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ADD_ENTITY_ID_INPUT_PATH)
        self.ADD_ENTITY_ID_OUTPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ADD_ENTITY_ID_OUTPUT_PATH)

    def run(self):
        with open(self.ADD_ENTITY_ID_INPUT_PATH, "r", encoding="utf-8") as f:
            inputs = json.load(f)
        print(f"Loaded {len(inputs)} add entity id examples from {os.path.relpath(self.ADD_ENTITY_ID_INPUT_PATH, CUSTOM_CORPUS_HOME)}.")

        entity_id_beg = 0
        for cur_input in inputs:
            # Filter entities
            filtered_entities = []
            if "entity" not in cur_input:
                continue
            for entity in cur_input['entity']:
                if "entity_name" in entity and "entity_description" in entity:
                    filtered_entities.append(entity)

            entity_name_to_id = {}
            for cur_entity in filtered_entities:
                if "entity_id" not in cur_entity: # Skip if entity_id is already assigned
                    cur_entity["entity_id"] = entity_id_beg
                entity_name_to_id[cur_entity["entity_name"]] = cur_entity["entity_id"]
                entity_id_beg += 1

            cur_input['entity'] = filtered_entities

            filter_relationships = []
            for cur_relationship in cur_input['relationship']:
                source_entity_name = cur_relationship['source_entity_name']
                target_entity_name = cur_relationship['target_entity_name']
                if source_entity_name in entity_name_to_id and target_entity_name in entity_name_to_id:
                    filter_relationships.append(cur_relationship)

            for cur_relationship in filter_relationships:
                source_entity_name = cur_relationship['source_entity_name']
                target_entity_name = cur_relationship['target_entity_name']
                cur_relationship['source_entity_id'] = entity_name_to_id[source_entity_name]
                cur_relationship['target_entity_id'] = entity_name_to_id[target_entity_name]
            
            cur_input['relationship'] = filter_relationships

        with open(self.ADD_ENTITY_ID_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(inputs, f, indent=2, ensure_ascii=False)

        return inputs