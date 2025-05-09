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
from concurrent.futures import ThreadPoolExecutor, as_completed

from .. import (
    OpenAIModel,
    extract_largest_json,
    CUSTOM_CORPUS_HOME,
    CLIENT,
    MODEL_NAME,
    EMBEDDING_CLINET,
    EMBEDDING_MODEL_NAME
)
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

class EntityEliminator:
    def __init__(self):
        self.CLIENT = CLIENT
        self.MODEL_NAME = MODEL_NAME

        self.EMBEDDING_CLIENT = EMBEDDING_CLINET
        self.EMBEDDING_MODEL_NAME = EMBEDDING_MODEL_NAME

        self.ENTITY_ELIMINATOR_INPUT_PATH, self.ENTITY_ELIMINATOR_OUTPUT_PATH, self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH = None, None, None
        if os.getenv("ENTITY_ELIMINATOR_INPUT_PATH") != None:
            self.ENTITY_ELIMINATOR_INPUT_PATH = os.getenv("ENTITY_ELIMINATOR_INPUT_PATH")
            self.ENTITY_ELIMINATOR_OUTPUT_PATH = os.getenv("ENTITY_ELIMINATOR_OUTPUT_PATH")
            self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH = os.getenv("ENTITY_ELIMINATOR_OUTPUT_MAP_PATH")
            self.ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD = os.getenv("ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD")
        else:
            raise EnvironmentError("Environment variable 'ENTITY_ELIMINATOR_INPUT_PATH' is not set.")
        
        self.ENTITY_ELIMINATOR_INPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_ELIMINATOR_INPUT_PATH)
        self.ENTITY_ELIMINATOR_OUTPUT_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_ELIMINATOR_OUTPUT_PATH)
        self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH = os.path.join(CUSTOM_CORPUS_HOME, self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH)
        self.ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD = float(self.ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD)

        self.ENTITY_ELIMINATOR_MAX_NEW_TOKENS = os.getenv("self.ENTITY_ELIMINATOR_MAX_NEW_TOKENS", None)
        self.ENTITY_ELIMINATOR_STOP_WORDS = os.getenv("ENTITY_ELIMINATOR_STOP_WORDS", None)
        self.ENTITY_ELIMINATOR_NUM_WORKERS = int(os.getenv("ENTITY_ELIMINATOR_NUM_WORKERS", 4))

        self.openai_model = OpenAIModel(MODEL_NAME, self.ENTITY_ELIMINATOR_STOP_WORDS,self.ENTITY_ELIMINATOR_MAX_NEW_TOKENS)

    def group_entities_by_name(self, all_entities):
        """
        Group entities by their `entity_name`.
        """
        grouped = defaultdict(list)
        for entity in all_entities:
            grouped[entity["entity_name"]].append(entity)
        return grouped

    def resolve_single_conflict_with_same_name(self, entity_name, entity_group, max_attempts=3):
        """
        Use GPT to resolve entities with the same `entity_name` and update mapping relationships based on `entity_id`.
        """
        all_entity_ids = [entity["entity_id"] for entity in entity_group]
        
        origin_prompt = f"""Here are multiple entity descriptions with the name “{entity_name}”:
{json.dumps(entity_group, ensure_ascii=False)}

Please classify these descriptions:
1. If the descriptions convey the same meaning, merge them into a single entity and assign it a single consistent name.
2. If the descriptions convey different meanings, generate more specific names and descriptions for each distinct entity. Ensure that the more specific names clearly reflect the different meanings.
3. You must not skip the processing of any entity.

Only entities with the same meaning are considered the same. For example, "HARVARD UNIVERSITY" and "DUKE UNIVERSITY" are both "UNIVERSITY," but their descriptions refer to different institutions, so they should be treated as separate entities.

The output format should be:
[
    {{"entity_ids": [123, 456], "new_entity_name": "New Specific Name", "new_entity_description": "Updated Description"}},
    {{"entity_ids": [789], "new_entity_name": "New Specific Name", "new_entity_description": "Updated Description"}},
    ...
]
You must strictly adhere to this JSON format in your output!!!
You must strictly adhere to this JSON format in your output!!!
You must strictly adhere to this JSON format in your output!!!
"""
        prompt = origin_prompt
        try:
            attempt = 0
            while attempt < max_attempts:
                attempt += 1
                try:
                    response, _ = self.openai_model.generate(self.CLIENT, prompt, TEMPERATURE)
                    resolved_group = extract_largest_json(response.strip())
                    generated_entity_ids = []

                    for resolved in resolved_group:
                        for entity_id in resolved["entity_ids"]:
                            generated_entity_ids.append(entity_id)
                    generated_entity_ids = sorted(list(set(generated_entity_ids)))

                    if sorted(generated_entity_ids) != sorted(all_entity_ids):
                        # print(f"Error in resolving conflicts for {entity_name}: {list(sorted(generated_entity_ids))} != {list(sorted(all_entity_ids))}")
                        generated_minus_original = set(generated_entity_ids) - set(all_entity_ids)
                        original_minus_generated = set(all_entity_ids) - set(generated_entity_ids)
                        prompt = origin_prompt +f"Don't missed the following entity IDs in your answer: {list(original_minus_generated)}\n"
                        prompt += f"Don't add extra entity IDs in your answer: {list(generated_minus_original)}\n"
                        raise ValueError("Resolved entity IDs do not match the original entity IDs.")
                    
                    break

                except ValueError as e:
                    if attempt < max_attempts:
                        pass
                        # print(f"Attempt {attempt} failed, retrying...")
                    else:
                        # print(f"Attempt {attempt} failed, giving up.")
                        raise e

            # Return the resolved group with updated entities
            return [
                {
                    "entity_ids": resolved["entity_ids"],
                    "entity_name": resolved["new_entity_name"],
                    "entity_description": resolved["new_entity_description"]
                }
                for resolved in resolved_group
            ]
        except Exception as e:
            # Fallback: Map all entities in the group to their original names and descriptions
            return [
                {
                    "entity_ids": [entity["entity_id"]],
                    "entity_name": entity["entity_name"],
                    "entity_description": entity["entity_description"],
                }
                for entity in entity_group
            ]

    def resolve_conflicts_in_large_group_with_same_name(self, entity_name, all_entities, max_chunk_size=10, max_attempts=2):
        """
        Resolve conflicts in large groups by processing in chunks and iteratively merging until no further merges occur.

        Args:
        entity_name (str): The name of the entities to resolve.
        all_entities (list of dict): List of entities with the same name.
        max_chunk_size (int): Maximum chunk size for processing.
        max_attempts (int): Maximum number of attempts for resolving conflicts.
        """
        
        entities_to_process = copy.deepcopy(all_entities)
        attempts_without_reduction = 0
        entityid2entityid, resolved_entities = {}, []
        max_recycle = 5
        cur_recycle = 0
        while True:
            # print(len(entities_to_process), end=" ")
            prev_num_entities = len(entities_to_process)
            
            random.shuffle(entities_to_process)

            chunks = [entities_to_process[i:i + max_chunk_size] for i in range(0, len(entities_to_process), max_chunk_size)]
            resolved_entities = []
            for chunk in chunks:
                resolved_chunk = self.resolve_single_conflict_with_same_name(entity_name, chunk)
                for resolved_entity in resolved_chunk:
                    smallest_entity_id = min(resolved_entity["entity_ids"])
                    for entity_id in resolved_entity["entity_ids"]:
                        entityid2entityid[entity_id] = smallest_entity_id
                    resolved_entities.append({
                        "entity_id": smallest_entity_id,
                        "entity_name": resolved_entity["entity_name"],
                        "entity_description": resolved_entity["entity_description"]
                    })
            
            curr_num_entities = len(resolved_entities)
            
            if curr_num_entities < prev_num_entities:
                # Reduction occurred, reset attempts and continue processing
                cur_recycle += 1
                if cur_recycle >= max_recycle:
                    # Break the loop after max_recycle attempts
                    break

                entities_to_process = resolved_entities
            else:
                attempts_without_reduction += 1
                if attempts_without_reduction >= max_attempts:
                    # No further reduction after max_attempts, break the loop
                    break
                else:
                    entities_to_process = resolved_entities
        
        # print()
        return resolved_entities, entityid2entityid

    def resolve_conflicts_with_same_name(self, all_entities):
        """
        Simplified conflict resolution by grouping entities with the same name.
        
        Args:
            all_entities (list of dict): List of entities with the same name.
        
        Returns:
            tuple: (resolved_entities, entityid2entityid)
                - resolved_entities: List of resolved entity dictionaries.
                - entityid2entityid: Mapping from original entity_id to resolved entity_id.
        """
        # Group entities by their `entity_name`
        grouped_entities = self.group_entities_by_name(all_entities)
        print(f"Grouped into {len(grouped_entities)} unique entity names.")
        
        # Count how many groups have more than one entity
        groups_to_resolve = sum(1 for group in grouped_entities.values() if len(group) > 1)
        print(f"Total {groups_to_resolve} entity groups with the same name to resolve.")
        
        entityid2entityid = {}
        resolved_entities = []
        
        for entity_name, entity_group in grouped_entities.items():
            if len(entity_group) == 1:
                # No conflict, map the entity to itself
                entity = entity_group[0]
                resolved_entities.append({
                    "entity_id": entity["entity_id"],
                    "entity_name": entity["entity_name"],
                    "entity_description": entity["entity_description"]
                })
                entityid2entityid[entity["entity_id"]] = entity["entity_id"]
            else:
                # Conflict exists, choose a representative (e.g., smallest entity_id)
                representative = min(entity_group, key=lambda x: x["entity_id"])
                resolved_entities.append({
                    "entity_id": representative["entity_id"],
                    "entity_name": representative["entity_name"],
                    "entity_description": representative["entity_description"]
                })
                for entity in entity_group:
                    entity_id = entity["entity_id"]
                    entityid2entityid[entity_id] = representative["entity_id"]
        
        print(f"Resolved entities with the same name. {len(resolved_entities)} entities after resolution.")
        
        return resolved_entities, entityid2entityid

    def vectorize_entities(self, all_entities):
        """
        Convert entity descriptions into vectors using OpenAI embedding model.
        
        Args:
            all_entities (list of dict): List of entities with 'entity_description' keys.
        
        Returns:
            dict: Dictionary where keys are entity indices and values are embedding vectors.
        """
        descriptions = [entity['entity_description'] for entity in all_entities]
        embeddings = self.get_openai_embeddings(descriptions)
        return embeddings

    def get_openai_embeddings(self, texts):
        """
        Generate embeddings using OpenAI's embedding model.
        
        Args:
            texts (list of str): List of texts to generate embeddings for.
        
        Returns:
            list of numpy arrays: List of embedding vectors.
        """
        response = self.EMBEDDING_CLIENT.embeddings.create(
            model=self.EMBEDDING_MODEL_NAME,
            input=texts
        )
        return [np.array(data.embedding) for data in response.data]

    def cluster_entities(self, all_entities, similarity_matrix, threshold):
        """
        Cluster entities based on similarity matrix using DBSCAN.
        
        Args:
            all_entities (list): List of entities.
            similarity_matrix (numpy.ndarray): Precomputed similarity matrix.
            threshold (float): Similarity threshold for clustering.
        
        Returns:
            list: A list of clusters, where each cluster is a list of entity indices.
        """
        # Transform similarity matrix to distance matrix (1 - similarity)
        distance_matrix = 1 - similarity_matrix

        # DBSCAN clustering with a minimum similarity threshold
        clustering = DBSCAN(eps=1-threshold, min_samples=1, metric="precomputed")
        labels = clustering.fit_predict(distance_matrix)

        # Group indices by cluster labels
        clusters = [[] for _ in range(max(labels) + 1)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        return clusters

    def resolve_single_conflict_with_different_name(self, entity_group, max_attempts=3):
        """
        Use GPT to resolve entities with different names and update mapping relationships based on `entity_id`.
        """
        # Build prompt
        all_entity_ids = [entity["entity_id"] for entity in entity_group]
        origin_prompt = f"""Here are multiple entity descriptions:
{json.dumps(entity_group, ensure_ascii=False)}

Please classify these descriptions:
1. If the descriptions convey the same meaning, merge them into a single entity and assign it a single consistent name.
2. If the descriptions convey different meanings, generate more specific names and descriptions for each distinct entity. Ensure that the more specific names clearly reflect the different meanings.
3. You must not skip the processing of any entity.

Only entities with the same meaning are considered the same. For example, "HARVARD UNIVERSITY" and "DUKE UNIVERSITY" are both "UNIVERSITY," but their descriptions refer to different institutions, so they should be treated as separate entities.

The output format should be:
[
    {{"entity_ids": [123, 456], "new_entity_name": "New Specific Name", "new_entity_description": "Updated Description"}},
    {{"entity_ids": [789], "new_entity_name": "New Specific Name", "new_entity_description": "Updated Description"}},
    ...
]
You must strictly adhere to this JSON format in your output!!!
You must strictly adhere to this JSON format in your output!!!
You must strictly adhere to this JSON format in your output!!!
"""
        prompt = origin_prompt
        try:

            attampt = 0
            while attampt < max_attempts:
                attampt += 1
                try:
                    response, _ = self.openai_model.generate(self.CLIENT, prompt, TEMPERATURE)
                    resolved_group = extract_largest_json(response.strip())
                    generated_entity_ids = []

                    for resolved in resolved_group:
                        for entity_id in resolved["entity_ids"]:
                            generated_entity_ids.append(entity_id)
                    
                    if sorted(generated_entity_ids) != sorted(all_entity_ids):
                        # print(f"Error in resolving conflicts: {list(sorted(generated_entity_ids))} != {list(sorted(all_entity_ids))}")
                        generated_minus_original = set(generated_entity_ids) - set(all_entity_ids)
                        original_minus_generated = set(all_entity_ids) - set(generated_entity_ids)
                        prompt = origin_prompt + f"Don't missed the following entity IDs in your answer: {list(original_minus_generated)}\n"
                        prompt += f"Don't add extra entity IDs in your answer: {list(generated_minus_original)}\n"
                        raise ValueError("Resolved entity IDs do not match the original entity IDs.")
                    
                    break

                except ValueError as e:
                    if attampt < max_attempts:
                        pass
                        # print(f"Attempt {attampt} failed, retrying...")
                    else:
                        # print(f"Attempt {attampt} failed, giving up.")
                        raise e

            return [
                {
                    "entity_ids": resolved["entity_ids"],
                    "entity_name": resolved["new_entity_name"],
                    "entity_description": resolved["new_entity_description"]
                }
                for resolved in resolved_group
            ]
        except Exception as e:
            # Fallback: Map all entities in the cluster to their original names/descriptions
            return [
                {
                    "entity_ids": [entity["entity_id"]],
                    "entity_name": entity["entity_name"],
                    "entity_description": entity["entity_description"],
                }
                for entity in entity_group
            ]

    def resolve_conflicts_in_large_group_with_different_name(self, all_entities, max_chunk_size=10, max_attempts = 2):
        """
        Resolve conflicts in large clusters by processing in chunks and iteratively merging until no further merges occur.
        """
        all_entity_ids = [entity["entity_id"] for entity in all_entities]
        entities_to_process = copy.deepcopy(all_entities)
        attempts_without_reduction = 0
        max_recycle = 5
        cur_recycle = 0
        entityid2entityid, resolved_entities = {}, []
        while True:
            prev_num_entities = len(entities_to_process)
            # print(len(entities_to_process), end=" ")
            
            random.shuffle(entities_to_process)
            
            # Process in chunks
            chunks = [entities_to_process[i:i + max_chunk_size] for i in range(0, len(entities_to_process), max_chunk_size)]
            resolved_entities = []
            for chunk in chunks:
                resolved_chunk = self.resolve_single_conflict_with_different_name(chunk)
                for resolved_entity in resolved_chunk:
                    smallest_entity_id = min(resolved_entity["entity_ids"])
                    for entity_id in resolved_entity["entity_ids"]:
                        entityid2entityid[entity_id] = smallest_entity_id
                    resolved_entities.append({
                        "entity_id": smallest_entity_id,
                        "entity_name": resolved_entity["entity_name"],
                        "entity_description": resolved_entity["entity_description"]
                    })
            
            # Combine resolved entities
            curr_num_entities = len(resolved_entities)
            
            if curr_num_entities < prev_num_entities:
                # Reduction occurred, reset attempts
                cur_recycle += 1

                if cur_recycle >= max_recycle:
                    # Break the loop after max_recycle attempts
                    break

                entities_to_process = resolved_entities
            else:
                attempts_without_reduction += 1
                if attempts_without_reduction >= max_attempts:
                    # No further reduction after max_attempts, break the loop
                    break
                else:
                    entities_to_process = resolved_entities  # Update entities to process for next iteration
        # print()
        return resolved_entities, entityid2entityid

    def resolve_conflicts_with_different_name(self, all_entities):
        """
        Resolve all entities into a mapping dictionary, optimizing pairwise comparisons.
        """
        # Convert descriptions into vectors
        entityid2entityid = {}
        vectors = self.vectorize_entities(all_entities)
        similarity_matrix = cosine_similarity(vectors)
        similarity_matrix = np.nan_to_num(similarity_matrix)  # Replace NaN with 0
        # Replace 1 with 0.999 to avoid 1.0 similarity
        similarity_matrix[similarity_matrix >= 1] = 0.999
        print(f"Computed similarity matrix of shape {similarity_matrix.shape}.")

        # Define a similarity threshold
        threshold = self.ENTITY_ELIMINATOR_SIMILARITY_THRESHOLD  # Adjust based on the dataset

        # Clustering entities based on similarity
        clusters = self.cluster_entities(all_entities, similarity_matrix, threshold)
        print(f"Generated {len(clusters)} clusters with different name for resolution.")
        print(f"Maximum cluster size: {max(len(cluster) for cluster in clusters)}")
            
        # Resolve conflicts within each cluster using GPT
        entityid2entityid, resolved_entities = {}, []
        
        # Use ThreadPoolExecutor for parallel processing of large groups
        with ThreadPoolExecutor(max_workers=self.ENTITY_ELIMINATOR_NUM_WORKERS) as executor:
            future_to_cluster = {
                executor.submit(self.resolve_conflicts_in_large_group_with_different_name, 
                                [all_entities[idx] for idx in cluster]): cluster
                for cluster in clusters if len(cluster) >= 2
            }
            
            # Handle single entity clusters immediately
            for cluster in clusters:
                if len(cluster) < 2:
                    tmp_resolved_entity = all_entities[cluster[0]]
                    entityid2entityid[tmp_resolved_entity["entity_id"]] = tmp_resolved_entity["entity_id"]
                    resolved_entities.append({
                        "entity_id": tmp_resolved_entity["entity_id"],
                        "entity_name": tmp_resolved_entity["entity_name"],
                        "entity_description": tmp_resolved_entity["entity_description"]
                    })

            # Collect results from futures
            for future in tqdm(as_completed(future_to_cluster), total=len(future_to_cluster), desc="Resolving clusters", dynamic_ncols=True):
                try:
                    tmp_resolved_entities, tmp_entityid2entityid = future.result(timeout=10*60)
                    resolved_entities.extend(tmp_resolved_entities)
                    entityid2entityid.update(tmp_entityid2entityid)
                except TimeoutError as e:
                    print(f"TimeoutError: {e}")
                except Exception as e:
                    print(f"Error resolving conflicts: {e}")
            
        return resolved_entities, entityid2entityid

    def run(self):
        if os.path.exists(self.ENTITY_ELIMINATOR_OUTPUT_PATH):
            print(f"Entity mapping already exists at {os.path.relpath(self.ENTITY_ELIMINATOR_OUTPUT_PATH, CUSTOM_CORPUS_HOME)}.")
            return

        # Load the input data
        with open(self.ENTITY_ELIMINATOR_INPUT_PATH, "r", encoding="utf-8") as f:
            inputs = json.load(f)
        print(f"Loaded {len(inputs)} examples from {os.path.relpath(self.ENTITY_ELIMINATOR_INPUT_PATH, CUSTOM_CORPUS_HOME)}.")

        # Extract all entities from the input data
        all_entities = []
        for cur_input in inputs:
            if "entity" not in cur_input:
                continue
            for cur_entity in cur_input['entity']:
                all_entities.append(cur_entity)

        all_entities = list(sorted(all_entities, key=lambda x: x["entity_name"]))
        print(f"Total {len(all_entities)} entities to resolve.")

        # Resolve conflicts with the same name
        resolved_entities, entityid2entityid_1 = self.resolve_conflicts_with_same_name(all_entities)
        if not os.path.exists("cache"):
            os.makedirs("cache")
        base_name = os.path.basename(self.ENTITY_ELIMINATOR_INPUT_PATH)
        with open(f"cache/resolved_entities_same_name_{base_name}.json", "w", encoding="utf-8") as f:
            json.dump(resolved_entities, f, indent=2, ensure_ascii=False)
        print(f"Resolved entities with same name saved to cache/resolved_entities_same_name_{base_name}.json.")
        with open(f"cache/entityid2entityid_1_{base_name}.json", "w", encoding="utf-8") as f:
            json.dump(entityid2entityid_1, f, indent=2, ensure_ascii=False)
        
        base_name = os.path.basename(self.ENTITY_ELIMINATOR_INPUT_PATH)
        with open(f"cache/resolved_entities_same_name_{base_name}.json", "r", encoding="utf-8") as f:
            resolved_entities = json.load(f)
        with open(f"cache/entityid2entityid_1_{base_name}.json", "r", encoding="utf-8") as f:
            entityid2entityid_1 = json.load(f)

        # # Resolve conflicts with different names
        # resolved_entities, entityid2entityid_2 = self.resolve_conflicts_with_different_name(resolved_entities)
        # base_name = os.path.basename(self.ENTITY_ELIMINATOR_INPUT_PATH)
        # with open(f"cache/resolved_entities_different_name_{base_name}.json", "w", encoding="utf-8") as f:
        #     json.dump(resolved_entities, f, indent=2, ensure_ascii=False)
        # print(f"Resolved entities with different name saved to cache/resolved_entities_different_name_{base_name}.json.")
        # with open(f"cache/entityid2entityid_2_{base_name}.json", "w", encoding="utf-8") as f:
        #     json.dump(entityid2entityid_2, f, indent=2, ensure_ascii=False)
        
        # base_name = os.path.basename(self.ENTITY_ELIMINATOR_INPUT_PATH)
        # with open(f"cache/resolved_entities_different_name_{base_name}.json", "r", encoding="utf-8") as f:
        #     resolved_entities = json.load(f)
        # with open(f"cache/entityid2entityid_2_{base_name}.json", "r", encoding="utf-8") as f:
        #     entityid2entityid_2 = json.load(f)

        # Combine the two mappings
        entityid2entityid = {
            int(key): value for key, value in {
                **entityid2entityid_1,
                # **entityid2entityid_2
            }.items()
        }

        entityid2entity = {}
        for entity in all_entities:
            entityid2entity[int(entity["entity_id"])] = entity

        # Update the entity ids in the input data
        merge_num = 0
        for cur_input in inputs:
            # Filter entities
            filtered_entities = []
            if "entity" not in cur_input:
                continue
            for entity in cur_input['entity']:
                if "entity_name" in entity and "entity_description" in entity:
                    filtered_entities.append(entity)
            
            cur_input['entity'] = filtered_entities

            for cur_entity in filtered_entities:
                if cur_entity["entity_id"] in entityid2entityid:
                    merge_num += 1
                    origin_entityid = cur_entity["entity_id"]
                    new_entityid = entityid2entityid[origin_entityid]
                    cur_entity["entity_id"] = new_entityid
                    cur_entity["entity_name"] = entityid2entity[new_entityid]["entity_name"]
                    cur_entity["entity_description"] = entityid2entity[new_entityid]["entity_description"]

            filter_relationships = []
            for cur_relationship in cur_input['relationship']:
                if "source_entity_id" in cur_relationship or "target_entity_id" in cur_relationship:
                    filter_relationships.append(cur_relationship)
            
            for cur_relationship in filter_relationships:

                origin_source_entityid = cur_relationship["source_entity_id"]
                origin_target_entityid = cur_relationship["target_entity_id"]
                if origin_source_entityid in entityid2entityid:
                    new_source_entity_id = entityid2entityid[origin_source_entityid]
                    cur_relationship["source_entity_id"] = new_source_entity_id
                    cur_relationship["source_entity_name"] = entityid2entity[new_source_entity_id]["entity_name"]
                if origin_target_entityid in entityid2entityid:
                    new_target_entityid = entityid2entityid[origin_target_entityid]
                    cur_relationship["target_entity_id"] = new_target_entityid
                    cur_relationship["target_entity_name"] = entityid2entity[new_target_entityid]["entity_name"]
            
            cur_input['relationship'] = filter_relationships
                    
        # Save the results
        with open(self.ENTITY_ELIMINATOR_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(inputs, f, indent=2, ensure_ascii=False)
        print(f"Entity mapping saved to {os.path.relpath(self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH, CUSTOM_CORPUS_HOME)}, {merge_num} entities merged.")
        with open(self.ENTITY_ELIMINATOR_OUTPUT_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(entityid2entityid, f, indent=2, ensure_ascii=False)
        """
        The final input will include entities, each with an entity_id, entity_name, and entity_description, which are the results of entity resolution.
        This enables mapping from entity_id to chunk_id, and from chunk_id to doc, thereby establishing a mapping from entity_id to doc.
        """

        return inputs