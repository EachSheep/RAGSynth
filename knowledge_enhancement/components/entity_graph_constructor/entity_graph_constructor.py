import json
from collections import deque

from .. import convert_set_to_list

class EntityRelationshipGraph:
    def __init__(self, data):
        """
        Initializes the EntityRelationshipGraph with the provided data.

        Parameters:
        - data: A list of dictionaries containing 'id', 'entity', and 'relationship' information.
        """
        # Stores entity information keyed by entity_id
        """An example of the entities dictionary:
        {
            0: {
                'entity_name': 'STANFORD UNIVERSITY',
                'entity_description': 'Stanford University is a prestigious institution that provides a dynamic learning environment...',
                'chunk_names': {'chunk_0'},
                'relationships': [
                    {
                        'source_entity_id': 0,
                        'source_entity_name': 'STANFORD UNIVERSITY',
                        'target_entity_id': 1,
                        'target_entity_name': 'STANFORD ADMISSIONS',
                        'relationship_description': "Stanford University's admissions process is managed by Stanford Admissions...",
                        'id': 'chunk_0'
                    },
                    # Add other relationships as needed
                ]
            },
            # Add other entities as needed
        }
        """
        self.entities = {}
        # Processes the data to populate entities and relationships
        self._process_data(data)

    def items(self):
        """
        Returns all the items (entity_id, entity) in the entities dictionary.
        """
        return self.entities.items()
    
    def keys(self):
        """
        Returns all the keys (entity_id) in the entities dictionary.
        """
        return self.entities.keys()
    
    def values(self):
        """
        Returns all the values (entity information) in the entities dictionary.
        """
        return self.entities.values()
    
    def __getitem__(self, entity_id):
        """
        Returns the entity information for the given entity_id.
        """
        return self.entities[entity_id]
    
    def __contains__(self, entity_id):
        """
        Checks if the entity_id is present in the entities dictionary.
        """
        return entity_id in self.entities
    
    def __len__(self):
        """
        Returns the number of entities in the entities dictionary.
        """
        return len(self.entities)
    
    def __iter__(self):
        """
        Returns an iterator over the entity_ids in the entities dictionary.
        """
        return iter(self.entities)
    
    def get(self, entity_id, default=None):
        """
        Returns the entity associated with the given entity_id, or a default value if not found.
        
        Parameters:
        - entity_id: The ID of the entity to retrieve.
        - default: The value to return if the entity_id is not found (defaults to None).
        """
        return self.entities.get(entity_id, default)

    def _process_data(self, data):
        """
        Internal method to process the data and build the entities and relationships.

        Parameters:
        - data: The input data containing entities and relationships.
        """
        for chunk in data:
            # print("chunk: ", chunk)
            id = chunk['id']
            # input("Press Enter to continue...")

            # Process entities in the current chunk
            for entity in chunk.get('entity', []):
                entity_id = entity['entity_id']
                entity_name = entity['entity_name']
                entity_description = entity.get('entity_description', '')

                # Initialize or update the entity's information
                if entity_id not in self.entities:
                    # Initialize the entity's information
                    self.entities[entity_id] = {
                        'entity_id': entity_id,
                        'entity_name': entity_name,
                        'entity_description': entity_description,
                        'chunk_names': set([id]),
                        'relationships': []
                    }
                else:
                    # Entity already exists, update its chunk_names and description if needed
                    self.entities[entity_id]['chunk_names'].add(id)
                    if not self.entities[entity_id]['entity_description']:
                        self.entities[entity_id]['entity_description'] = entity_description

            # Process relationships in the current chunk
            for relation in chunk.get('relationship', []):

                if relation['source_entity_id'] == None or relation['target_entity_id'] == None:
                    # If entity IDs are not found, skip this relationship
                    continue

                # Create a relationship object
                relationship = {
                    **relation,
                    'id': id
                }

                # Add the relationship to both source and target entities
                self.entities[relation['source_entity_id']]['relationships'].append(relationship)
                self.entities[relation['target_entity_id']]['relationships'].append(relationship)

    def get_entity_id_to_chunk_names(self):
        """
        Returns a dictionary mapping entity_id to a list of chunk_names where the entity appears.

        Returns:
        - entity_id_to_chunk_names: A dictionary with entity_id as keys and lists of chunk_names as values.
        """
        entity_id_to_chunk_names = {}
        for entity_id, entity_info in self.entities.items():
            entity_id_to_chunk_names[entity_id] = list(entity_info['chunk_names'])
        return entity_id_to_chunk_names

    def get_entity_id_to_entity_names(self):
        """
        Returns a dictionary mapping entity_id to entity_name.

        Returns:
        - entity_id_to_entity_names: A dictionary with entity_id as keys and entity_name as values.
        """
        entity_id_to_entity_names = {}
        for entity_id, entity_info in self.entities.items():
            entity_id_to_entity_names[entity_id] = entity_info['entity_name']
        return entity_id_to_entity_names
    
    def get_entity_id_to_entity_descriptions(self):
        """
        Returns a dictionary mapping entity_id to entity_description.

        Returns:
        - entity_id_to_entity_descriptions: A dictionary with entity_id as keys and entity_description as values.
        """
        entity_id_to_entity_descriptions = {}
        for entity_id, entity_info in self.entities.items():
            entity_id_to_entity_descriptions[entity_id] = entity_info['entity_description']
        return entity_id_to_entity_descriptions

    def get_entity_id_to_relationships(self):
        """
        Returns a dictionary mapping entity_id to a list of relationships involving that entity.

        Returns:
        - entity_id_to_relationships: A dictionary with entity_id as keys and lists of relationships as values.
        """
        entity_id_to_relationships = {}
        for entity_id, entity_info in self.entities.items():
            entity_id_to_relationships[entity_id] = entity_info['relationships']
        return entity_id_to_relationships

    def get_entities(self):
        """
        Returns the entities dictionary containing all entity information.

        Returns:
        - entities: A dictionary with entity_id as keys and entity information as values.
        """
        return self.entities
    
    def get_subgraph(self, entity_id, depth=1):
        """
        Returns a subgraph containing all entities and relationships connected to the given entity_id up to the specified depth,

        Parameters:
        - entity_id: The ID of the starting entity.
        - depth: The depth of the connections to traverse (default is 1).

        Returns:
        - subgraph: A dictionary containing 'entities' and 'relationships' in the same format.
        """
        visited_entities = set()
        visited_relationships = set()
        queue = deque()
        subgraph_entities = {}
        subgraph_relationships = []

        # Initialize the queue with the starting entity at depth 0
        queue.append((entity_id, 0))

        while queue:
            current_entity_id, current_depth = queue.popleft()
            if current_entity_id in visited_entities:
                continue
            visited_entities.add(current_entity_id)

            # Add the entity to the subgraph
            entity_info = self.entities[current_entity_id]
            subgraph_entities[current_entity_id] = {
                'entity_id': current_entity_id,
                'entity_name': entity_info['entity_name'],
                'entity_description': entity_info['entity_description'],
                'chunk_names': entity_info['chunk_names'],
                'relationships': []  # We'll populate this later
            }

            if current_depth < depth:
                # Traverse relationships
                for relationship in entity_info['relationships']:
                    
                    # Create a unique identifier for the relationship to avoid duplicates
                    relationship_id = (relationship['source_entity_id'], relationship['target_entity_id'], relationship['relationship_description'])
                    if relationship_id in visited_relationships:
                        continue
                    visited_relationships.add(relationship_id)

                    # Add the relationship to the subgraph
                    subgraph_relationships.append(relationship)

                    # Add the relationship to the entity's relationships in the subgraph
                    subgraph_entities[current_entity_id]['relationships'].append(relationship)

                    # Determine the next entity to visit
                    next_entity_id = relationship['target_entity_id'] if relationship['source_entity_id'] == current_entity_id else relationship['source_entity_id']
                    if next_entity_id not in visited_entities:
                        queue.append((next_entity_id, current_depth + 1))

        # Build the subgraph in the same format
        subgraph = {
            'entities': subgraph_entities,
            'relationships': subgraph_relationships
        }
        return subgraph

# Example usage:
if __name__ == "__main__":
    # Assume 'data' is the provided data list with entity_ids
    data = [
        {
            "id": "chunk_0",
            "entity": [
                {
                    "entity_id": 0,
                    "entity_name": "STANFORD UNIVERSITY",
                    "entity_description": "Stanford University is a prestigious institution that provides a dynamic learning environment..."
                },
                {
                    "entity_id": 2,
                    "entity_name": "RICHARD H. SHAW",
                    "entity_description": "Richard H. Shaw is the Dean of Undergraduate Admission at Stanford University..."
                },
                {
                    "entity_id": 1,
                    "entity_name": "STANFORD ADMISSIONS",
                    "entity_description": "Stanford Admissions handles the university's admissions process..."
                },
                # Add other entities as needed
            ],
            "relationship": [
                {
                    "source_entity_name": "STANFORD UNIVERSITY",
                    "target_entity_name": "STANFORD ADMISSIONS",
                    "relationship_description": "Stanford University's admissions process is managed by Stanford Admissions...",
                    "source_entity_id": 0,
                    "target_entity_id": 1
                },
                {
                    "source_entity_name": "RICHARD H. SHAW",
                    "target_entity_name": "STANFORD ADMISSIONS",
                    "relationship_description": "Richard H. Shaw, as the Dean of Undergraduate Admission, oversees the admissions process at Stanford University...",
                    "source_entity_id": 2,
                    "target_entity_id": 1
                },
                # Add other relationships as needed
            ]
        },
        # Add other chunks as needed
    ]

    # Initialize the EntityRelationshipGraph with the data
    graph = EntityRelationshipGraph(data)

    # Get entity_id to chunk_names mapping
    entity_id_to_chunk_names = graph.get_entity_id_to_chunk_names()
    print("Entity ID to chunk names mapping:")
    for entity_id, chunks in entity_id_to_chunk_names.items():
        entity_name = graph.entities[entity_id]['entity_name']
        print(f"Entity ID {entity_id} ({entity_name}): {chunks}")

    # Get entity_id to relationships mapping
    entity_id_to_relationships = graph.get_entity_id_to_relationships()
    print("\nEntity ID to relationships mapping:")
    for entity_id, relationships in entity_id_to_relationships.items():
        entity_name = graph.entities[entity_id]['entity_name']
        print(f"Entity ID {entity_id} ({entity_name}):")
        for relation in relationships:
            source_entity_id = relation['source_entity_id']
            target_entity_id = relation['target_entity_id']
            source_entity_name = graph.entities[source_entity_id]['entity_name']
            target_entity_name = graph.entities[target_entity_id]['entity_name']
            print(f"  - Relationship with Entity ID {target_entity_id} ({target_entity_name}): {relation['relationship_description']}")
    
    # Initialize the EntityRelationshipGraph with the data
    graph = EntityRelationshipGraph(data)
    # Get a subgraph for entity_id 0 (STANFORD UNIVERSITY) with depth 1
    subgraph_depth_1 = graph.get_subgraph(entity_id=0, depth=1)
    subgraph_depth_1 = convert_set_to_list(subgraph_depth_1)
    print("Subgraph with depth 1 for Entity ID 0:")
    print(json.dumps(subgraph_depth_1, indent=2, ensure_ascii=False))
    # Get a subgraph for entity_id 0 (STANFORD UNIVERSITY) with depth 2
    subgraph_depth_2 = graph.get_subgraph(entity_id=0, depth=2)
    subgraph_depth_2 = convert_set_to_list(subgraph_depth_2)
    print("\nSubgraph with depth 2 for Entity ID 0:")
    print(json.dumps(subgraph_depth_2, indent=2, ensure_ascii=False))