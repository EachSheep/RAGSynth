import json

def convert_set_to_list(obj):
    """
    Recursively convert sets to lists in a nested structure.
    """
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {key: convert_set_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_set_to_list(item) for item in obj]
    else:
        return obj


def extract_largest_json(response):
    """
    Extract the largest valid JSON object from a string response using a stack-based approach.
    
    Args:
        response (str): The string response from the language model.
    
    Returns:
        dict or list: The extracted JSON object.
    
    Raises:
        ValueError: If no valid JSON object is found in the response.
    """
    stack = []
    start_indices = []  # Track starting indices of potential JSON objects/arrays
    potential_jsons = []  # Collect potential JSON strings

    for i, char in enumerate(response):
        if char == '{' or char == '[':
            # Push the opening bracket and its position onto the stack
            stack.append(char)
            start_indices.append(i)
        elif char == '}' or char == ']':
            if stack:
                # Pop the last opening bracket
                opening_bracket = stack.pop()
                start_index = start_indices.pop()
                
                # Check if brackets match
                if (opening_bracket == '{' and char == '}') or (opening_bracket == '[' and char == ']'):
                    # Extract the potential JSON string
                    potential_json = response[start_index:i+1]
                    potential_jsons.append(potential_json)
            else:
                # Unmatched closing bracket, skip it
                continue

    # Parse and find the largest valid JSON object
    largest_json = None
    largest_size = 0
    for potential_json in potential_jsons:
        try:
            parsed_json = json.loads(potential_json)
            size = len(potential_json)
            if size > largest_size:
                largest_json = parsed_json
                largest_size = size
        except json.JSONDecodeError:
            continue

    if largest_json == None:
        raise ValueError("No valid JSON object found in the response.")

    return largest_json

def reformat_objective_facts(data):
    result = {"Objective Facts": []}

    # Reformat Objective Facts
    for idx, fact in enumerate(data['objective-facts'], start=1):
        result["Objective Facts"].append(
            f"{idx}. <detailed-desc>{fact}</detailed-desc>"
        )
    
    result_str = ""
    for key, values in result.items():
        result_str += f"{key}:\n" + "\n".join(values) + "\n"
    
    return result_str