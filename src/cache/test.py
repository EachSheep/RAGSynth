import json

def convert_to_ensure_ascii_false(input_file, output_file):
    """
    Converts a JSON file to ensure_ascii=False format.

    Args:
        input_file: Path to the input JSON file.
        output_file: Path to the output JSON file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:  # Specify UTF-8 encoding for input
            data = json.load(f)

        with open(output_file, 'w', encoding='utf-8') as f:  # Specify UTF-8 encoding for output
            json.dump(data, f, ensure_ascii=False, indent=4)  # Use ensure_ascii=False and indent for pretty printing

        print(f"Successfully converted '{input_file}' to '{output_file}' with ensure_ascii=False.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage:
input_filename = "docs_tier_1.json"  # Replace with your input filename
output_filename = "docs_tier_1_1.json"  # Replace with your desired output filename

convert_to_ensure_ascii_false(input_filename, output_filename)