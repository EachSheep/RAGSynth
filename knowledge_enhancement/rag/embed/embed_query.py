import os
import json
import numpy as np
import logging
import torch
import argparse
from sentence_transformers import SentenceTransformer

from peft import PeftModel, PeftConfig
from transformers import AutoModel

from rag.utils import extract_doc_to_sen

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME is None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_model(embed_model_path, lora_adapter_path=None, max_seq_length=32768):
    """Load and configure the model."""
    logging.getLogger().setLevel(logging.WARNING)
    
    if any(model_name in embed_model_path for model_name in [
        "snowflake-arctic-embed-m-v1.5",
        "MedEmbed-small-v0.1"
    ]):
        model = SentenceTransformer(
            embed_model_path,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16}
        )
    elif any(model_name in embed_model_path for model_name in [
        "stella_en_400M_v5",
        "gte-multilingual-base",
        "snowflake-arctic-embed-m-long",
        "rubert-tiny-turbo",
    ]):
        model = SentenceTransformer(
            embed_model_path,
            trust_remote_code=True
        )
    else:
        raise ValueError(f"Model {embed_model_path} not supported.")
    
    model.max_seq_length = max_seq_length

    # If a LoRA adapter path is provided, load and merge it using PEFT
    if lora_adapter_path:
        try:
            print(f"Loading LoRA adapter from {lora_adapter_path}")
            # Access the underlying HuggingFace model
            # Adjust the attribute access based on your SentenceTransformer version
            base_model = model._first_module().auto_model  # Modify if necessary

            # Load the LoRA configuration
            peft_config = PeftConfig.from_pretrained(lora_adapter_path)

            # Load the LoRA model
            peft_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

            # Merge the LoRA weights into the base model
            peft_model = peft_model.merge_and_unload()

            # Replace the base model in SentenceTransformer with the merged model
            model._first_module().auto_model = peft_model

            print(f"Successfully merged LoRA adapter from {lora_adapter_path}")
        except Exception as e:
            print(f"Failed to load and merge LoRA adapter: {e}")
            raise e
        
    return model

def add_eos_func(input_examples, model):
    """Add EOS token to each input example."""
    return [input_example + model.tokenizer.eos_token for input_example in input_examples]

def load_document_embeddings_by_path(index_path):
    """Load document embeddings from the specified path."""
    doc_embeddings = np.load(index_path)
    return doc_embeddings

def load_content(query_file_path):
    """Load contents from a JSON file."""
    with open(query_file_path, "r") as f:
        contents = json.load(f)
    return contents

def calculate_similarities(query_embeddings, doc_embeddings, top_k=20, query_batch_size=2):
    """Calculate similarity scores between queries and documents and find top-K similar documents."""
    results = []
    num_queries = len(query_embeddings)

    for start_idx in range(0, num_queries, query_batch_size):
        end_idx = min(start_idx + query_batch_size, num_queries)
        batch_query_embeddings = query_embeddings[start_idx:end_idx]

        # Calculate similarity scores for the current batch
        scores = (batch_query_embeddings @ doc_embeddings.T) * 100

        # Process each query in the batch individually
        for score in scores:
            top_k_indices = score.argsort()[-top_k:][::-1]  # Get top K indices
            results.append([int(idx) for idx in top_k_indices])

    return results

def is_val_in_top_k(top_k_documents, target_vals, top_k_values):
    """
    Check if each string in the target string list is among the top_k elements of top_k_documents.
    :param top_k_documents: list of str, the retrieved top_k document IDs.
    :param target_vals: list of str, the target document ID list.
    :param top_k_values: list of int, the values for top_k.
    :return: dict, with top_k values as keys and boolean values indicating whether all target documents are in the top_k retrieval results.
    """
    results = {}
    for top_k_value in top_k_values:
        top_k_elements = top_k_documents[:top_k_value]
        results[top_k_value] = all(val in top_k_elements for val in target_vals)
    return results

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

def are_all_elements_in_list(target_vals, source_list):
    """
    Check if all elements in the target_vals list are present in the source_list.
    :param target_vals: list of str, the list of target elements.
    :param source_list: list of str, the source list.
    :return: bool, returns True if all target elements are in the source list, otherwise returns False.
    """
    return all(val in source_list for val in target_vals)

def process_questions(contents, question_type, question_category, model, doc_embeddings, chunkid_2_dict, args):
    """Process questions in the specified categories and retrieve top-K documents."""
    all_questions = []
    question_mapping = []  # List to keep track of question positions

    if question_type == "content":
        for chunk_idx, chunk in enumerate(contents):
            chunk_id = chunk.get('id')
            proposed_questions = chunk.get("proposed-questions", {})
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                if question_category == "original-question":
                    question = proposed_question_dict.get("question")
                    if question:
                        all_questions.append(question)
                        question_mapping.append((chunk_idx, proposed_question_type, chunk_id, proposed_question_dict))
                elif question_category in ["rephrased-questions"]:
                    rephrased_questions = proposed_question_dict.get(question_category, [])
                    for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions):
                        question = rephrased_question_dict.get("result")
                        if question:
                            all_questions.append(question)
                            question_mapping.append((
                                chunk_idx, proposed_question_type, rephrased_question_type, chunk_id, rephrased_question_dict))
                elif question_category in ["rephrased-questions-part", "rephrased-questions-hybrid"]:
                    rephrased_questions = proposed_question_dict.get(question_category, [])
                    for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions):
                        question = rephrased_question_dict.get("reordered-question")
                        if question:
                            all_questions.append(question)
                            question_mapping.append((
                                chunk_idx, proposed_question_type, rephrased_question_type, chunk_id, rephrased_question_dict))
                else:
                    raise ValueError(f"Invalid question category: {question_category}")
    else:  # entity_graph
        for chunk_idx, chunk in contents.items():
            proposed_questions = chunk.get("proposed-questions", {})
            # all_objective_relationships = chunk['selected-relationships']['objective-relationships']
            # objective_relationship_id_2_objective_relationship = {
            #     idx: fact for idx, fact in enumerate(all_objective_relationships, start=1)
            # }
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                # Get real related documents
                # needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict.get('objective-relationship-id', '').strip())
                # needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                # needed_related_relationships = [
                #     objective_relationship_id_2_objective_relationship[
                #         int(clue_id)] for clue_id in needed_objective_relationship_ids if clue_id and int(clue_id) in objective_relationship_id_2_objective_relationship
                #     ]
                # needed_corpusids = list(set([cur_related_relationship['id'] for cur_related_relationship in needed_related_relationships if 'id' in cur_related_relationship]))
                if "positive" not in proposed_question_dict:
                    continue         
                positive = proposed_question_dict["positive"]
                if not positive:
                    continue
                needed_corpusid_2_sens = extract_doc_to_sen(positive)
                needed_corpusids = list(needed_corpusid_2_sens.keys())

                if question_category == "original-question":
                    question = proposed_question_dict.get("question")
                    if question:
                        all_questions.append(question)
                        question_mapping.append((
                            chunk_idx, proposed_question_type, needed_corpusids, proposed_question_dict))
                elif question_category in ["rephrased-questions"]:
                    rephrased_questions = proposed_question_dict.get(question_category, [])
                    for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions):
                        question = rephrased_question_dict.get("result")
                        if question:
                            all_questions.append(question)
                            question_mapping.append((
                                chunk_idx, proposed_question_type, rephrased_question_type, needed_corpusids, rephrased_question_dict))
                elif question_category in ["rephrased-questions-part", "rephrased-questions-hybrid"]:
                    rephrased_questions = proposed_question_dict.get(question_category, [])
                    for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions):
                        question = rephrased_question_dict.get("reordered-question")
                        if question:
                            all_questions.append(question)
                            question_mapping.append((
                                chunk_idx, proposed_question_type, rephrased_question_type, needed_corpusids, rephrased_question_dict))
                else:
                    raise ValueError(f"Invalid question category: {question_category}")

    if not all_questions:
        return

    # Optionally add EOS token
    if args.add_eos:
        all_questions = add_eos_func(all_questions, model)

    # Encode queries
    query_prefix = args.query_prefix if args.query_prefix else None
    prompt_name = args.prompt_name if args.prompt_name else None
    query_embeddings = model.encode(
        all_questions,
        batch_size=args.query_batch_size,
        prompt=query_prefix,
        prompt_name=prompt_name,
        normalize_embeddings=args.normalize_embeddings,
        show_progress_bar=False
    )

    # Calculate similarities and get top-K results
    top_k_results = calculate_similarities(
        query_embeddings,
        doc_embeddings,
        top_k=args.top_k,
        query_batch_size=args.query_batch_size
    )

    # Update the contents with retrieved documents
    for i, mapping in enumerate(question_mapping):
        top_k_result = top_k_results[i]
        top_k_documents = [chunkid_2_dict[tmp_idx]["id"] for tmp_idx in top_k_result]

        if question_type == "content":
            if question_category == "original-question":
                chunk_idx, proposed_question_type, chunk_id, proposed_question_dict = mapping
                target_vals = [chunk_id]
            elif question_category in ["rephrased-questions", "rephrased-questions-part", "rephrased-questions-hybrid"]:
                chunk_idx, proposed_question_type, rephrased_question_type, chunk_id, proposed_question_dict = mapping
                target_vals = [chunk_id]
            else:
                raise ValueError(f"Invalid question category: {question_category}")
            # Update the proposed_question_dict
            proposed_question_dict["top_k_documents"] = top_k_documents
            proposed_question_dict["total_docs"] = len(doc_embeddings)
            # Compute all_in_top_k
            tmp_result = is_val_in_top_k(top_k_documents, target_vals, args.top_k_values)
            for top_k_value in args.top_k_values:
                all_in_top_k_key_name = f"all_in_top_{top_k_value}"
                proposed_question_dict[all_in_top_k_key_name] = tmp_result[top_k_value]
        else:  # entity_graph
            if question_category == "original-question":
                chunk_idx, proposed_question_type, needed_corpusids, proposed_question_dict = mapping
            elif question_category in ["rephrased-questions", "rephrased-questions-part", "rephrased-questions-hybrid"]:
                chunk_idx, proposed_question_type, rephrased_question_type, needed_corpusids, proposed_question_dict = mapping
            else:
                raise ValueError(f"Invalid question category: {question_category}")
            # Update the proposed_question_dict
            proposed_question_dict["top_k_documents"] = top_k_documents
            proposed_question_dict["total_docs"] = len(doc_embeddings)
            # Compute all_in_top_k
            for top_k_value in args.top_k_values:
                all_in_top_k_key_name = f"all_in_top_{top_k_value}"
                proposed_question_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:top_k_value])

def main():
    print("-" * 50)
    parser = argparse.ArgumentParser(description="Find similar documents for queries.")
    parser.add_argument("--question_type", type=str, choices=["content", "entity_graph"], help="Type of question to process.")

    parser.add_argument("--embed_model_path", type=str, help="Path to the model directory.")
    parser.add_argument("--query_prefix", type=str, default=None, help="Query prefix.")
    parser.add_argument("--prompt_name", type=str, default=None, help="Prompt name.")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Maximum sequence length.")
    parser.add_argument("--normalize_embeddings", action="store_true", help="Normalize embeddings.")
    parser.add_argument("--add_eos", action="store_true", help="Add EOS token to each input example.")

    parser.add_argument("--chunk_path", type=str, help="Path to the JSON file containing chunks.")
    parser.add_argument("--index_path", type=str, help="Path of document embeddings.")
    parser.add_argument("--input_path", type=str, help="Path to the contents JSON file.")
    parser.add_argument("--output_path", type=str, help="Path to save the updated contents with top-K results.")
    parser.add_argument("--query_batch_size", type=int, default=2, help="Query batch size.")
    parser.add_argument('--top_k_values', type=int, nargs='+', default=[3, 5], help="List of top_k values for which precision will be calculated.")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top documents to retrieve.")
    
    parser.add_argument("--lora_adapter_path", type=str, default=None, help="Path to the LoRA adapter to be merged.")
    args = parser.parse_args()

    args.chunk_path = os.path.join(CUSTOM_CORPUS_HOME, args.chunk_path)
    args.index_path = os.path.join(CUSTOM_CORPUS_HOME, args.index_path)
    args.input_path = os.path.join(CUSTOM_CORPUS_HOME, args.input_path)
    args.output_path = os.path.join(CUSTOM_CORPUS_HOME, args.output_path)

    # Copy file at chunk_path to output_path
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    chunk_output_path = os.path.join(output_dir, os.path.basename(args.chunk_path))
    if not os.path.exists(chunk_output_path):
        os.system(f"cp {args.chunk_path} {chunk_output_path}")

    if os.path.exists(args.output_path):
        print(f"Output file {args.output_path} already exists. Skipping...")
        print("-" * 50)
        exit(0)

    print(f"Processing contents to save to {args.output_path}")
    with open(args.chunk_path, 'r') as f:
        chunks_data = json.load(f)

    chunkid_2_dict = {i: cur_dict for i, cur_dict in enumerate(chunks_data)}

    doc_embeddings = load_document_embeddings_by_path(args.index_path)
    contents = load_content(args.input_path)
    model = load_model(args.embed_model_path, args.lora_adapter_path, args.max_seq_length)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model loaded: {args.embed_model_path} on {model.device}")

    # Define the question categories to process
    question_categories = ["original-question", "rephrased-questions", "rephrased-questions-part", "rephrased-questions-hybrid"]
    for question_category in question_categories:
        process_questions(contents, args.question_type, question_category, model, doc_embeddings, chunkid_2_dict, args)

    # Save the updated contents
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(contents, f, indent=2, ensure_ascii=False)
    print(f"Updated contents saved to {args.output_path}")
    print("-" * 50)

if __name__ == "__main__":
    main()
