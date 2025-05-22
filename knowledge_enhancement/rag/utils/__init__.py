import re
import math
import numpy as np
from collections import defaultdict
from scipy.stats import entropy
from scipy.special import rel_entr
from typing import Dict, List

def get_needed_corpusid2senposes(needed_corpusid2corpus, needed_corpusid2senids):
    """
    Generate a mapping `needed_corpusid2senposes` that contains the start and end character positions of each specified sentence.

    Parameters:
    - needed_corpusid2corpus: dict, keys are corpusid, values are texts with [Sen x] annotations added.
    - needed_corpusid2senids: dict, keys are corpusid, values are lists of senids for which positions need to be obtained.

    Returns:
    - needed_corpusid2senposes: dict, keys are corpusid, values are dictionaries mapping senid to (start, end) positions.

    needed_corpusid2corpus = {
        'corpus1': "这是第一句话。[Sen 1]\n这是第二句话。[Sen 2]\n这是第三句话。[Sen 3]",
        'corpus2': "另一个语料的第一句。[Sen 1]\n第二句内容。[Sen 2]"
    }

    needed_corpusid2senids = {
        'corpus1': [1, 3],
        'corpus2': [2]
    }

    senposes = get_needed_corpusid2senposes(needed_corpusid2corpus, needed_corpusid2senids)
    for corpusid, senid2pos in senposes.items():
        for senid, pos in senid2pos.items():
            print(f"Corpus {corpusid}, Sentence {senid}: {pos}, Text: {needed_corpusid2corpus[corpusid][pos[0]:pos[1]]}")
    Output:
    {
        'corpus1': {
            1: (0, 7),
            3: (21, 28)
        },
        'corpus2': {
            2: (14, 21)
        }
    }
    """
    needed_corpusid2senposes = {}

    # Regular expression pattern to match [Sen x]
    sen_pattern = re.compile(r'\[Sen (\d+)\]')

    for corpusid, corpus in needed_corpusid2corpus.items():
        senids_needed = set(needed_corpusid2senids.get(corpusid, []))
        if not senids_needed:
            continue  # Skip if the current corpusid has no required senids

        senposes = {}
        prev_end = 0  # End position of the previous sentence (initially 0)

        for match in sen_pattern.finditer(corpus):
            senid = int(match.group(1))
            marker_start = match.start()  # Start position of [Sen x]

            if senid in senids_needed:
                sen_start = prev_end
                sen_end = marker_start
                senposes[senid] = (sen_start, sen_end)

            # Update prev_end to the end position of the current [Sen x] for the next sentence
            prev_end = match.end()

        needed_corpusid2senposes[corpusid] = senposes

    return needed_corpusid2senposes

def replace_clue_with_doc_and_sen(all_clueid2docid2senidlist: Dict[int, Dict[int, List[int]]], positive_answer: str) -> str:
    """
    Replaces [Clue xx] or [Clue xx-yy] citations in the positive_answer with formatted [Doc xx, Sen xx] citations.
    
    Parameters:
    - all_clueid2docid2senidlist: Dict mapping clue IDs to another dict mapping doc IDs to lists of sentence IDs.
      Example:
      {
          1: {1: [1, 2, 3]},
          2: {1: [4, 5]},
          3: {2: [1, 2]},
          4: {2: [3]},
      }
    - positive_answer: String containing [Clue xx] or [Clue xx-yy] patterns.
    
    Returns:
    - new_answer: String with [Clue xx] patterns replaced by formatted citations.
    """
    
    def expand_range(token: str) -> List[int]:
        """
        Expands a token which can be a single number or a range (e.g., '2' or '2-8') into a list of integers.
        """
        if '-' in token:
            start, end = token.split('-')
            return list(range(int(start), int(end) + 1))
        else:
            return [int(token)]
    
    def expand_range_in_list(tokens: List[str]) -> List[int]:
        """
        Processes a list of tokens, expanding ranges and collecting all clue IDs.
        """
        clue_ids = []
        for token in tokens:
            if '-' in token:
                clue_ids.extend(expand_range(token))
            else:
                if token.isdigit():
                    clue_ids.append(int(token))
        return clue_ids
    
    def expand_sen_ranges(nums: List[int]) -> List[str]:
        """
        Converts a sorted list of integers into a list with ranges for consecutive numbers.
        Example: [1,2,3,5] -> ['1-3', '5']
        """
        if not nums:
            return []

        nums = sorted(nums)
        ranges = []
        start = prev = nums[0]

        for num in nums[1:]:
            if num == prev + 1:
                prev = num
            else:
                if prev - start >= 2:
                    ranges.append(f"{start}-{prev}")
                elif prev - start == 1:
                    ranges.append(str(start))
                    ranges.append(str(prev))
                else:
                    ranges.append(str(start))
                start = prev = num

        # Handle the last range
        if prev - start >= 2:
            ranges.append(f"{start}-{prev}")
        elif prev - start == 1:
            ranges.append(str(start))
            ranges.append(str(prev))
        else:
            ranges.append(str(start))

        return ranges

    # Regular expression to find [Clue xx], [Clue xx, yy], [Clue xx-yy], etc.
    clue_pattern = re.compile(r'\[Clue\s+([^\]]+)\]')
    
    def replacement(match):
        # print("match:", match)
        clue_ids_str = match.group(1)
        # Split by comma and/or whitespace
        tokens = re.split(r'[,\s]+', clue_ids_str)
        # Expand tokens to individual clue IDs
        clue_ids = expand_range_in_list(tokens)
        # print("clue_ids:", clue_ids)
        
        # Map doc_id to set of sen_ids
        doc_to_sens = {}
        for cid in clue_ids:
            if cid in all_clueid2docid2senidlist:
                for doc_id, sen_ids in all_clueid2docid2senidlist[cid].items():
                    if doc_id not in doc_to_sens:
                        doc_to_sens[doc_id] = set()
                    doc_to_sens[doc_id].update(sen_ids)
        
        if not doc_to_sens:
            # No valid clues found, return the original string
            return match.group(0)
        
        # Build the citation strings
        citations = []
        for doc_id in sorted(doc_to_sens.keys()):
            sen_list = sorted(doc_to_sens[doc_id])
            sen_ranges = expand_sen_ranges(sen_list)

            if sen_ranges:
                # Prepend 'Sen ' to each range
                sen_formatted = [f"{s}" for s in sen_ranges]
                sen_formatted[0] = f"Sen {sen_formatted[0]}"
                # Join sentence parts with comma
                sen_str = ", ".join(sen_formatted)
                citations.append(f"Doc {doc_id}, {sen_str}")
            else:
                citations.append(f"")
        
        # Format multiple documents with separate brackets
        if len(citations) == 1:
            return f"[{citations[0]}]"
        else:
            # Each document citation in its own brackets
            return "".join(f"[{cit}]" for cit in citations)
    
    # Replace all [Clue ...] patterns using the replacement function
    new_answer = clue_pattern.sub(replacement, positive_answer)
    
    return new_answer

def list_to_docided_string(string_dict):
    """
    Convert a list of strings into a docided string.

    :param string_list: list of str, the list of strings to be converted
    :return: str, the resulting numbered string
    """
    numbered_string = ""
    for index, (doc_id, doc_content) in enumerate(string_dict.items()):
        numbered_string += f"""{index}. <doc>
    <doc-name>{doc_id}</doc-name>
    <detailed-desc>{doc_content}</detailed-desc>
</doc>
"""
    return numbered_string.strip()

def extract_and_remove_think_tags(text):
    # Find all content inside <think> tags
    think_contents = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    
    # Remove all <think> tags and their contents from the text
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    return think_contents, cleaned_text

def dcg(relevances, p):
    """Calculate DCG for the top p items."""
    dcg_value = 0.0
    for i in range(p):
        dcg_value += relevances[i] / math.log2(i + 2)  # i+2 because log_2(1) is 0, using i+1+1
    return dcg_value

def idcg(relevances, p):
    """Calculate iDCG for the top p items."""
    sorted_relevances = sorted(relevances, reverse=True)
    return dcg(sorted_relevances, p)

def kl_divergence(P, Q):

    P = np.array(P)
    Q = np.array(Q)
    
    # Ensure that P and Q are probability distributions
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    
    # Compute the KL divergence from P to Q
    KL_PQ = entropy(P, Q)
    
    return KL_PQ

def jensen_shannon_divergence(P, Q, epsilon=1e-10):
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)

    # Add epsilon to the sum to prevent division by zero
    sum_P = np.sum(P)
    sum_Q = np.sum(Q)
    
    P = P / (sum_P + epsilon)
    Q = Q / (sum_Q + epsilon)
    
    # Compute the midpoint distribution M
    M = 0.5 * (P + Q)
    
    # Add epsilon to prevent log(0)
    P = np.where(P == 0, epsilon, P)
    Q = np.where(Q == 0, epsilon, Q)
    M = np.where(M == 0, epsilon, M)
    
    # Calculate KL divergence with added epsilon to prevent division by zero
    KL_P_M = np.sum(rel_entr(P, M))
    KL_Q_M = np.sum(rel_entr(Q, M))
    
    # Compute the Jensen-Shannon Divergence
    JSD = 0.5 * (KL_P_M + KL_Q_M)
    
    return JSD

def idcg_calculator_with_weight(difference_alpha, proposed_question_dict, needed_corpusids, top_k_documents):
    
    # if 'Qwen2.5-7B-Instruct-base-prob' in proposed_question_dict and proposed_question_dict['Qwen2.5-7B-Instruct-base-prob'] != None:
    #     base_prob = proposed_question_dict['Qwen2.5-7B-Instruct-base-prob']
    #     new_probs = proposed_question_dict['Qwen2.5-7B-Instruct-new-probs']
    #     differences = []
    #     for i, new_prob in enumerate(new_probs):
    #         cur_jensen_shannon_divergence = jensen_shannon_divergence(base_prob, new_prob)
    #         differences.append(cur_jensen_shannon_divergence)
    #     softmax_diff = softmax_with_temperature(differences, difference_alpha=difference_alpha)
    #     # Group `softmax_diff` by the values of `needed_corpusids` and calculate the sum for each group.
    #     grouped_sums = defaultdict(float)
    #     for cur_doc_id, diff in zip(needed_corpusids, softmax_diff):
    #         grouped_sums[cur_doc_id] += diff
    #     grouped_sums = dict(grouped_sums)
    # else:
    grouped_sums = defaultdict(float)
    for cur_doc_id in needed_corpusids:
        grouped_sums[cur_doc_id] = 1.0
    grouped_sums = dict(grouped_sums)
    return grouped_sums

def idcg_calculator(needed_corpusids):
    
    grouped_sums = defaultdict(float)
    for cur_doc_id in needed_corpusids:
        grouped_sums[cur_doc_id] = 1.0
    grouped_sums = dict(grouped_sums)
    return grouped_sums

def softmax_with_temperature(logits, difference_alpha=1.0):
    """
    Compute the softmax of vector `logits` with a given `difference_alpha`.

    :param logits: A list or numpy array of logits.
    :param difference_alpha: A float representing the difference_alpha parameter.
    :return: A numpy array representing the softmax probabilities.
    """
    if difference_alpha <= 0:
        raise ValueError("Temperature must be greater than zero.")

    logits = np.array(logits)
    exp_logits = np.exp(difference_alpha * logits)
    softmax_probs = exp_logits / np.sum(exp_logits)

    return softmax_probs

def is_val_in_top_k(source_list, target_val, top_k_values):
    """
    Check if the target is among the cur_top_k_value elements of the list.
    
    :param source_list: list of str, the list of strings to check.
    :param target_val: str, the target string to find.
    :param top_k_values: list of int, list of cur_top_k_value values to consider.
    :return: dict, key is the cur_top_k_value value, value is a boolean indicating if the target string is in the cur_top_k_value elements.
    """
    results = {}
    for cur_top_k_value in top_k_values:
        # Check if the target string is within the first cur_top_k_value elements
        results[cur_top_k_value] = target_val in source_list[:cur_top_k_value]
    return results

def are_all_elements_in_list(list1, list2):
    """
    Check if all elements of list1 are present in list2.

    :param list1: The first list
    :param list2: The second list
    :return: True if all elements of list1 are in list2; otherwise False
    """
    return all(element in list2 for element in list1)

def list_to_numbered_string(string_list):
    """
    Convert a list of strings into a numbered string.

    :param string_list: list of str, the list of strings to be converted
    :return: str, the resulting numbered string
    """
    numbered_string = ""
    for index, string in enumerate(string_list, start=1):
        numbered_string += f"{index}. {string}\n"
    return numbered_string.strip()

def extract_pure_clues_or_sens(text, begin_tag):
    # begin_tag = "Clue" or "Sen"

    # Use regular expressions to match patterns like [Clue X]
    pattern = rf'\[{begin_tag}\s+([0-9]+(?:,\s*[0-9]+|-?[0-9]+)*)\]'
    matches = re.findall(pattern, text)
    
    clues = []
    for match in matches:
        # Split each matched string by commas and remove extra spaces
        items = match.split(',')
        for item in items:
            item = item.strip()
            if '-' in item:
                # Handle ranges of clues, e.g., 3-8
                start, end = map(int, item.split('-'))
                clues.extend(range(start, end + 1))
            else:
                # Handle single clue numbers
                clues.append(int(item))
    
    return clues

def expand_numbers_and_ranges(numbers_and_ranges):
    expanded_numbers = []
    for item in numbers_and_ranges:
        if '-' in item:  # It's a range like 'xx1-xx2'
            start, end = map(int, item.split('-'))
            if start > end:
                end, start = start, end
            expanded_numbers.extend(range(start, end + 1))
        else:  # It's a single number
            expanded_numbers.append(int(item))
    expanded_numbers = list(sorted(list(set(expanded_numbers))))
    return expanded_numbers

def cal_percentage_of_elements_in_list(needed_corpus_ids, top_k_documents):
    """
    Calculate the ratio of the length of the intersection of top_k_documents 
    and needed_corpus_ids to the length of top_k_documents.

    Parameters:
    needed_corpus_ids (list): List of needed related document IDs.
    top_k_documents (list): List of Top K document IDs.

    Returns:
    float: The ratio of the intersection length to the length of top_k_documents.
           Returns 0.0 if top_k_documents is empty.
    """
    if not top_k_documents or not needed_corpus_ids:
        raise ValueError("Either top_k_documents or needed_corpus_ids is empty.")

    # Convert lists to sets for more efficient look-up
    needed_set = set(needed_corpus_ids)
    top_k_set = set(top_k_documents)

    # Calculate the intersection
    intersection = needed_set.intersection(top_k_set)

    # Calculate the proportion
    proportion = len(intersection) / min(len(top_k_documents), len(needed_corpus_ids))

    return proportion


def extract_doc_to_sen(text):
    """
    Extracts each document and its corresponding sentence numbers from the given text, returning a dictionary mapping.

    Args:
        text (str): Input text containing references in the format [Doc xx, Sen yy1, Sen yy2-yy3, ...] or [Doc xx, Sen yy1, yy2-yy3, ...].

    Returns:
        dict: A dictionary where keys are document identifiers (strings), and values are lists of sentence numbers (integers).
    
    Example:
        This is a statement inferred from multiple sentences [Doc ABC, Sen 2, 4-6, 8].
        Another statement from a different document [Doc XYZ, Sen 3].
        References to multiple documents [Doc ABC, Sen 10][Doc DEF, Sen 5-7].
        An original statement with no references.

        result = extract_doc_to_sen(sample_text)
        print(result)
        # Output:
        # {
        #     'ABC': [2, 4, 5, 6, 8, 10],
        #     'XYZ': [3],
        #     'DEF': [5, 6, 7]
        # }
    """
    # Updated regular expression to handle commas in Doc identifier
    pattern = r'\[Doc\s+(.+?)\s*,\s*Sen\s+([0-9]+(?:\s*-\s*[0-9]+)?(?:\s*,\s*(?:Sen\s+)?[0-9]+(?:\s*-\s*[0-9]+)?)*)\]'

    matches = re.findall(pattern, text)

    doc_to_sen = defaultdict(list)

    for doc_id, sen_part in matches:
        doc_id = doc_id.strip()  # Remove whitespace from both ends of the document identifier
        # Split sentence part by commas
        sen_items = [item.strip() for item in sen_part.split(',')]

        for item in sen_items:
            # Remove optional 'Sen' prefix
            if item.startswith('Sen'):
                item = item[3:].strip()
            
            if '-' in item:
                # Handle ranges, e.g., 5-8
                try:
                    start, end = map(int, item.split('-'))
                    if start > end:
                        raise ValueError(f"Sentence range start value {start} is greater than end value {end}.")
                    doc_to_sen[doc_id].extend(range(start, end + 1))
                except ValueError as ve:
                    print(f"Invalid sentence range '{item}' in document '{doc_id}': {ve}")
            else:
                # Handle single sentence numbers
                try:
                    sen_num = int(item)
                    doc_to_sen[doc_id].append(sen_num)
                except ValueError:
                    print(f"Invalid sentence number '{item}' in document '{doc_id}'.")

    # Convert defaultdict to a regular dictionary, remove duplicates and sort
    doc_to_sen_final = {doc: sorted(list(set(sens))) for doc, sens in doc_to_sen.items()}

    return doc_to_sen_final