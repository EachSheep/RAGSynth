import os
import json
import re
import copy
import argparse
from tqdm import tqdm
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed
from rag.utils.request_openai_utils import OpenAIModel

from rag.utils import (
    dcg,
    idcg,
    idcg_calculator_with_weight,
    idcg_calculator,
    is_val_in_top_k,
    are_all_elements_in_list,
    expand_numbers_and_ranges,
    extract_doc_to_sen,
    extract_and_remove_think_tags,
    cal_percentage_of_elements_in_list
)


CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

API_KEY = os.getenv("API_KEY", "None")
BASE_URL = os.getenv("BASE_URL", None)
if BASE_URL == None:
    CLIENT = OpenAI(api_key=API_KEY)
else:
    CLIENT = OpenAI(base_url=f"{BASE_URL}", api_key=API_KEY)

MODEL_NAME = os.getenv("MODEL_NAME", None)
if MODEL_NAME == None:
    raise EnvironmentError("MODEL_NAME environment variable is not set")
STOP_WORDS="------"
MAX_NEW_TOKENS="None"
openai_model = OpenAIModel(MODEL_NAME, STOP_WORDS, MAX_NEW_TOKENS)

def extract_consistency_with_sens_output(text):
    try:
        # Regular expressions to match reason, score, and total score
        reason_pattern = re.compile(r'<reason>(.*?)</reason>', re.DOTALL)
        score_pattern = re.compile(r'<judge>(.*?)</judge>')

        # Find all reasons and scores
        reasons = reason_pattern.findall(text)
        scores = score_pattern.findall(text)
        # Create a list of dictionaries for reasons and scores
        result = [{'reason': reason.strip(), 'score': 1} if "true" in score.lower() else {'reason': reason.strip(), 'score': 0} for reason, score in zip(reasons, scores)]

        total_score = sum([item['score'] for item in result])

        return result, total_score
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], 0

def convert_corpusid2senids_to_string(needed_corpusid2senids):
    # Create an empty list to store each line as a string
    lines = ["<Sentences>"]

    # Initialize a counter to keep track of sentence numbers
    sentence_number = 1

    # Iterate over each document and its corresponding list of sentence IDs in the dictionary
    for corpus_id, sen_ids in needed_corpusid2senids.items():
        # Iterate over each sentence ID
        for sen_id in sen_ids:
            # Format the string and add it to the list
            line = f"    <Sentence-{sentence_number}>[Doc {corpus_id}, Sen {sen_id}]</Sentence-{sentence_number}>"
            lines.append(line)
            # Increment the sentence counter
            sentence_number += 1

    # Close the tag
    lines.append("</Sentences>")

    # Join the lines in the list into a single string, adding a newline between each line
    result = "\n".join(lines)

    return result

def convert_corpusid2corpus_str(needed_corpusid2corpus):
    # Create an empty list to store each line as a string
    lines = ["<Documents>"]

    # Initialize a counter to keep track of document numbers
    document_number = 1

    # Iterate over each corpus ID and its corresponding text in the dictionary
    for corpus_id, text in needed_corpusid2corpus.items():
        # Add the opening tag for the document
        lines.append(f"    <Document-{document_number}>")
        # Add the document ID
        lines.append(f"        <Doc-Id>{corpus_id}</Doc-Id>")
        # Add the text (assuming it's already formatted with sentence markers)
        lines.append("        <Text>")
        lines.append(f"            {text}")
        lines.append("        </Text>")
        # Add the closing tag for the document
        lines.append(f"    </Document-{document_number}>")
        # Increment the document counter
        document_number += 1

    # Close the main tag
    lines.append("</Documents>")

    # Join the lines in the list into a single string, adding a newline between each line
    result = "\n".join(lines)

    return result

def judge_consistency_with_sens(question, documents, standard_sentences, generated_answer):
    cur_prompt_template = f"""Your task is to evaluate whether the Generated Answer includes the information expressed in each of the Sentences. The Generated Answer is a classmate's response to the Question (if the  Sentences can support answering the entire question) or a response to a specific aspect of the Question (if the Sentences only support a partial answer to the question). Assign `True` for each key point included from a sentence; if a key point is not included, assign `False`.

------
Example:

<Documents>
    <Document-1>
        <Doc-Id>acd_hkaho_bjfbl_pdf</Doc-Id>
        <Text>
            The sky is a vast expanse of blue, stretching endlessly above us, creating a serene and calming atmosphere [Sen 1]. On a clear day, the blue sky is often dotted with fluffy white clouds, adding depth and texture to the horizon [Sen 2]. Below this beautiful canopy, the grass spreads out like a lush green carpet, vibrant and full of life [Sen 3]. The green grass sways gently in the breeze, its color intensified by the sunlight that filters through the leaves [Sen 4]. Together, the blue sky and green grass create a harmonious landscape that is both peaceful and invigorating, inviting us to pause and appreciate the beauty of the natural world [Sen 5].
        </Text>
    </Document-1>
</Documents>

<Question>What color is the sky and the grass? Which color do Americans like? Why?</Question>

<Sentences>
    <Sentence-1>[Doc acd_hkaho_bjfbl_pdf, Sen 1]</Sentence-1>
    <Sentence-2>[Doc acd_hkaho_bjfbl_pdf, Sen 2]</Sentence-2>
<Sentences>

<Generated-Answer>I know that the sky is blue and the grass is green. However, I am not sure about the color that Americans like or the reason why the sky is blue.</Generated-Answer>

Judges:
1. <reason>The answer includes the color of the sky, corresponding to the Sentence-1 indicated by [Doc acd_hkaho_bjfbl_pdf, Sen 1].</reason>
<judge>True</judge>
2. <reason>The answer includes the color of the grass, corresponding to the Sentence-2 indicated by [Doc acd_hkaho_bjfbl_pdf, Sen 2].</reason>
<judge>True</judge>
------
You output in the following format. Do not output any other content.

1. <reason>...</reason>
<judge>...</judge>
2. ...
...
------

[[DOCUMENTS]]

<Question>[[QUESTION]]</Question>

[[STANDARD SENTENCES]]

<Generated-Answer>[[GENERATED ANSWER]]</Generated-Answer>

Begin! Please note that you need to provide [[NUM_SENTENCES]] reasons, with each reason corresponding to one sentence.
""" 
    cur_prompt = cur_prompt_template.replace("[[QUESTION]]", question)
    cur_prompt = cur_prompt.replace("[[DOCUMENTS]]", documents)
    cur_prompt = cur_prompt.replace("[[STANDARD SENTENCES]]", standard_sentences)
    cur_prompt = cur_prompt.replace("[[GENERATED ANSWER]]", generated_answer)
    cur_prompt = cur_prompt.replace("[[NUM_SENTENCES]]", str(len(standard_sentences.splitlines()) - 2))
    # print("generated_answer:", generated_answer)
    # print("str(len(standard_sentences.splitlines()) - 2):", str(len(standard_sentences.splitlines()) - 2))
    # input("Press Enter to continue...")

    generator_response, _ = openai_model.generate(CLIENT, cur_prompt)
    return generator_response

def judge_consistency_with_sens_2(question, documents, standard_sentences, generated_answer):
    cur_prompt_template = f"""Your task is to evaluate whether the Generated Answer includes the information expressed in each of the Sentences. The Generated Answer is a classmate's response to the Question (if the  Sentences can support answering the entire question) or a response to a specific aspect of the Question (if the Sentences only support a partial answer to the question). Assign `True` for each key point included from a sentence; if a key point is not included, assign `False`.

------
Example:

<Documents>
    <Document-1>
        <Doc-Id>acd_hkaho_bjfbl_pdf</Doc-Id>
        <Text>
            The sky is a vast expanse of blue, stretching endlessly above us, creating a serene and calming atmosphere [Sen 1]. On a clear day, the blue sky is often dotted with fluffy white clouds, adding depth and texture to the horizon [Sen 2]. Below this beautiful canopy, the grass spreads out like a lush green carpet, vibrant and full of life [Sen 3]. The green grass sways gently in the breeze, its color intensified by the sunlight that filters through the leaves [Sen 4]. Together, the blue sky and green grass create a harmonious landscape that is both peaceful and invigorating, inviting us to pause and appreciate the beauty of the natural world [Sen 5].
        </Text>
    </Document-1>
</Documents>

<Question>What color is the sky and the grass? Which color do Americans like? Why?</Question>

<Sentences>
    <Sentence-1>[Doc acd_hkaho_bjfbl_pdf, Sen 1]</Sentence-1>
    <Sentence-2>[Doc acd_hkaho_bjfbl_pdf, Sen 2]</Sentence-2>
<Sentences>

<Generated-Answer>I know that the sky is blue and the grass is green. However, I am not sure about the color that Americans like or the reason why the sky is blue.</Generated-Answer>

Judges:
1. <reason>The answer includes the color of the sky, corresponding to the Sentence-1 indicated by [Doc acd_hkaho_bjfbl_pdf, Sen 1].</reason>
<judge>True</judge>
2. <reason>The answer includes the color of the grass, corresponding to the Sentence-2 indicated by [Doc acd_hkaho_bjfbl_pdf, Sen 2].</reason>
<judge>True</judge>
------
You output in the following format. Do not output any other content.

1. <reason>...</reason>
<judge>...</judge>
2. ...
...
------

[[DOCUMENTS]]

<Question>[[QUESTION]]</Question>

[[STANDARD SENTENCES]]

<Generated-Answer>[[GENERATED ANSWER]]</Generated-Answer>

Begin! Please note that you need to provide [[NUM_SENTENCES]] reasons, with each reason corresponding to one sentence. You should strictly assess whether the answer aligns completely with the meaning of the sentence. If the answer fabricates facts not present in the sentence, the corresponding point should be marked as False.
""" 
    cur_prompt = cur_prompt_template.replace("[[QUESTION]]", question)
    cur_prompt = cur_prompt.replace("[[DOCUMENTS]]", documents)
    cur_prompt = cur_prompt.replace("[[STANDARD SENTENCES]]", standard_sentences)
    cur_prompt = cur_prompt.replace("[[GENERATED ANSWER]]", generated_answer)
    cur_prompt = cur_prompt.replace("[[NUM_SENTENCES]]", str(len(standard_sentences.splitlines()) - 2))
    # print("generated_answer:", generated_answer)
    # print("str(len(standard_sentences.splitlines()) - 2):", str(len(standard_sentences.splitlines()) - 2))
    # input("Press Enter to continue...")

    generator_response, _ = openai_model.generate(CLIENT, cur_prompt)
    return generator_response

def convert_to_rules_string(judge_criterias):
    """
    Convert a list of judge criteria into a formatted XML-like string.

    Args:
        judge_criterias (list of str): A list of criteria strings.

    Returns:
        str: A formatted string representing the criteria as numbered rules.
    """
    # Start the Rules section
    result = "<Rules>\n"
    
    # Iterate over the criteria and format each into a numbered rule
    for index, criteria in enumerate(judge_criterias, start=1):
        # Create a rule with the current index
        result += f"    <Rule-{index}>{criteria}</Rule-{index}>\n"
    
    # Close the Rules section
    result += "</Rules>"
    
    return result


def judge_dont_know_partthing(question, rules, generated_answer):
    cur_prompt_template = f"""Your task is to evaluate whether the Generated Answer admits that it don't know some part about the question according to the rules assigned to you.

------
Examples:

Example 1:
<Question>What color is the sky and the grass? Which color do Americans like? Why?</Question>
<Rules>
    <Rule-1>If the answer admits that it don't know the color do Americans like, assign 1 point.</Rule-1>
    <Rule-2>If the answer admits that it don't know the reason why the sky is blue, assign 1 point.</Rule-2>
</Rules>
<Generated-Answer>I know that the sky is blue and the grass is green. However, I am not sure about the color that Americans like or the reason why the sky is blue.</Generated-Answer>

Judges:
1. <reason>The answer admits that it don't know the color do Americans like according to the first rule.</reason>
<judge>True</judge>
2. <reason>Follwing the second rule, the answer admits that it don't know the reason why the sky is blue.</reason>
<judge>True</judge>

Example 2:
<Question>What color is the sky and the grass? Which color do Americans like? Why?</Question>
<Rules>
    <Rule-1>If the answer admits that it don't know the color do Americans like, assign 1 point.</Rule-1>
    <Rule-2>If the answer admits that it don't know the reason why the sky is blue, assign 1 point.</Rule-2>
</Rules>
<Generated-Answer>I know that the sky is blue and the grass is green. Apart from this, I don't know anything else.</Generated-Answer>

Judges:
1. <reason>The answer admits that it don't know the color do Americans like according to the first rule.</reason>
<judge>True</judge>
2. <reason>Follwing the second rule, the answer admits that it don't know the reason why the sky is blue.</reason>
<judge>True</judge>
------
You output in the following format. Do not output any other content.

1. <reason>...</reason>
<judge>...</judge>
2. ...
...
------

<Question>[[QUESTION]]</Question>
[[RULES]]

<Generated-Answer>[[GENERATED ANSWER]]</Generated-Answer>

Begin!
""" 
    # Begin! Note: For each rule given to you, judge if it's right. Please note that you should not make rigid judgments. Instead, evaluate like a teacher assessing a student's work. As long as the answer meets the conditions described in the grading criteria, points should be awarded. For example, if a grading criterion is "If the answer admits that it doesn't know the color Americans like, assign 1 point," and the answer is "I know that the sky is blue and the grass is green," it doesn't directly admit to not knowing the color Americans like, but since it doesn't mention "the color Americans like," points should be given. Similarly, if the answer is "I know that the sky is blue and the grass is green. I don't know any other things," the phrase "don't know any other things" implies "admits that it doesn't know the color Americans like," and therefore, points should be awarded.
    cur_prompt = cur_prompt_template.replace("[[QUESTION]]", question)
    cur_prompt = cur_prompt.replace("[[RULES]]", rules)
    cur_prompt = cur_prompt.replace("[[GENERATED ANSWER]]", generated_answer)

    generator_response, _ = openai_model.generate(CLIENT, cur_prompt)
    return generator_response


def judge_dont_know_anything(question, generated_answer):
    cur_prompt_template = f"""Your task is to evaluate whether the Generated Answer admits that it don't know anything about the question. The Generated Answer is a classmate's response to the Question.

------
Example:

<Question>What color is the sky and the grass? Which color do Americans like? Why?</Question>

<Generated-Answer>I am not sure about the color of the key and the grass. I also don't know what Americans like or the reason why the sky is blue.</Generated-Answer>

Judge:
<reason>The answer admits that it don't know anything about the question.</reason>
<judge>True</judge>
------
You output in the following format. Do not output any other content.

<reason>...</reason>
<judge>...</judge>
------

<Question>[[QUESTION]]</Question>

<Generated-Answer>[[GENERATED ANSWER]]</Generated-Answer>

Begin!
""" 
    # Begin! Note: For generated answer given to you, judge if it admits that it don't know anything about the question. Please note that you should not make rigid judgments. Instead, evaluate like a teacher assessing a student's work. As long as the answer conveys the meaning that he don't know anything, points should be awarded.
    cur_prompt = cur_prompt_template.replace("[[QUESTION]]", question)
    cur_prompt = cur_prompt.replace("[[GENERATED ANSWER]]", generated_answer)

    generator_response, _ = openai_model.generate(CLIENT, cur_prompt)
    return generator_response

def judge_dont_know_partthing_2(question, rules, generated_answer):
    cur_prompt_template = f"""Your task is to evaluate whether the Generated Answer admits that it don't know some part about the question according to the rules assigned to you.

------
Examples:

Example 1:
<Question>What color is the sky and the grass? Which color do Americans like? Why?</Question>
<Rules>
    <Rule-1>If the answer admits that it don't know the color do Americans like, assign 1 point.</Rule-1>
    <Rule-2>If the answer admits that it don't know the reason why the sky is blue, assign 1 point.</Rule-2>
</Rules>
<Generated-Answer>I know that the sky is blue and the grass is green. However, I am not sure about the color that Americans like or the reason why the sky is blue.</Generated-Answer>

Judges:
1. <reason>The answer admits that it don't know the color do Americans like according to the first rule.</reason>
<judge>True</judge>
2. <reason>Follwing the second rule, the answer admits that it don't know the reason why the sky is blue.</reason>
<judge>True</judge>

Example 2:
<Question>What color is the sky and the grass? Which color do Americans like? Why?</Question>
<Rules>
    <Rule-1>If the answer admits that it don't know the color do Americans like, assign 1 point.</Rule-1>
    <Rule-2>If the answer admits that it don't know the reason why the sky is blue, assign 1 point.</Rule-2>
</Rules>
<Generated-Answer>I know that the sky is blue and the grass is green. Apart from this, I don't know anything else.</Generated-Answer>

Judges:
1. <reason>The answer admits that it don't know the color do Americans like according to the first rule.</reason>
<judge>True</judge>
2. <reason>Follwing the second rule, the answer admits that it don't know the reason why the sky is blue.</reason>
<judge>True</judge>
------
You output in the following format. Do not output any other content.

1. <reason>...</reason>
<judge>...</judge>
2. ...
...
------

<Question>[[QUESTION]]</Question>
[[RULES]]

<Generated-Answer>[[GENERATED ANSWER]]</Generated-Answer>

Begin! The response must clearly state that it doesn't know certain parts of the question according to the rules. Vague answers should not receive points.
""" 
    # Begin! Note: For each rule given to you, judge if it's right. Please note that you should not make rigid judgments. Instead, evaluate like a teacher assessing a student's work. As long as the answer meets the conditions described in the grading criteria, points should be awarded. For example, if a grading criterion is "If the answer admits that it doesn't know the color Americans like, assign 1 point," and the answer is "I know that the sky is blue and the grass is green," it doesn't directly admit to not knowing the color Americans like, but since it doesn't mention "the color Americans like," points should be given. Similarly, if the answer is "I know that the sky is blue and the grass is green. I don't know any other things," the phrase "don't know any other things" implies "admits that it doesn't know the color Americans like," and therefore, points should be awarded.
    cur_prompt = cur_prompt_template.replace("[[QUESTION]]", question)
    cur_prompt = cur_prompt.replace("[[RULES]]", rules)
    cur_prompt = cur_prompt.replace("[[GENERATED ANSWER]]", generated_answer)

    generator_response, _ = openai_model.generate(CLIENT, cur_prompt)
    return generator_response


def judge_dont_know_anything_2(question, generated_answer):
    cur_prompt_template = f"""Your task is to evaluate whether the Generated Answer admits that it don't know anything about the question. The Generated Answer is a classmate's response to the Question.

------
Example:

<Question>What color is the sky and the grass? Which color do Americans like? Why?</Question>

<Generated-Answer>I am not sure about the color of the key and the grass. I also don't know what Americans like or the reason why the sky is blue.</Generated-Answer>

Judge:
<reason>The answer admits that it don't know anything about the question.</reason>
<judge>True</judge>
------
You output in the following format. Do not output any other content.

<reason>...</reason>
<judge>...</judge>
------

<Question>[[QUESTION]]</Question>

<Generated-Answer>[[GENERATED ANSWER]]</Generated-Answer>

Begin! The response must clearly state that it doesn't know anything. Vague answers is not allowable.
""" 
    # Begin! Note: For generated answer given to you, judge if it admits that it don't know anything about the question. Please note that you should not make rigid judgments. Instead, evaluate like a teacher assessing a student's work. As long as the answer conveys the meaning that he don't know anything, points should be awarded.
    cur_prompt = cur_prompt_template.replace("[[QUESTION]]", question)
    cur_prompt = cur_prompt.replace("[[GENERATED ANSWER]]", generated_answer)

    generator_response, _ = openai_model.generate(CLIENT, cur_prompt)
    return generator_response

def compute_list_operations_similar(dict1, dict2, different_keys=2):
    # Helper function: Determine if two keys are similar, with at most two character differences
    def are_keys_similar(k1, k2):
        if abs(len(k1) - len(k2)) > different_keys:
            return False
        differ = 0
        # Iterate over the characters of both keys and count the number of differing characters
        for a, b in zip(k1, k2):
            if a != b:
                differ += 1
                if differ > different_keys:
                    return False
        # Consider the case where the keys have different lengths
        differ += abs(len(k1) - len(k2))
        return differ <= different_keys

    intersection = {}
    similar_matches = {}       # Store the similar key matches (dict1_key: dict2_key)
    matched_dict2_keys = set() # Keys from dict2 that have been matched

    # Handle keys that match exactly
    for key in dict1:
        if key in dict2:
            common_elements = list(set(dict1[key]) & set(dict2[key]))
            if common_elements:
                intersection[key] = common_elements

    # Find keys from dict1 and dict2 that have not been matched
    dict1_keys_not_matched = set(dict1.keys()) - set(dict2.keys())
    dict2_keys_not_matched = set(dict2.keys()) - set(dict1.keys())

    # Attempt to match similar keys
    for k1 in dict1_keys_not_matched:
        for k2 in dict2_keys_not_matched:
            if k2 not in matched_dict2_keys and are_keys_similar(k1, k2):
                similar_matches[k1] = k2
                matched_dict2_keys.add(k2)
                # Calculate the intersection of similar matching keys
                common_elements = list(set(dict1[k1]) & set(dict2[k2]))
                if common_elements:
                    intersection[k1] = common_elements
                break  # Each k1 can match at most one k2

    # Calculate dict1 minus dict2
    dict1_minus_dict2 = {}
    # Keys from dict1 that did not pass similar matching
    for key in dict1_keys_not_matched - set(similar_matches.keys()):
        dict1_minus_dict2[key] = dict1[key]
    # Keys from dict1 that passed similar matching
    for k1, k2 in similar_matches.items():
        difference = list(set(dict1[k1]) - set(dict2[k2]))
        if difference:
            dict1_minus_dict2[k1] = difference

    # Calculate dict2 minus dict1
    dict2_minus_dict1 = {}
    # Keys from dict2 that did not pass similar matching
    for key in dict2_keys_not_matched - set(matched_dict2_keys):
        dict2_minus_dict1[key] = dict2[key]
    # Keys from dict2 that passed similar matching
    for k1, k2 in similar_matches.items():
        difference = list(set(dict2[k2]) - set(dict1[k1]))
        if difference:
            dict2_minus_dict1[k2] = difference

    # Calculate the length of each part
    intersection_length = sum(len(v) for v in intersection.values())
    dict1_minus_dict2_length = sum(len(v) for v in dict1_minus_dict2.values())
    dict2_minus_dict1_length = sum(len(v) for v in dict2_minus_dict1.values())

    return intersection_length, dict1_minus_dict2_length, dict2_minus_dict1_length

def compute_list_operations(dict1, dict2):
    intersection = {}
    for key in dict1:
        if key in dict2:
            common_elements = list(set(dict1[key]) & set(dict2[key]))
            if common_elements:
                intersection[key] = common_elements

    dict1_minus_dict2 = {}
    for key in dict1:
        if key not in dict2:
            dict1_minus_dict2[key] = dict1[key]
        else:
            difference = list(set(dict1[key]) - set(dict2[key]))
            if difference:
                dict1_minus_dict2[key] = difference

    dict2_minus_dict1 = {}
    for key in dict2:
        if key not in dict1:
            dict2_minus_dict1[key] = dict2[key]
        else:
            difference = list(set(dict2[key]) - set(dict1[key]))
            if difference:
                dict2_minus_dict1[key] = difference

    intersection_length = sum(len(v) for v in intersection.values())
    dict1_minus_dict2_length = sum(len(v) for v in dict1_minus_dict2.values())
    dict2_minus_dict1_length = sum(len(v) for v in dict2_minus_dict1.values())

    return intersection_length, dict1_minus_dict2_length, dict2_minus_dict1_length


def eval_generation_results_for_file_content(args, data, corpusid_2_context, max_process_num=300, max_workers=8, output_path=None):

    res_dicts = {}
    res_answer_key_names = []
    
    if max_process_num == -1:
        max_process_num = len(data)
    else:
        max_process_num = min(max_process_num, len(data))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_data = {}

        for cur_dict in data[:max_process_num]:

            if 'proposed-questions' not in cur_dict:
                continue
            proposed_questions = cur_dict['proposed-questions']

            """calculate generation metrics"""
            chunk_id = cur_dict['id'] # admission.stanford.edu.filter_index.htm.md_chunk_0
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                if "top_k_documents" not in proposed_question_dict:
                    continue
                top_k_documents = proposed_question_dict['top_k_documents']
                
                # needed_objective_fact_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-facts'].strip())
                # needed_objective_fact_ids = expand_numbers_and_ranges(needed_objective_fact_ids)
                needed_corpusids = [chunk_id]
                
                needed_corpusid2corpus = {chunk_id: corpusid_2_context[chunk_id]}
                
                # needed_corpusid2senids = {chunk_id: []}
                # for needed_objective_fact_id in needed_objective_fact_ids:
                #     if needed_objective_fact_id > len(cur_dict['sens']):
                #         continue
                #     cur_choosen_sens = re.findall(r'\d+-\d+|\d+', cur_dict['sens'][needed_objective_fact_id-1])
                #     cur_choosen_sens = expand_numbers_and_ranges(cur_choosen_sens)
                #     needed_corpusid2senids[chunk_id].extend(cur_choosen_sens)
                # needed_corpusid2senids[chunk_id] = list(set(needed_corpusid2senids[chunk_id]))
                positive = proposed_question_dict["positive"]
                if not positive:
                    continue
                needed_corpusid2senids = extract_doc_to_sen(positive)
                needed_corpusids = list(needed_corpusid2senids.keys())
                if chunk_id not in needed_corpusid2senids or not needed_corpusid2senids[chunk_id]:
                    continue

                # if not args.not_gen_for_original:

                #     original_question = proposed_question_dict['question']
                    
                #     # Reference Intersection Score
                #     answer_key_names = [key for key in proposed_question_dict.keys() if key not in ['answer', 'corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                #     res_answer_key_names.extend(answer_key_names)
                #     for answer_key_name in answer_key_names: # something like qwen2a5-72b-instruct-3-answer
                #         generated_answer = proposed_question_dict[answer_key_name]
                #         if not generated_answer:
                #             continue
                #         generated_reason, generated_answer = extract_and_remove_think_tags(generated_answer)
                        
                #         top_k_pattern = r"top_k_value-(\d+)"
                #         top_k_match = re.search(top_k_pattern, answer_key_name)
                #         cur_top_k_value = None
                #         if top_k_match:
                #             cur_top_k_value = int(top_k_match.group(1))
                #         all_in_top_k = None
                #         if cur_top_k_value:
                #             all_in_top_k = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                #         else:
                #             raise ValueError(f"Cannot find top_k_value in {answer_key_name}")

                #         generated_corpusid2senids = extract_doc_to_sen(generated_answer) # doc id to sen ids
                #         referencing_num_1, referencing_num_2, referencing_num_3 = compute_list_operations_similar(generated_corpusid2senids, needed_corpusid2senids)
                                                    
                #         if all_in_top_k:

                #             # referencing_recall_keyname = f"original-{answer_key_name}-referencing-recall"
                #             # if referencing_recall_keyname not in res_dicts:
                #             #     res_dicts[referencing_recall_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_3)]
                #             # else:
                #             #     res_dicts[referencing_recall_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_3))
                            
                #             # referencing_precision_keyname = f"original-{answer_key_name}-referencing-precision"
                #             # if referencing_precision_keyname not in res_dicts:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_2)]
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname] = [0]
                #             # else:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_2))
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname].append(0)
                            
                #             referencing_F1_keyname = f"original-{answer_key_name}-F1-score"
                #             if referencing_num_1 + referencing_num_3:
                #                 recall = referencing_num_1 / (referencing_num_1 + referencing_num_3)
                #             else:
                #                 recall = 0
                #             if referencing_num_1 + referencing_num_2:
                #                 precision = referencing_num_1 / (referencing_num_1 + referencing_num_2)
                #             else:
                #                 precision = 0
                #             if recall + precision:
                #                 referencing_F1_value = 2 * (recall * precision) / (recall + precision)
                #             else:
                #                 referencing_F1_value = 0
                #             if referencing_F1_keyname not in res_dicts:
                #                 res_dicts[referencing_F1_keyname] = [referencing_F1_value]
                #             else:
                #                 res_dicts[referencing_F1_keyname].append(referencing_F1_value)

                #             analysis_process_key_name = f"original-{answer_key_name}-consistency_with_sens2-reason"
                #             score_key_name = f"original-{answer_key_name}-consistency_with_sens2-score"
                #             total_score = sum([len(v) for v in needed_corpusid2senids.values()])
                #             if not args.only_eval_F1 and args.eval_answerable and (score_key_name not in proposed_question_dict or proposed_question_dict[score_key_name] == None ):
                #                 corpusid2corpus_str = convert_corpusid2corpus_str(needed_corpusid2corpus)
                #                 corpusid2senids_str = convert_corpusid2senids_to_string(needed_corpusid2senids)
                #                 future = executor.submit(judge_consistency_with_sens_2, original_question, corpusid2corpus_str, corpusid2senids_str, generated_answer)
                #                 futures_to_data[future] = (proposed_question_dict, answer_key_name, cur_top_k_value, total_score, 'consistency_with_sens2', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_answerable and score_key_name in proposed_question_dict and proposed_question_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     if total_score:
                #                         res_dicts[score_key_name] = [proposed_question_dict[score_key_name] / total_score]
                #                     else:
                #                         pass
                #                 else:
                #                     if total_score:
                #                         res_dicts[score_key_name].append(proposed_question_dict[score_key_name] / total_score)
                #                     else:
                #                         pass
                                
                #         else:
                #             
                #             analysis_process_key_name = f"original-{answer_key_name}-dont_know_anything-reason"
                #             score_key_name = f"original-{answer_key_name}-dont_know_anything-score"
                #             if not args.only_eval_F1 and args.eval_unanswerable and (score_key_name not in proposed_question_dict or proposed_question_dict[score_key_name] == None):
                #                 future = executor.submit(judge_dont_know_anything_2, original_question, generated_answer)
                #                 futures_to_data[future] = (proposed_question_dict, answer_key_name, cur_top_k_value, 1, 'dont_know_anything', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_unanswerable and score_key_name in proposed_question_dict and proposed_question_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     res_dicts[score_key_name] = [proposed_question_dict[score_key_name]]
                #                 else:
                #                     res_dicts[score_key_name].append(proposed_question_dict[score_key_name])
                        
                # tmp_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                # rephrased_questions = []
                # rephrased_questions_indexes = []
                # for only_eval_at_rephrased_pos in args.only_eval_at_rephrased_poses:
                #     if len(tmp_rephrased_questions) > only_eval_at_rephrased_pos:
                #         rephrased_questions.append(tmp_rephrased_questions[only_eval_at_rephrased_pos])
                #         rephrased_questions_indexes.append(only_eval_at_rephrased_pos + 1)
                # for rephrased_question_type, rephrased_question_dict in zip(rephrased_questions_indexes, rephrased_questions):
                #     if "top_k_documents" not in rephrased_question_dict:
                #         continue
                #     top_k_documents = rephrased_question_dict['top_k_documents']
                    
                #     if "reordered-question" in rephrased_question_dict:
                #         rephrased_question_str = rephrased_question_dict['reordered-question']
                #     else:
                #         rephrased_question_str = rephrased_question_dict['result']

                #     # Reference Intersection Score
                #     answer_key_names = [key for key in rephrased_question_dict.keys() if key not in ['answer', 'corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                #     res_answer_key_names.extend(answer_key_names)
                #     for answer_key_name in answer_key_names:
                #         generated_answer = rephrased_question_dict[answer_key_name]
                #         if not generated_answer:
                #             continue
                #         generated_reason, generated_answer = extract_and_remove_think_tags(generated_answer)
                        
                #         top_k_pattern = r"top_k_value-(\d+)"
                #         top_k_match = re.search(top_k_pattern, answer_key_name)
                #         cur_top_k_value = None
                #         if top_k_match:
                #             cur_top_k_value = int(top_k_match.group(1))
                #         all_in_top_k = None
                #         if cur_top_k_value:
                #             all_in_top_k = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                #         else:
                #             raise ValueError(f"Cannot find top_k_value in {answer_key_name}")

                #         generated_corpusid2senids = extract_doc_to_sen(generated_answer) # doc id to sen ids
                #         referencing_num_1, referencing_num_2, referencing_num_3 = compute_list_operations_similar(generated_corpusid2senids, needed_corpusid2senids)

                #         if all_in_top_k:
                #             # referencing_recall_keyname = f"rephrased-{answer_key_name}-referencing-recall"
                #             # if referencing_recall_keyname not in res_dicts:
                #             #     res_dicts[referencing_recall_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_3)]
                #             # else:
                #             #     res_dicts[referencing_recall_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_3))
                            
                #             # referencing_precision_keyname = f"rephrased-{answer_key_name}-referencing-precision"
                #             # if referencing_precision_keyname not in res_dicts:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_2)]
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname] = [0]
                #             # else:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_2))
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname].append(0)
                            
                #             referencing_F1_keyname = f"rephrased-{answer_key_name}-F1-score"
                #             if referencing_num_1 + referencing_num_3:
                #                 recall = referencing_num_1 / (referencing_num_1 + referencing_num_3)
                #             else:
                #                 recall = 0
                #             if referencing_num_1 + referencing_num_2:
                #                 precision = referencing_num_1 / (referencing_num_1 + referencing_num_2)
                #             else:
                #                 precision = 0
                #             if recall + precision:
                #                 referencing_F1_value = 2 * (recall * precision) / (recall + precision)
                #             else:
                #                 referencing_F1_value = 0
                #             if referencing_F1_keyname not in res_dicts:
                #                 res_dicts[referencing_F1_keyname] = [referencing_F1_value]
                #             else:
                #                 res_dicts[referencing_F1_keyname].append(referencing_F1_value)
                            
                #             analysis_process_key_name = f"rephrased-{answer_key_name}-consistency_with_sens2-reason"
                #             score_key_name = f"rephrased-{answer_key_name}-consistency_with_sens2-score"
                #             total_score = sum([len(v) for v in needed_corpusid2senids.values()])
                #             if not args.only_eval_F1 and args.eval_answerable and (score_key_name not in rephrased_question_dict or rephrased_question_dict[score_key_name] == None):
                #                 corpusid2corpus_str = convert_corpusid2corpus_str(needed_corpusid2corpus)
                #                 corpusid2senids_str = convert_corpusid2senids_to_string(needed_corpusid2senids)
                #                 future = executor.submit(judge_consistency_with_sens_2, rephrased_question_str, corpusid2corpus_str, corpusid2senids_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_dict, answer_key_name, cur_top_k_value, total_score, 'consistency_with_sens2', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_answerable and score_key_name in rephrased_question_dict and rephrased_question_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     if total_score:
                #                         res_dicts[score_key_name] = [rephrased_question_dict[score_key_name] / total_score]
                #                     else:
                #                         pass
                #                 else:
                #                     if total_score:
                #                         res_dicts[score_key_name].append(rephrased_question_dict[score_key_name] / total_score)
                #                     else:
                #                         pass

                #         else:
                            
                #             analysis_process_key_name = f"rephrased-{answer_key_name}-dont_know_anything-reason"
                #             score_key_name = f"rephrased-{answer_key_name}-dont_know_anything-score"
                #             if not args.only_eval_F1 and args.eval_unanswerable and (score_key_name not in rephrased_question_dict or rephrased_question_dict[score_key_name] == None):
                #                 future = executor.submit(judge_dont_know_anything_2, rephrased_question_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_dict, answer_key_name, cur_top_k_value, 1, 'dont_know_anything', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_unanswerable and score_key_name in rephrased_question_dict and rephrased_question_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     res_dicts[score_key_name] = [rephrased_question_dict[score_key_name]]
                #                 else:
                #                     res_dicts[score_key_name].append(rephrased_question_dict[score_key_name])

                # tmp_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                # rephrased_questions_part = []
                # rephrased_questions_indexes_part = []
                # for only_eval_at_rephrased_pos_part in args.only_eval_at_rephrased_poses_part:
                #     if len(tmp_rephrased_questions_part) > only_eval_at_rephrased_pos_part:
                #         rephrased_questions_part.append(tmp_rephrased_questions_part[only_eval_at_rephrased_pos_part])
                #         rephrased_questions_indexes_part.append(only_eval_at_rephrased_pos_part + 1)
                # for rephrased_question_part_type, rephrased_question_part_dict in zip(rephrased_questions_indexes_part, rephrased_questions_part):
                #     if "top_k_documents" not in rephrased_question_part_dict:
                #         continue
                #     top_k_documents = rephrased_question_part_dict['top_k_documents']

                #     if "reordered-question" in rephrased_question_part_dict:
                #         rephrased_question_part_str = rephrased_question_part_dict['reordered-question']
                #     else:
                #         rephrased_question_part_str = rephrased_question_part_dict['result']

                #     # Reference Intersection Score
                #     answer_key_names = [key for key in rephrased_question_part_dict.keys() if key not in ['answer', 'corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                #     res_answer_key_names.extend(answer_key_names)
                #     for answer_key_name in answer_key_names:
                #         generated_answer = rephrased_question_part_dict[answer_key_name]
                #         if not generated_answer:
                #             continue
                #         generated_reason, generated_answer = extract_and_remove_think_tags(generated_answer)
                        
                #         top_k_pattern = r"top_k_value-(\d+)"
                #         top_k_match = re.search(top_k_pattern, answer_key_name)
                #         cur_top_k_value = None
                #         if top_k_match:
                #             cur_top_k_value = int(top_k_match.group(1))
                #         all_in_top_k = None
                #         if cur_top_k_value:
                #             all_in_top_k = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                #         else:
                #             raise ValueError(f"Cannot find top_k_value in {answer_key_name}")
                        
                #         generated_corpusid2senids = extract_doc_to_sen(generated_answer)
                #         referencing_num_1, referencing_num_2, referencing_num_3 = compute_list_operations_similar(generated_corpusid2senids, needed_corpusid2senids)

                #         if all_in_top_k:
                #             # referencing_recall_keyname = f"rephrased_part-{answer_key_name}-referencing-recall"
                #             # if referencing_recall_keyname not in res_dicts:
                #             #     res_dicts[referencing_recall_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_3)]
                #             # else:
                #             #     res_dicts[referencing_recall_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_3))
                            
                #             # referencing_precision_keyname = f"rephrased_part-{answer_key_name}-referencing-precision"
                #             # if referencing_precision_keyname not in res_dicts:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_2)]
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname] = [0]
                #             # else:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_2))
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname].append(0)
                            
                #             referencing_F1_keyname = f"rephrased_part-{answer_key_name}-F1-score"
                #             if referencing_num_1 + referencing_num_3:
                #                 recall = referencing_num_1 / (referencing_num_1 + referencing_num_3)
                #             else:
                #                 recall = 0
                #             if referencing_num_1 + referencing_num_2:
                #                 precision = referencing_num_1 / (referencing_num_1 + referencing_num_2)
                #             else:
                #                 precision = 0
                #             if recall + precision:
                #                 referencing_F1_value = 2 * (recall * precision) / (recall + precision)
                #             else:
                #                 referencing_F1_value = 0
                #             if referencing_F1_keyname not in res_dicts:
                #                 res_dicts[referencing_F1_keyname] = [referencing_F1_value]
                #             else:
                #                 res_dicts[referencing_F1_keyname].append(referencing_F1_value)

                #             analysis_process_key_name = f"rephrased_part-{answer_key_name}-consistency_with_sens2-reason"
                #             score_key_name = f"rephrased_part-{answer_key_name}-consistency_with_sens2-score"
                #             total_score = sum([len(v) for v in needed_corpusid2senids.values()])
                #             if not args.only_eval_F1 and args.eval_answerable and (score_key_name not in rephrased_question_part_dict or rephrased_question_part_dict[score_key_name] == None):
                #                 corpusid2corpus_str = convert_corpusid2corpus_str(needed_corpusid2corpus)
                #                 corpusid2senids_str = convert_corpusid2senids_to_string(needed_corpusid2senids)
                #                 future = executor.submit(judge_consistency_with_sens_2, rephrased_question_part_str, corpusid2corpus_str, corpusid2senids_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_part_dict, answer_key_name, cur_top_k_value, total_score, 'consistency_with_sens2', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_answerable and score_key_name in rephrased_question_part_dict and rephrased_question_part_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     if total_score:
                #                         res_dicts[score_key_name] = [rephrased_question_part_dict[score_key_name] / total_score]
                #                     else:
                #                         pass
                #                 else:
                #                     if total_score:
                #                         res_dicts[score_key_name].append(rephrased_question_part_dict[score_key_name] / total_score)
                #                     else:
                #                         pass

                #             analysis_process_key_name = f"rephrased_part-{answer_key_name}-dont_know_partthing-reason"
                #             score_key_name = f"rephrased_part-{answer_key_name}-dont_know_partthing-score"
                #             judge_criterias = []
                #             for tmp_rephrased_question_part in tmp_rephrased_questions_part[:rephrased_question_part_type]:
                #                 cur_criteria = tmp_rephrased_question_part['scoring-criteria']
                #                 # print("cur_criteria:", cur_criteria)
                #                 # input("Press Enter to continue...")
                #                 judge_criterias.append(cur_criteria)
                #             judge_criterias_str = convert_to_rules_string(judge_criterias)
                #             total_score = len(judge_criterias)
                #             if not args.only_eval_F1 and args.eval_partanswerable and (score_key_name not in rephrased_question_part_dict or rephrased_question_part_dict[score_key_name] == None):
                #                 future = executor.submit(judge_dont_know_partthing_2, rephrased_question_part_str, judge_criterias_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_part_dict, answer_key_name, cur_top_k_value, total_score, 'dont_know_partthing', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_partanswerable and score_key_name in rephrased_question_part_dict and rephrased_question_part_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     if total_score:
                #                         res_dicts[score_key_name] = [rephrased_question_part_dict[score_key_name] / total_score]
                #                     else:
                #                         pass
                #                 else:
                #                     if total_score:
                #                         res_dicts[score_key_name].append(rephrased_question_part_dict[score_key_name] / total_score)
                #                     else:
                #                         pass
                #                 # print("question:", rephrased_question_part_str)
                #                 # print("generated_answer:", generated_answer)
                #                 # print("judge_criterias:", judge_criterias_str)
                #                 # print("rephrased_question_part_dict[score_key_name]:", rephrased_question_part_dict[score_key_name])
                #                 # print("total_score:", total_score)
                #                 # print()
                #                 # input("Press Enter to continue...")
                #         else:
                            
                #             analysis_process_key_name = f"rephrased_part-{answer_key_name}-dont_know_anything-reason"
                #             score_key_name = f"rephrased_part-{answer_key_name}-dont_know_anything-score"
                #             if not args.only_eval_F1 and args.eval_unanswerable and (score_key_name not in rephrased_question_part_dict or rephrased_question_part_dict[score_key_name] == None):
                #                 future = executor.submit(judge_dont_know_anything_2, rephrased_question_part_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_part_dict, answer_key_name, cur_top_k_value, 1, 'dont_know_anything', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_unanswerable and score_key_name in rephrased_question_part_dict and rephrased_question_part_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     res_dicts[score_key_name] = [rephrased_question_part_dict[score_key_name]]
                #                 else:
                #                     res_dicts[score_key_name].append(rephrased_question_part_dict[score_key_name])

                tmp_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                rephrased_questions_hybrid = []
                rephrased_questions_indexes_hybrid = []
                for only_eval_at_rephrased_pos_hybrid in args.only_eval_at_rephrased_poses_hybrid:
                    if len(tmp_rephrased_questions_hybrid) > only_eval_at_rephrased_pos_hybrid:
                        rephrased_questions_hybrid.append(tmp_rephrased_questions_hybrid[only_eval_at_rephrased_pos_hybrid])
                        rephrased_questions_indexes_hybrid.append(only_eval_at_rephrased_pos_hybrid + 1)
                for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in zip(rephrased_questions_indexes_hybrid, rephrased_questions_hybrid):
                    if "top_k_documents" not in rephrased_question_hybrid_dict:
                        continue
                    top_k_documents = rephrased_question_hybrid_dict['top_k_documents']

                    if "reordered-question" in rephrased_question_hybrid_dict:
                        rephrased_question_hybrid_str = rephrased_question_hybrid_dict['reordered-question']
                    else:
                        rephrased_question_hybrid_str = rephrased_question_hybrid_dict['result']

                    # Reference Intersection Score
                    answer_key_names = [key for key in rephrased_question_hybrid_dict.keys() if key not in ['answer', 'corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                    # answer_key_names = [key for key in rephrased_question_hybrid_dict.keys() if key not in ['corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                    res_answer_key_names.extend(answer_key_names)
                    for answer_key_name in answer_key_names:
                        generated_answer = rephrased_question_hybrid_dict[answer_key_name]
                        if not generated_answer:
                            continue
                        generated_reason, generated_answer = extract_and_remove_think_tags(generated_answer)
                            
                        top_k_pattern = r"top_k_value-(\d+)"
                        top_k_match = re.search(top_k_pattern, answer_key_name)
                        cur_top_k_value = None
                        if top_k_match:
                            cur_top_k_value = int(top_k_match.group(1))
                        all_in_top_k = None
                        if cur_top_k_value:
                            all_in_top_k = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                        else:
                            raise ValueError(f"Cannot find top_k_value in {answer_key_name}")
                        
                        # print("generated_answer:", generated_answer)
                        generated_corpusid2senids = extract_doc_to_sen(generated_answer)
                        # print("generated_corpusid2senids:", generated_corpusid2senids)
                        referencing_num_1, referencing_num_2, referencing_num_3 = compute_list_operations_similar(generated_corpusid2senids, needed_corpusid2senids)
                        # print("referencing_num:", referencing_num_1, referencing_num_2, referencing_num_3)

                        if all_in_top_k:
                            # referencing_recall_keyname = f"rephrased_hybrid-{answer_key_name}-referencing-recall"
                            # if referencing_recall_keyname not in res_dicts:
                            #     res_dicts[referencing_recall_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_3)]
                            # else:
                            #     res_dicts[referencing_recall_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_3))
                            
                            # referencing_precision_keyname = f"rephrased_hybrid-{answer_key_name}-referencing-precision"
                            # if referencing_precision_keyname not in res_dicts:
                            #     if referencing_num_1 + referencing_num_2:
                            #         res_dicts[referencing_precision_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_2)]
                            #     else:
                            #         res_dicts[referencing_precision_keyname] = [0]
                            # else:
                            #     if referencing_num_1 + referencing_num_2:
                            #         res_dicts[referencing_precision_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_2))
                            #     else:
                            #         res_dicts[referencing_precision_keyname].append(0)
                            
                            referencing_F1_keyname = f"rephrased_hybrid-{answer_key_name}-F1-score"
                            if referencing_num_1 + referencing_num_3:
                                recall = referencing_num_1 / (referencing_num_1 + referencing_num_3)
                            else:
                                recall = 0
                            if referencing_num_1 + referencing_num_2:
                                precision = referencing_num_1 / (referencing_num_1 + referencing_num_2)
                            else:
                                precision = 0
                            if recall + precision:
                                referencing_F1_value = 2 * (recall * precision) / (recall + precision)
                            else:
                                referencing_F1_value = 0
                            # print("referencing_F1_value:", referencing_F1_value)
                            if referencing_F1_keyname not in res_dicts:
                                res_dicts[referencing_F1_keyname] = [referencing_F1_value]
                            else:
                                res_dicts[referencing_F1_keyname].append(referencing_F1_value)
                            # input()
                            
                            analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-consistency_with_sens2-reason"
                            score_key_name = f"rephrased_hybrid-{answer_key_name}-consistency_with_sens2-score"
                            total_score = sum([len(v) for v in needed_corpusid2senids.values()])
                            if not args.only_eval_F1 and args.eval_answerable and (score_key_name not in rephrased_question_hybrid_dict or rephrased_question_hybrid_dict[score_key_name] == None):
                                corpusid2corpus_str = convert_corpusid2corpus_str(needed_corpusid2corpus)
                                corpusid2senids_str = convert_corpusid2senids_to_string(needed_corpusid2senids)
                                future = executor.submit(judge_consistency_with_sens_2, rephrased_question_hybrid_str, corpusid2corpus_str, corpusid2senids_str, generated_answer)
                                futures_to_data[future] = (rephrased_question_hybrid_dict, answer_key_name, cur_top_k_value, total_score, 'consistency_with_sens2', analysis_process_key_name, score_key_name)
                            elif not args.only_eval_F1 and args.eval_answerable and score_key_name in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict[score_key_name] != None:
                                if score_key_name not in res_dicts:
                                    if total_score:
                                        res_dicts[score_key_name] = [rephrased_question_hybrid_dict[score_key_name] / total_score]
                                    else:
                                        pass
                                else:
                                    if total_score:
                                        res_dicts[score_key_name].append(rephrased_question_hybrid_dict[score_key_name] / total_score)
                                    else:
                                        pass

                            # analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_partthing-reason"
                            # score_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_partthing-score"
                            analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_partthing2-reason"
                            score_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_partthing2-score"
                            judge_criterias = []
                            for tmp_rephrased_question_hybrid in tmp_rephrased_questions_hybrid[:rephrased_question_hybrid_type]:
                                transformation_type = tmp_rephrased_question_hybrid['transformation']
                                if 'Partial Transformation' in transformation_type:
                                    cur_criteria = tmp_rephrased_question_hybrid['scoring-criteria']
                                    judge_criterias.append(cur_criteria)
                            if judge_criterias:
                                judge_criterias_str = convert_to_rules_string(judge_criterias)
                                total_score = len(judge_criterias)
                            if not args.only_eval_F1 and args.eval_partanswerable and (score_key_name not in rephrased_question_hybrid_dict or rephrased_question_hybrid_dict[score_key_name] == None):
                                    if judge_criterias:
                                        # future = executor.submit(judge_dont_know_partthing_2, rephrased_question_hybrid_str, judge_criterias_str, generated_answer)
                                        future = executor.submit(judge_dont_know_partthing_2, rephrased_question_hybrid_str, judge_criterias_str, generated_answer)
                                        futures_to_data[future] = (rephrased_question_hybrid_dict, answer_key_name, cur_top_k_value, total_score, 'dont_know_partthing', analysis_process_key_name, score_key_name)
                            elif not args.only_eval_F1 and args.eval_partanswerable and score_key_name in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict[score_key_name] != None:
                                if score_key_name not in res_dicts:
                                    if total_score:
                                        res_dicts[score_key_name] = [rephrased_question_hybrid_dict[score_key_name] / total_score]
                                    else:
                                        pass
                                else:
                                    if total_score:
                                        res_dicts[score_key_name].append(rephrased_question_hybrid_dict[score_key_name] / total_score)
                                    else:
                                        pass
                        else:

                            # analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_anything-reason"
                            # score_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_anything-score"
                            analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_anything2-reason"
                            score_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_anything2-score"
                            
                            # print("doc1:", corpusid_2_context[top_k_documents[0]])
                            # print("-" * 100)
                            # print("doc2:", corpusid_2_context[top_k_documents[1]])
                            # print("-" * 100)
                            # print("doc3:", corpusid_2_context[top_k_documents[2]])
                            # print("-" * 100)
                            # print("rephrased_question_hybrid_str:", rephrased_question_hybrid_str)
                            # print("-" * 100)
                            # print("answer:", rephrased_question_hybrid_dict["answer"])
                            # print("-" * 100)
                            # print("generated_answer:", generated_answer)
                            # print("-" * 100)
                            
                            if not args.only_eval_F1 and args.eval_unanswerable and (score_key_name not in rephrased_question_hybrid_dict or rephrased_question_hybrid_dict[score_key_name] == None):
                                # future = executor.submit(judge_dont_know_anything_2, rephrased_question_hybrid_str, generated_answer)
                                future = executor.submit(judge_dont_know_anything_2, rephrased_question_hybrid_str, generated_answer)
                                futures_to_data[future] = (rephrased_question_hybrid_dict, answer_key_name, cur_top_k_value, 1, 'dont_know_anything', analysis_process_key_name, score_key_name)
                            elif not args.only_eval_F1 and args.eval_unanswerable and score_key_name in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict[score_key_name] != None:
                                if score_key_name not in res_dicts:
                                    res_dicts[score_key_name] = [rephrased_question_hybrid_dict[score_key_name]]
                                else:
                                    res_dicts[score_key_name].append(rephrased_question_hybrid_dict[score_key_name])
                            # print(rephrased_question_hybrid_dict[score_key_name])
                            # input()

        new_gen_num = 0
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
            proposed_question_dict, answer_key_name, cur_top_k_value, total_score, score_type, analysis_process_key_name, score_key_name = futures_to_data[future]
            try:
                score_response = future.result(timeout=5*60)
                if score_type in ['consistency_with_sens2']:
                    analysis_process, score = extract_consistency_with_sens_output(score_response)
                    proposed_question_dict[analysis_process_key_name] = analysis_process
                    proposed_question_dict[score_key_name] = score
                elif score_type == 'dont_know_partthing':
                    analysis_process, score = extract_consistency_with_sens_output(score_response)
                    proposed_question_dict[analysis_process_key_name] = analysis_process
                    proposed_question_dict[score_key_name] = score
                elif score_type in ['dont_know_anything']:
                    analysis_process, score = extract_consistency_with_sens_output(score_response)
                    proposed_question_dict[analysis_process_key_name] = analysis_process
                    proposed_question_dict[score_key_name] = score
                    
                else:
                    raise ValueError(f"Unknown score_type: {score_type}")

                if score_key_name not in res_dicts:
                    if score != None:
                        if total_score:
                            res_dicts[score_key_name] = [score / total_score]
                        else:
                            pass
                else:
                    if score != None:
                        if total_score:
                            res_dicts[score_key_name].append(score / total_score)
                        else:
                            pass

                new_gen_num += 1
                if (new_gen_num + 1) % args.save_interval == 0:
                    if output_path:
                        print(f"Saving results to {output_path}")
                        with open(output_path, 'w') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"Processed {new_gen_num} scoring tasks.")

            except Exception as e:
                print(f"Error processing {score_type} for answer_key_name {answer_key_name}: {e}")
                continue

        if output_path and (new_gen_num or not os.path.exists(output_path)):
            print(f"Saving results to {output_path}")
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Processed {new_gen_num} scoring tasks.")

    res_answer_key_names = list(set(res_answer_key_names))
    rephrased_types = ["original", "rephrased", "rephrased_part", "rephrased_hybrid"]
    num_part_and_any = {}
    for rephrased_type in rephrased_types:
        for answer_key_name in res_answer_key_names:
            dont_key_names = [
                f"{rephrased_type}-{answer_key_name}-dont_know_anything2-score",
                f"{rephrased_type}-{answer_key_name}-dont_know_partthing2-score"
            ]
            dont_key_name = f"{rephrased_type}-{answer_key_name}-dont2-score"
            for cur_dont_key_name in dont_key_names:
                if cur_dont_key_name in res_dicts:                    
                    if dont_key_name not in res_dicts:
                        res_dicts[dont_key_name] = copy.deepcopy(res_dicts[cur_dont_key_name])
                    else:
                        res_dicts[dont_key_name].extend(res_dicts[cur_dont_key_name])
                    num_part_and_any[cur_dont_key_name] = num_part_and_any.get(cur_dont_key_name, 0) + len(res_dicts[cur_dont_key_name])
    
    ratio_dicts = {}
    for rephrased_type in rephrased_types:
        for answer_key_name in res_answer_key_names:
            part_key_name = f"{rephrased_type}-{answer_key_name}-dont_know_partthing2-score"
            any_key_name = f"{rephrased_type}-{answer_key_name}-dont_know_anything2-score"
            dont_key_name = f"{rephrased_type}-{answer_key_name}-dont2-score"
            
            if part_key_name in num_part_and_any and any_key_name in num_part_and_any:
                fenzi = num_part_and_any[part_key_name]
                fenmu = num_part_and_any[any_key_name] + num_part_and_any[part_key_name]
                ratio_dicts[dont_key_name] = f"{fenzi / fenmu * 100:.2f} = {fenzi} / {fenmu}"
                
    metric_dicts = {}
    for key, value in res_dicts.items():
        metric_dicts[key] = f"{sum(value) / len(value) * 100:.2f}"
        if key in ratio_dicts:
            metric_dicts[key] += f" (part / all: {ratio_dicts[key]})"
    
    metric_dicts = dict(sorted(metric_dicts.items(), key=lambda x: x[0]))

    return metric_dicts

def eval_retrieval_results_for_file_content(args, data, max_process_num=300, max_workers=8):

    res_dicts = {}

    if max_process_num == -1:
        max_process_num = len(data)
    else:
        max_process_num = min(max_process_num, len(data))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for cur_dict in data[:max_process_num]:
            """calculate precision and nDCG
            """
            if 'proposed-questions' not in cur_dict:
                continue
            
            proposed_questions = cur_dict['proposed-questions']
            chunk_id = cur_dict['id'] # admission.stanford.edu.filter_index.htm.md_chunk_0
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                if 'top_k_documents' not in proposed_question_dict:
                    continue
                top_k_documents = proposed_question_dict['top_k_documents']

                tmp_result = is_val_in_top_k(top_k_documents, chunk_id, args.top_k_values)
                for cur_top_k_value in args.top_k_values:
                    all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                    proposed_question_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]
                
                # precision
                for cur_top_k_value in args.top_k_values:
                    precision_key_name = f"Precision@{cur_top_k_value}; full; transformation@{0}"
                    if precision_key_name not in res_dicts:
                        res_dicts[precision_key_name] = [tmp_result[cur_top_k_value]]
                    else:
                        res_dicts[precision_key_name].append(tmp_result[cur_top_k_value])
                    
                    precision_key_name = f"Precision@{cur_top_k_value}; part; transformation@{0}"
                    if precision_key_name not in res_dicts:
                        res_dicts[precision_key_name] = [tmp_result[cur_top_k_value]]
                    else:
                        res_dicts[precision_key_name].append(tmp_result[cur_top_k_value])

                    precision_key_name = f"Precision@{cur_top_k_value}; hybrid; transformation@{0}"
                    if precision_key_name not in res_dicts:
                        res_dicts[precision_key_name] = [tmp_result[cur_top_k_value]]
                    else:
                        res_dicts[precision_key_name].append(tmp_result[cur_top_k_value])
                
                # precision for rephrased questions
                rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions, start=1):
                    if "top_k_documents" not in rephrased_question_dict:
                        continue
                    top_k_documents = rephrased_question_dict['top_k_documents']
                    tmp_result = is_val_in_top_k(top_k_documents, chunk_id, args.top_k_values)
                    
                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]
                    
                    for cur_top_k_value in args.top_k_values:
                        precision_key_name = f"Precision@{cur_top_k_value}; full; transformation@{rephrased_question_type}"
                        if precision_key_name not in res_dicts:
                            res_dicts[precision_key_name] = [tmp_result[cur_top_k_value]]
                        else:
                            res_dicts[precision_key_name].append(tmp_result[cur_top_k_value])
                
                # precision for rephrased questions part
                rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                for rephrased_question_part_type, rephrased_question_part_dict in enumerate(rephrased_questions_part,start=1):
                    if "top_k_documents" not in rephrased_question_part_dict:
                        continue
                    top_k_documents = rephrased_question_part_dict['top_k_documents']
                    tmp_result = is_val_in_top_k(top_k_documents, chunk_id, args.top_k_values)

                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_part_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]
                    
                    for cur_top_k_value in args.top_k_values:
                        precision_key_name = f"Precision@{cur_top_k_value}; part; transformation@{rephrased_question_part_type}"
                        if precision_key_name not in res_dicts:
                            res_dicts[precision_key_name] = [tmp_result[cur_top_k_value]]
                        else:
                            res_dicts[precision_key_name].append(tmp_result[cur_top_k_value])
                
                # precision for rephrased questions hybrid
                rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid,start=1):
                    if "top_k_documents" not in rephrased_question_hybrid_dict:
                        continue
                    top_k_documents = rephrased_question_hybrid_dict['top_k_documents']
                    tmp_result = is_val_in_top_k(top_k_documents, chunk_id, args.top_k_values)

                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_hybrid_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]

                    for cur_top_k_value in args.top_k_values:
                        precision_key_name = f"Precision@{cur_top_k_value}; hybrid; transformation@{rephrased_question_hybrid_type}"
                        if precision_key_name not in res_dicts:
                            res_dicts[precision_key_name] = [tmp_result[cur_top_k_value]]
                        else:
                            res_dicts[precision_key_name].append(tmp_result[cur_top_k_value])

                # # nDCG
                # needed_corpusids = [chunk_id]
                # for cur_top_k_value in args.top_k_values:
                #     all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                #     proposed_question_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                # # grouped_sums = idcg_calculator_with_weight(args.difference_alpha, proposed_question_dict, needed_corpusids, top_k_documents)
                # grouped_sums = idcg_calculator(needed_corpusids)
                
                # relevances = [grouped_sums[cur_doc_id] if cur_doc_id in needed_corpusids else 0 for cur_doc_id in top_k_documents]
                # idea_relevances = list(sorted(grouped_sums.values(), reverse=True)) + [0] * (len(top_k_documents) - len(grouped_sums))
                # for cur_top_k_value in args.top_k_values:
                #     tmp_dcg = dcg(relevances, cur_top_k_value)
                #     tmp_idcg = idcg(idea_relevances, cur_top_k_value)
                #     tmp_ndcg = tmp_dcg / tmp_idcg if tmp_idcg != 0 else 0.0

                #     nDCG_key_name = f"nDCG@{cur_top_k_value}; full; transformation@{0}"
                #     if nDCG_key_name not in res_dicts:
                #         res_dicts[nDCG_key_name] = [tmp_ndcg]
                #     else:
                #         res_dicts[nDCG_key_name].append(tmp_ndcg)
                    
                #     nDCG_key_name = f"nDCG@{cur_top_k_value}; part; transformation@{0}"
                #     if nDCG_key_name not in res_dicts:
                #         res_dicts[nDCG_key_name] = [tmp_ndcg]
                #     else:
                #         res_dicts[nDCG_key_name].append(tmp_ndcg)

                #     nDCG_key_name = f"nDCG@{cur_top_k_value}; hybrid; transformation@{0}"
                #     if nDCG_key_name not in res_dicts:
                #         res_dicts[nDCG_key_name] = [tmp_ndcg]
                #     else:
                #         res_dicts[nDCG_key_name].append(tmp_ndcg)
                
                # # nDCG for rephrased questions
                # rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                # for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions):
                #     if "top_k_documents" not in rephrased_question_dict:
                #         continue
                #     top_k_documents = rephrased_question_dict['top_k_documents']
                #     tmp_result = is_val_in_top_k(top_k_documents, chunk_id, args.top_k_values)

                #     for cur_top_k_value in args.top_k_values:
                #         all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                #         rephrased_question_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]
                    
                #     relevances = [grouped_sums[cur_doc_id] if cur_doc_id in needed_corpusids else 0 for cur_doc_id in top_k_documents]
                #     idea_relevances = list(sorted(grouped_sums.values(), reverse=True)) + [0] * (len(top_k_documents) - len(grouped_sums))
                #     for cur_top_k_value in args.top_k_values:
                #         tmp_dcg = dcg(relevances, cur_top_k_value)
                #         tmp_idcg = idcg(idea_relevances, cur_top_k_value)
                #         tmp_ndcg = tmp_dcg / tmp_idcg if tmp_idcg != 0 else 0.0
                #         nDCG_key_name = f"nDCG@{cur_top_k_value}; full; transformation@{rephrased_question_type+1}"
                #         if nDCG_key_name not in res_dicts:
                #             res_dicts[nDCG_key_name] = [tmp_ndcg]
                #         else:
                #             res_dicts[nDCG_key_name].append(tmp_ndcg)
            
                # # nDCG for rephrased questions part
                # rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                # for rephrased_question_part_type, rephrased_question_part_dict in enumerate(rephrased_questions_part):
                #     if "top_k_documents" not in rephrased_question_part_dict:
                #         continue
                #     top_k_documents = rephrased_question_part_dict['top_k_documents']
                #     tmp_result = is_val_in_top_k(top_k_documents, chunk_id, args.top_k_values)

                #     for cur_top_k_value in args.top_k_values:
                #         all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                #         rephrased_question_part_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]
                    
                #     relevances = [grouped_sums[cur_doc_id] if cur_doc_id in needed_corpusids else 0 for cur_doc_id in top_k_documents]
                #     idea_relevances = list(sorted(grouped_sums.values(), reverse=True)) + [0] * (len(top_k_documents) - len(grouped_sums))
                #     for cur_top_k_value in args.top_k_values:
                #         tmp_dcg = dcg(relevances, cur_top_k_value)
                #         tmp_idcg = idcg(idea_relevances, cur_top_k_value)
                #         tmp_ndcg = tmp_dcg / tmp_idcg if tmp_idcg != 0 else 0.0
                #         nDCG_key_name = f"nDCG@{cur_top_k_value}; part; transformation@{rephrased_question_part_type+1}"
                #         if nDCG_key_name not in res_dicts:
                #             res_dicts[nDCG_key_name] = [tmp_ndcg]
                #         else:
                #             res_dicts[nDCG_key_name].append(tmp_ndcg)
                
                # # nDCG for rephrased questions hybrid
                # rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                # for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid):
                #     if "top_k_documents" not in rephrased_question_hybrid_dict:
                #         continue
                #     top_k_documents = rephrased_question_hybrid_dict['top_k_documents']
                #     tmp_result = is_val_in_top_k(top_k_documents, chunk_id, args.top_k_values)

                #     for cur_top_k_value in args.top_k_values:
                #         all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                #         rephrased_question_hybrid_dict[all_in_top_k_key_name] = tmp_result[cur_top_k_value]
                    
                #     relevances = [grouped_sums[cur_doc_id] if cur_doc_id in needed_corpusids else 0 for cur_doc_id in top_k_documents]
                #     idea_relevances = list(sorted(grouped_sums.values(), reverse=True)) + [0] * (len(top_k_documents) - len(grouped_sums))
                #     for cur_top_k_value in args.top_k_values:
                #         tmp_dcg = dcg(relevances, cur_top_k_value)
                #         tmp_idcg = idcg(idea_relevances, cur_top_k_value)
                #         tmp_ndcg = tmp_dcg / tmp_idcg if tmp_idcg != 0 else 0.0
                #         nDCG_key_name = f"nDCG@{cur_top_k_value}; hybrid; transformation@{rephrased_question_hybrid_type+1}"
                #         if nDCG_key_name not in res_dicts:
                #             res_dicts[nDCG_key_name] = [tmp_ndcg]
                #         else:
                #             res_dicts[nDCG_key_name].append(tmp_ndcg)

    metric_dicts = {}
    for key, value in res_dicts.items():
        metric_dicts[key] = f"{sum(value) / len(value) * 100:.2f}"
    
    metric_dicts = dict(sorted(metric_dicts.items(), key=lambda x: x[0]))

    return metric_dicts

def eval_generation_results_for_file_entity_graph(args, data, corpusid_2_context, max_process_num=300, max_workers=8, output_path=None):

    res_dicts = {}
    res_answer_key_names = []

    if max_process_num == -1:
        max_process_num = len(data)
    else:
        max_process_num = min(max_process_num, len(data))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_data = {}

        for entity_dict in list(data.values())[:max_process_num]:
            if 'proposed-questions' not in entity_dict:
                continue
            
            proposed_questions = entity_dict['proposed-questions']
            objective_relationships = entity_dict['selected-relationships']['objective-relationships']
            objective_relationship_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}

            """calculate generation metrics"""
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                if "top_k_documents" not in proposed_question_dict:
                    continue
                top_k_documents = proposed_question_dict['top_k_documents']

                # needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'])
                # needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                # needed_related_relationships = [objective_relationship_id_2_objective_relationship[int(clue_id)] for clue_id in needed_objective_relationship_ids if clue_id and int(clue_id) in objective_relationship_id_2_objective_relationship]
                # needed_corpusids = [cur_related_relationship['id'] for cur_related_relationship in needed_related_relationships if 'id' in cur_related_relationship]
                # needed_corpusids = list(sorted(list(set(needed_corpusids))))
                # needed_corpusid2corpus = {
                #     cur_related_relationship['id']: corpusid_2_context[cur_related_relationship['id']]
                #     for cur_related_relationship in needed_related_relationships if 'id' in cur_related_relationship and cur_related_relationship['id'] in corpusid_2_context
                # } 

                # needed_corpusid2senids = {} # doc id to sens
                # for cur_related_relationship in needed_related_relationships:
                #     sentences_used = re.findall(r'\d+-\d+|\d+', cur_related_relationship['sentences_used'])
                #     sentences_used = expand_numbers_and_ranges(sentences_used)
                #     if not sentences_used:
                #         continue
                #     needed_corpusid2senids[cur_related_relationship['id']] = sentences_used
                positive = proposed_question_dict["positive"]
                needed_corpusid2senids = extract_doc_to_sen(positive)
                needed_corpusids = list(needed_corpusid2senids.keys())
                needed_corpusid2corpus = {
                    cur_corpusid: corpusid_2_context[cur_corpusid]
                    for cur_corpusid in needed_corpusids if cur_corpusid in corpusid_2_context
                }
                if not needed_corpusid2corpus:
                    continue
                
                # if not args.not_gen_for_original:

                #     original_question = proposed_question_dict['question']

                #     # Referencing Intersection Score
                #     answer_key_names = [key for key in proposed_question_dict.keys() if key not in ['answer', 'corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                #     res_answer_key_names.extend(answer_key_names)
                #     for answer_key_name in answer_key_names:
                #         generated_answer = proposed_question_dict[answer_key_name]
                #         if not generated_answer:
                #             continue
                #         generated_reason, generated_answer = extract_and_remove_think_tags(generated_answer)

                #         top_k_pattern = r"top_k_value-(\d+)"
                #         top_k_match = re.search(top_k_pattern, answer_key_name)
                #         cur_top_k_value = None
                #         if top_k_match:
                #             cur_top_k_value = int(top_k_match.group(1))
                #         all_in_top_k = None
                #         if cur_top_k_value:
                #             all_in_top_k = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                #         else:
                #             raise ValueError(f"Cannot find top_k_value in {answer_key_name}")
                        
                #         generated_corpusid2senids = extract_doc_to_sen(generated_answer)

                #         cur_needed_corpusid2corpus = {k: v for k, v in needed_corpusid2corpus.items() if k in top_k_documents[:cur_top_k_value]}
                #         cur_needed_corpusid2senids = {k: v for k, v in needed_corpusid2senids.items() if k in top_k_documents[:cur_top_k_value]}
                #         referencing_num_1, referencing_num_2, referencing_num_3 = compute_list_operations_similar(generated_corpusid2senids, cur_needed_corpusid2senids)

                #         if all_in_top_k:

                #             # referencing_recall_keyname = f"original-{answer_key_name}-referencing-recall"
                #             # if referencing_recall_keyname not in res_dicts:
                #             #     res_dicts[referencing_recall_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_3)]
                #             # else:
                #             #     res_dicts[referencing_recall_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_3))
                            
                #             # referencing_precision_keyname = f"original-{answer_key_name}-referencing-precision"
                #             # if referencing_precision_keyname not in res_dicts:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_2)]
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname] = [0]
                #             # else:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_2))
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname].append(0)
                            
                #             referencing_F1_keyname = f"original-{answer_key_name}-F1-score"
                #             if referencing_num_1 + referencing_num_3:
                #                 recall = referencing_num_1 / (referencing_num_1 + referencing_num_3)
                #             else:
                #                 recall = 0
                #             if referencing_num_1 + referencing_num_2:
                #                 precision = referencing_num_1 / (referencing_num_1 + referencing_num_2)
                #             else:
                #                 precision = 0
                #             if recall + precision:
                #                 referencing_F1_value = 2 * (recall * precision) / (recall + precision)
                #             else:
                #                 referencing_F1_value = 0
                #             if referencing_F1_keyname not in res_dicts:
                #                 res_dicts[referencing_F1_keyname] = [referencing_F1_value]
                #             else:
                #                 res_dicts[referencing_F1_keyname].append(referencing_F1_value)
                            
                #             analysis_process_key_name = f"original-{answer_key_name}-consistency_with_sens2-reason"
                #             score_key_name = f"original-{answer_key_name}-consistency_with_sens2-score"
                #             total_score = sum([len(v) for v in cur_needed_corpusid2senids.values()])
                #             if not args.only_eval_F1 and args.eval_answerable and (score_key_name not in proposed_question_dict or proposed_question_dict[score_key_name] == None):
                #                 needed_corpusid2corpus_str = convert_corpusid2corpus_str(cur_needed_corpusid2corpus)
                #                 needed_corpusid2senids_str = convert_corpusid2senids_to_string(cur_needed_corpusid2senids)
                #                 future = executor.submit(judge_consistency_with_sens_2, original_question, needed_corpusid2corpus_str, needed_corpusid2senids_str, generated_answer)
                #                 futures_to_data[future] = (proposed_question_dict, answer_key_name, cur_top_k_value, total_score, 'consistency_with_sens2', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_answerable and score_key_name in proposed_question_dict and proposed_question_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     if total_score:
                #                         res_dicts[score_key_name] = [proposed_question_dict[score_key_name] / total_score]
                #                     else:
                #                         pass
                #                 else:
                #                     if total_score:
                #                         res_dicts[score_key_name].append(proposed_question_dict[score_key_name] / total_score)
                #                     else:
                #                         pass

                #             # omit partial transformation
                #         else:
                            
                #             analysis_process_key_name = f"original-{answer_key_name}-dont_know_anything-reason"
                #             score_key_name = f"original-{answer_key_name}-dont_know_anything-score"
                #             if not args.only_eval_F1 and args.eval_unanswerable and (score_key_name not in proposed_question_dict or proposed_question_dict[score_key_name] == None):
                #                 future = executor.submit(judge_dont_know_anything_2, original_question, generated_answer)
                #                 futures_to_data[future] = (proposed_question_dict, answer_key_name, cur_top_k_value, 1, 'dont_know_anything', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_unanswerable and score_key_name in proposed_question_dict and proposed_question_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     res_dicts[score_key_name] = [proposed_question_dict[score_key_name]]
                #                 else:
                #                     res_dicts[score_key_name].append(proposed_question_dict[score_key_name])
                        

                # tmp_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                # rephrased_questions = []
                # rephrased_questions_indexes = []
                # for only_eval_at_rephrased_pos in args.only_eval_at_rephrased_poses:
                #     if len(tmp_rephrased_questions) > only_eval_at_rephrased_pos:
                #         rephrased_questions.append(tmp_rephrased_questions[only_eval_at_rephrased_pos])
                #         rephrased_questions_indexes.append(only_eval_at_rephrased_pos + 1)
                # for rephrased_question_type, rephrased_question_dict in zip(rephrased_questions_indexes, rephrased_questions):
                #     if "top_k_documents" not in rephrased_question_dict:
                #         continue
                #     top_k_documents = rephrased_question_dict['top_k_documents']

                #     if "reordered-question" in rephrased_question_dict:
                #         rephrased_question_str = rephrased_question_dict['reordered-question']
                #     else:
                #         rephrased_question_str = rephrased_question_dict['result']

                #     # Referencing Intersection Score
                #     answer_key_names = [key for key in rephrased_question_dict.keys() if key not in ['answer', 'corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                #     res_answer_key_names.extend(answer_key_names)
                #     for answer_key_name in answer_key_names:
                #         generated_answer = rephrased_question_dict[answer_key_name]
                #         if not generated_answer:
                #             continue
                #         generated_reason, generated_answer = extract_and_remove_think_tags(generated_answer)
                        
                #         top_k_pattern = r"top_k_value-(\d+)"
                #         top_k_match = re.search(top_k_pattern, answer_key_name)
                #         cur_top_k_value = None
                #         if top_k_match:
                #             cur_top_k_value = int(top_k_match.group(1))
                #         all_in_top_k = None
                #         if cur_top_k_value:
                #             all_in_top_k = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                #         else:
                #             raise ValueError(f"Cannot find top_k_value in {answer_key_name}")

                #         generated_corpusid2senids = extract_doc_to_sen(generated_answer)

                #         cur_needed_corpusid2corpus = {k: v for k, v in needed_corpusid2corpus.items() if k in top_k_documents[:cur_top_k_value]}
                #         cur_needed_corpusid2senids = {k: v for k, v in needed_corpusid2senids.items() if k in top_k_documents[:cur_top_k_value]}
                #         referencing_num_1, referencing_num_2, referencing_num_3 = compute_list_operations_similar(generated_corpusid2senids, cur_needed_corpusid2senids)

                #         if all_in_top_k:

                #             # referencing_recall_keyname = f"rephrased-{answer_key_name}-referencing-recall"
                #             # if referencing_recall_keyname not in res_dicts:
                #             #     res_dicts[referencing_recall_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_3)]
                #             # else:
                #             #     res_dicts[referencing_recall_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_3))
                            
                #             # referencing_precision_keyname = f"rephrased-{answer_key_name}-referencing-precision"
                #             # if referencing_precision_keyname not in res_dicts:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_2)]
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname] = [0]
                #             # else:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_2))
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname].append(0)
                            
                #             referencing_F1_keyname = f"rephrased-{answer_key_name}-F1-score"
                #             if referencing_num_1 + referencing_num_3:
                #                 recall = referencing_num_1 / (referencing_num_1 + referencing_num_3)
                #             else:
                #                 recall = 0
                #             if referencing_num_1 + referencing_num_2:
                #                 precision = referencing_num_1 / (referencing_num_1 + referencing_num_2)
                #             else:
                #                 precision = 0
                #             if recall + precision:
                #                 referencing_F1_value = 2 * (recall * precision) / (recall + precision)
                #             else:
                #                 referencing_F1_value = 0
                #             if referencing_F1_keyname not in res_dicts:
                #                 res_dicts[referencing_F1_keyname] = [referencing_F1_value]
                #             else:
                #                 res_dicts[referencing_F1_keyname].append(referencing_F1_value)

                #             analysis_process_key_name = f"rephrased-{answer_key_name}-consistency_with_sens2-reason"
                #             score_key_name = f"rephrased-{answer_key_name}-consistency_with_sens2-score"
                #             total_score = sum([len(v) for v in cur_needed_corpusid2senids.values()])
                #             if not args.only_eval_F1 and args.eval_answerable and (score_key_name not in rephrased_question_dict or rephrased_question_dict[score_key_name] == None):
                #                 needed_corpusid2corpus_str = convert_corpusid2corpus_str(cur_needed_corpusid2corpus)
                #                 needed_corpusid2senids_str = convert_corpusid2senids_to_string(cur_needed_corpusid2senids)
                #                 future = executor.submit(judge_consistency_with_sens_2, rephrased_question_str, needed_corpusid2corpus_str, needed_corpusid2senids_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_dict, answer_key_name, cur_top_k_value, total_score, 'consistency_with_sens2', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_answerable and score_key_name in rephrased_question_dict and rephrased_question_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     if total_score:
                #                         res_dicts[score_key_name] = [rephrased_question_dict[score_key_name] / total_score]
                #                     else:
                #                         pass
                #                 else:
                #                     if total_score:
                #                         res_dicts[score_key_name].append(rephrased_question_dict[score_key_name] / total_score)
                #                     else:
                #                         pass

                #         else:
                            
                #             analysis_process_key_name = f"rephrased-{answer_key_name}-dont_know_anything-reason"
                #             score_key_name = f"rephrased-{answer_key_name}-dont_know_anything-score"
                #             if not args.only_eval_F1 and args.eval_unanswerable and (score_key_name not in rephrased_question_dict or rephrased_question_dict[score_key_name] == None):
                #                 future = executor.submit(judge_dont_know_anything_2, rephrased_question_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_dict, answer_key_name, cur_top_k_value, 1, 'dont_know_anything', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_unanswerable and score_key_name in rephrased_question_dict and rephrased_question_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     res_dicts[score_key_name] = [rephrased_question_dict[score_key_name]]
                #                 else:
                #                     res_dicts[score_key_name].append(rephrased_question_dict[score_key_name])

                # tmp_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                # rephrased_questions_part = []
                # rephrased_questions_indexes_part = []
                # for only_eval_at_rephrased_pos_part in args.only_eval_at_rephrased_poses_part:
                #     if len(tmp_rephrased_questions_part) > only_eval_at_rephrased_pos_part:
                #         rephrased_questions_part.append(tmp_rephrased_questions_part[only_eval_at_rephrased_pos_part])
                #         rephrased_questions_indexes_part.append(only_eval_at_rephrased_pos_part + 1)
                # for rephrased_question_part_type, rephrased_question_part_dict in zip(rephrased_questions_indexes_part, rephrased_questions_part):
                #     if "top_k_documents" not in rephrased_question_part_dict:
                #         continue
                #     top_k_documents = rephrased_question_part_dict['top_k_documents']

                #     if "reordered-question" in rephrased_question_part_dict:
                #         rephrased_question_part_str = rephrased_question_part_dict['reordered-question']
                #     else:
                #         rephrased_question_part_str = rephrased_question_part_dict['result']

                #     # Referencing Intersection Score
                #     answer_key_names = [key for key in rephrased_question_part_dict.keys() if key not in ['answer', 'corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                #     res_answer_key_names.extend(answer_key_names)
                #     for answer_key_name in answer_key_names:
                #         generated_answer = rephrased_question_part_dict[answer_key_name]
                #         if not generated_answer:
                #             continue
                #         generated_reason, generated_answer = extract_and_remove_think_tags(generated_answer)

                #         top_k_pattern = r"top_k_value-(\d+)"
                #         top_k_match = re.search(top_k_pattern, answer_key_name)
                #         cur_top_k_value = None
                #         if top_k_match:
                #             cur_top_k_value = int(top_k_match.group(1))
                #         all_in_top_k = None
                #         if cur_top_k_value:
                #             all_in_top_k = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                #         else:
                #             raise ValueError(f"Cannot find top_k_value in {answer_key_name}")
                        
                #         generated_corpusid2senids = extract_doc_to_sen(generated_answer)

                #         cur_needed_corpusid2corpus = {k: v for k, v in needed_corpusid2corpus.items() if k in top_k_documents[:cur_top_k_value]}
                #         cur_needed_corpusid2senids = {k: v for k, v in needed_corpusid2senids.items() if k in top_k_documents[:cur_top_k_value]}
                #         referencing_num_1, referencing_num_2, referencing_num_3 = compute_list_operations_similar(generated_corpusid2senids, cur_needed_corpusid2senids)

                #         if all_in_top_k:

                #             # referencing_recall_keyname = f"rephrased_part-{answer_key_name}-referencing-recall"
                #             # if referencing_recall_keyname not in res_dicts:
                #             #     res_dicts[referencing_recall_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_3)]
                #             # else:
                #             #     res_dicts[referencing_recall_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_3))
                            
                #             # referencing_precision_keyname = f"rephrased_part-{answer_key_name}-referencing-precision"
                #             # if referencing_precision_keyname not in res_dicts:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_2)]
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname] = [0]
                #             # else:
                #             #     if referencing_num_1 + referencing_num_2:
                #             #         res_dicts[referencing_precision_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_2))
                #             #     else:
                #             #         res_dicts[referencing_precision_keyname].append(0)
                            
                #             referencing_F1_keyname = f"rephrased_part-{answer_key_name}-F1-score"
                #             if referencing_num_1 + referencing_num_3:
                #                 recall = referencing_num_1 / (referencing_num_1 + referencing_num_3)
                #             else:
                #                 recall = 0
                #             if referencing_num_1 + referencing_num_2:
                #                 precision = referencing_num_1 / (referencing_num_1 + referencing_num_2)
                #             else:
                #                 precision = 0
                #             if recall + precision:
                #                 referencing_F1_value = 2 * (recall * precision) / (recall + precision)
                #             else:
                #                 referencing_F1_value = 0
                #             if referencing_F1_keyname not in res_dicts:
                #                 res_dicts[referencing_F1_keyname] = [referencing_F1_value]
                #             else:
                #                 res_dicts[referencing_F1_keyname].append(referencing_F1_value)

                #             analysis_process_key_name = f"rephrased_part-{answer_key_name}-consistency_with_sens2-reason"
                #             score_key_name = f"rephrased_part-{answer_key_name}-consistency_with_sens2-score"
                #             needed_corpusid2corpus_str = convert_corpusid2corpus_str(cur_needed_corpusid2corpus)
                #             needed_corpusid2senids_str = convert_corpusid2senids_to_string(cur_needed_corpusid2senids)
                #             total_score = sum([len(v) for v in cur_needed_corpusid2senids.values()])
                #             if not args.only_eval_F1 and args.eval_answerable and (score_key_name not in rephrased_question_part_dict or rephrased_question_part_dict[score_key_name] == None):
                #                 future = executor.submit(judge_consistency_with_sens_2, rephrased_question_part_str, needed_corpusid2corpus_str, needed_corpusid2senids_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_part_dict, answer_key_name, cur_top_k_value, total_score, 'consistency_with_sens2', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_answerable and score_key_name in rephrased_question_part_dict and rephrased_question_part_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     if total_score:
                #                         res_dicts[score_key_name] = [rephrased_question_part_dict[score_key_name] / total_score]
                #                     else:
                #                         pass
                #                 else:
                #                     if total_score:
                #                         res_dicts[score_key_name].append(rephrased_question_part_dict[score_key_name] / total_score)
                #                     else:
                #                         pass

                #             analysis_process_key_name = f"rephrased_part-{answer_key_name}-dont_know_partthing-reason"
                #             score_key_name = f"rephrased_part-{answer_key_name}-dont_know_partthing-score"
                #             judge_criterias = []
                #             for tmp_rephrased_question_part in tmp_rephrased_questions_part[:rephrased_question_part_type]:
                #                 cur_criteria = tmp_rephrased_question_part['scoring-criteria']
                #                 judge_criterias.append(cur_criteria)
                #             judge_criterias_str = convert_to_rules_string(judge_criterias)
                #             total_score = len(judge_criterias)
                #             if not args.only_eval_F1 and args.eval_partanswerable and (score_key_name not in rephrased_question_part_dict or rephrased_question_part_dict[score_key_name] == None):
                #                 future = executor.submit(judge_dont_know_partthing_2, rephrased_question_part_str, judge_criterias_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_part_dict, answer_key_name, cur_top_k_value, total_score, 'dont_know_partthing', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_partanswerable and score_key_name in rephrased_question_part_dict and rephrased_question_part_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     if total_score:
                #                         res_dicts[score_key_name] = [rephrased_question_part_dict[score_key_name] / total_score]
                #                     else:
                #                         pass
                #                 else:
                #                     if total_score:
                #                         res_dicts[score_key_name].append(rephrased_question_part_dict[score_key_name] / total_score)
                #                     else:
                #                         pass

                #         else:
                            
                #             analysis_process_key_name = f"rephrased_part-{answer_key_name}-dont_know_anything-reason"
                #             score_key_name = f"rephrased_part-{answer_key_name}-dont_know_anything-score"
                #             if not args.only_eval_F1 and args.eval_unanswerable and (score_key_name not in rephrased_question_part_dict or rephrased_question_part_dict[score_key_name] == None):
                #                 future = executor.submit(judge_dont_know_anything_2, rephrased_question_part_str, generated_answer)
                #                 futures_to_data[future] = (rephrased_question_part_dict, answer_key_name, cur_top_k_value, 1, 'dont_know_anything', analysis_process_key_name, score_key_name)
                #             elif not args.only_eval_F1 and args.eval_unanswerable and score_key_name in rephrased_question_part_dict and rephrased_question_part_dict[score_key_name] != None:
                #                 if score_key_name not in res_dicts:
                #                     res_dicts[score_key_name] = [rephrased_question_part_dict[score_key_name]]
                #                 else:
                #                     res_dicts[score_key_name].append(rephrased_question_part_dict[score_key_name])
                
                tmp_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                rephrased_questions_hybrid = []
                rephrased_questions_indexes_hybrid = []
                for only_eval_at_rephrased_pos_hybrid in args.only_eval_at_rephrased_poses_hybrid:
                    if len(tmp_rephrased_questions_hybrid) > only_eval_at_rephrased_pos_hybrid:
                        rephrased_questions_hybrid.append(tmp_rephrased_questions_hybrid[only_eval_at_rephrased_pos_hybrid])
                        rephrased_questions_indexes_hybrid.append(only_eval_at_rephrased_pos_hybrid + 1)
                for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in zip(rephrased_questions_indexes_hybrid, rephrased_questions_hybrid):
                    if "top_k_documents" not in rephrased_question_hybrid_dict:
                        continue
                    top_k_documents = rephrased_question_hybrid_dict['top_k_documents']
                    
                    if "reordered-question" in rephrased_question_hybrid_dict:
                        rephrased_question_hybrid_str = rephrased_question_hybrid_dict['reordered-question']
                    else:
                        rephrased_question_hybrid_str = rephrased_question_hybrid_dict['result']

                    # Referencing Intersection Score
                    answer_key_names = [key for key in rephrased_question_hybrid_dict.keys() if key not in ['answer', 'corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                    # answer_key_names = [key for key in rephrased_question_hybrid_dict.keys() if key not in ['corrected-answer'] and 'answer' in key and 'score' not in key and 'reason' not in key]
                    res_answer_key_names.extend(answer_key_names)
                    for answer_key_name in answer_key_names:
                        generated_answer = rephrased_question_hybrid_dict[answer_key_name]
                        if not generated_answer:
                            continue
                        generated_reason, generated_answer = extract_and_remove_think_tags(generated_answer)

                        top_k_pattern = r"top_k_value-(\d+)"
                        top_k_match = re.search(top_k_pattern, answer_key_name)
                        cur_top_k_value = None
                        if top_k_match:
                            cur_top_k_value = int(top_k_match.group(1))
                        all_in_top_k = None
                        if cur_top_k_value:
                            all_in_top_k = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                        else:
                            raise ValueError(f"Cannot find top_k_value in {answer_key_name}")    
                        
                        generated_corpusid2senids = extract_doc_to_sen(generated_answer)

                        cur_needed_corpusid2corpus = {k: v for k, v in needed_corpusid2corpus.items() if k in top_k_documents[:cur_top_k_value]}
                        cur_needed_corpusid2senids = {k: v for k, v in needed_corpusid2senids.items() if k in top_k_documents[:cur_top_k_value]}
                        referencing_num_1, referencing_num_2, referencing_num_3 = compute_list_operations_similar(generated_corpusid2senids, cur_needed_corpusid2senids)

                        if all_in_top_k:

                            # referencing_recall_keyname = f"rephrased_hybrid-{answer_key_name}-referencing-recall"
                            # if referencing_recall_keyname not in res_dicts:
                            #     res_dicts[referencing_recall_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_3)]
                            # else:
                            #     res_dicts[referencing_recall_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_3))
                            
                            # referencing_precision_keyname = f"rephrased_hybrid-{answer_key_name}-referencing-precision"
                            # if referencing_precision_keyname not in res_dicts:
                            #     if referencing_num_1 + referencing_num_2:
                            #         res_dicts[referencing_precision_keyname] =  [referencing_num_1 / (referencing_num_1 + referencing_num_2)]
                            #     else:
                            #         res_dicts[referencing_precision_keyname] = [0]
                            # else:
                            #     if referencing_num_1 + referencing_num_2:
                            #         res_dicts[referencing_precision_keyname].append(referencing_num_1 / (referencing_num_1 + referencing_num_2))
                            #     else:
                            #         res_dicts[referencing_precision_keyname].append(0)
                            
                            referencing_F1_keyname = f"rephrased_hybrid-{answer_key_name}-F1-score"
                            if referencing_num_1 + referencing_num_3:
                                recall = referencing_num_1 / (referencing_num_1 + referencing_num_3)
                            else:
                                recall = 0
                            if referencing_num_1 + referencing_num_2:
                                precision = referencing_num_1 / (referencing_num_1 + referencing_num_2)
                            else:
                                precision = 0
                            if recall + precision:
                                referencing_F1_value = 2 * (recall * precision) / (recall + precision)
                            else:
                                referencing_F1_value = 0
                            if referencing_F1_keyname not in res_dicts:
                                res_dicts[referencing_F1_keyname] = [referencing_F1_value]
                            else:
                                res_dicts[referencing_F1_keyname].append(referencing_F1_value)

                            analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-consistency_with_sens2-reason"
                            score_key_name = f"rephrased_hybrid-{answer_key_name}-consistency_with_sens2-score"
                            total_score = sum([len(v) for v in cur_needed_corpusid2senids.values()])
                            if not args.only_eval_F1 and args.eval_answerable and (score_key_name not in rephrased_question_hybrid_dict or rephrased_question_hybrid_dict[score_key_name] == None):
                                needed_corpusid2corpus_str = convert_corpusid2corpus_str(cur_needed_corpusid2corpus)
                                needed_corpusid2senids_str = convert_corpusid2senids_to_string(cur_needed_corpusid2senids)
                                future = executor.submit(judge_consistency_with_sens_2, rephrased_question_hybrid_str, needed_corpusid2corpus_str, needed_corpusid2senids_str, generated_answer)
                                futures_to_data[future] = (rephrased_question_hybrid_dict, answer_key_name, cur_top_k_value, total_score, 'consistency_with_sens2', analysis_process_key_name, score_key_name)
                            elif not args.only_eval_F1 and args.eval_answerable and score_key_name in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict[score_key_name] != None:
                                if score_key_name not in res_dicts:
                                    if total_score:
                                        res_dicts[score_key_name] = [rephrased_question_hybrid_dict[score_key_name] / total_score]
                                    else:
                                        pass
                                else:
                                    if total_score:
                                        res_dicts[score_key_name].append(rephrased_question_hybrid_dict[score_key_name] / total_score)
                                    else:
                                        pass

                            # analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_partthing-reason"
                            # score_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_partthing-score"
                            analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_partthing2-reason"
                            score_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_partthing2-score"
                            judge_criterias = []
                            for tmp_rephrased_question_hybrid in tmp_rephrased_questions_hybrid[:rephrased_question_hybrid_type]:
                                transformation_type = tmp_rephrased_question_hybrid['transformation']
                                if 'Partial Transformation' in transformation_type:
                                    cur_criteria = tmp_rephrased_question_hybrid['scoring-criteria']
                                    judge_criterias.append(cur_criteria)
                            if judge_criterias:
                                judge_criterias_str = convert_to_rules_string(judge_criterias)
                                total_score = len(judge_criterias)
                            if not args.only_eval_F1 and args.eval_partanswerable and (score_key_name not in rephrased_question_hybrid_dict or rephrased_question_hybrid_dict[score_key_name] == None):
                                if judge_criterias:
                                    # future = executor.submit(judge_dont_know_partthing_2, rephrased_question_hybrid_str, judge_criterias_str, generated_answer)
                                    future = executor.submit(judge_dont_know_partthing_2, rephrased_question_hybrid_str, judge_criterias_str, generated_answer)
                                    futures_to_data[future] = (rephrased_question_hybrid_dict, answer_key_name, cur_top_k_value, total_score, 'dont_know_partthing', analysis_process_key_name, score_key_name)
                            elif not args.only_eval_F1 and args.eval_partanswerable and score_key_name in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict[score_key_name] != None:
                                if score_key_name not in res_dicts:
                                    if total_score:
                                        res_dicts[score_key_name] = [rephrased_question_hybrid_dict[score_key_name] / total_score]
                                    else:
                                        pass
                                else:
                                    if total_score:
                                        res_dicts[score_key_name].append(rephrased_question_hybrid_dict[score_key_name] / total_score)
                                    else:
                                        pass
                        else:
                            # analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_anything-reason"
                            # score_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_anything-score"
                            analysis_process_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_anything2-reason"
                            score_key_name = f"rephrased_hybrid-{answer_key_name}-dont_know_anything2-score"
                            if not args.only_eval_F1 and args.eval_unanswerable and (score_key_name not in rephrased_question_hybrid_dict or rephrased_question_hybrid_dict[score_key_name] == None):
                                # future = executor.submit(judge_dont_know_anything_2, rephrased_question_hybrid_str, generated_answer)
                                future = executor.submit(judge_dont_know_anything_2, rephrased_question_hybrid_str, generated_answer)
                                futures_to_data[future] = (rephrased_question_hybrid_dict, answer_key_name, cur_top_k_value, 1, 'dont_know_anything', analysis_process_key_name, score_key_name)
                            elif not args.only_eval_F1 and args.eval_unanswerable and score_key_name in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict[score_key_name] != None:
                                if score_key_name not in res_dicts:
                                    res_dicts[score_key_name] = [rephrased_question_hybrid_dict[score_key_name]]
                                else:
                                    res_dicts[score_key_name].append(rephrased_question_hybrid_dict[score_key_name])
                        
        new_gen_num = 0
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
            proposed_question_dict, answer_key_name, cur_top_k_value, total_score, score_type, analysis_process_key_name, score_key_name = futures_to_data[future]
            try:
                score_response = future.result(timeout=5*60)
                if score_type in ['consistency_with_sens2']:
                    analysis_process, score = extract_consistency_with_sens_output(score_response)
                    proposed_question_dict[analysis_process_key_name] = analysis_process
                    proposed_question_dict[score_key_name] = score
                elif score_type in ['dont_know_partthing']:
                    analysis_process, score = extract_consistency_with_sens_output(score_response)
                    proposed_question_dict[analysis_process_key_name] = analysis_process
                    proposed_question_dict[score_key_name] = score
                elif score_type in ['dont_know_anything']:
                    analysis_process, score = extract_consistency_with_sens_output(score_response)
                    proposed_question_dict[analysis_process_key_name] = analysis_process
                    proposed_question_dict[score_key_name] = score

                if score_key_name not in res_dicts:
                    if score != None:
                        if total_score:
                            res_dicts[score_key_name] = [score / total_score]
                        else:
                            pass
                else:
                    if score != None:
                        if total_score:
                            res_dicts[score_key_name].append(score / total_score)
                        else:
                            pass

                new_gen_num += 1
                if (new_gen_num + 1) % args.save_interval == 0:
                    if output_path:
                        print(f"Saving results to {output_path}")
                        with open(output_path, 'w') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"Processed {new_gen_num} scoring tasks.")

            except Exception as e:
                print(f"Error processing {score_type} for answer_key_name {answer_key_name}: {e}")
                continue

    if output_path and (new_gen_num or not os.path.exists(output_path)):
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {new_gen_num} scoring tasks.")

    res_answer_key_names = list(set(res_answer_key_names))
    rephrased_types = ["original", "rephrased", "rephrased_part", "rephrased_hybrid"]
    num_part_and_any = {} #
    for rephrased_type in rephrased_types:
        for answer_key_name in res_answer_key_names:
            dont_key_names = [
                f"{rephrased_type}-{answer_key_name}-dont_know_anything-score",
                f"{rephrased_type}-{answer_key_name}-dont_know_partthing-score"
            ]
            dont_key_name = f"{rephrased_type}-{answer_key_name}-dont-score"
            for cur_dont_key_name in dont_key_names:
                if cur_dont_key_name in res_dicts:                    
                    if dont_key_name not in res_dicts:
                        res_dicts[dont_key_name] = copy.deepcopy(res_dicts[cur_dont_key_name])
                    else:
                        res_dicts[dont_key_name].extend(res_dicts[cur_dont_key_name])
                    num_part_and_any[cur_dont_key_name] = num_part_and_any.get(cur_dont_key_name, 0) + len(res_dicts[cur_dont_key_name])
    
    ratio_dicts = {}
    for rephrased_type in rephrased_types:
        for answer_key_name in res_answer_key_names:
            part_key_name = f"{rephrased_type}-{answer_key_name}-dont_know_partthing-score"
            any_key_name = f"{rephrased_type}-{answer_key_name}-dont_know_anything-score"
            dont_key_name = f"{rephrased_type}-{answer_key_name}-dont-score"
            
            if part_key_name in num_part_and_any and any_key_name in num_part_and_any:
                fenzi = num_part_and_any[part_key_name]
                fenmu = num_part_and_any[any_key_name] + num_part_and_any[part_key_name]
                ratio_dicts[dont_key_name] = f"{fenzi / fenmu * 100:.2f} = {fenzi} / {fenmu}"
                
    metric_dicts = {}
    for key, value in res_dicts.items():
        metric_dicts[key] = f"{sum(value) / len(value) * 100:.2f}"
        # print(f"{key}: {metric_dicts[key]}, sum: {sum(value)}, len: {len(value)}")
        # input("Press Enter to continue...")
        if key in ratio_dicts:
            metric_dicts[key] += f" ({ratio_dicts[key]})"
    
    metric_dicts = dict(sorted(metric_dicts.items(), key=lambda x: x[0]))

    return metric_dicts

def eval_retrieval_results_for_file_entity_graph(args, data, max_process_num=300, max_workers=8):

    res_dicts = {}

    if max_process_num == -1:
        max_process_num = len(data)
    else:
        max_process_num = min(max_process_num, len(data))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        for entity_dict in list(data.values())[:max_process_num]:
            if 'proposed-questions' not in entity_dict:
                continue
            
            proposed_questions = entity_dict['proposed-questions']
            objective_relationships = entity_dict['selected-relationships']['objective-relationships']
            objective_relationship_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}
            
            """calculate retrieval metrics"""
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                if 'top_k_documents' not in proposed_question_dict:
                    continue
                top_k_documents = proposed_question_dict['top_k_documents']

                # needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'])
                # needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                # needed_related_relationships = [objective_relationship_id_2_objective_relationship[int(clue_id)] for clue_id in needed_objective_relationship_ids if clue_id and int(clue_id) in objective_relationship_id_2_objective_relationship]
                # needed_corpusids = list(set([cur_related_relationship['id'] for cur_related_relationship in needed_related_relationships if 'id' in cur_related_relationship]))
                
                positive = proposed_question_dict["positive"]
                needed_corpusid2senids = extract_doc_to_sen(positive)
                needed_corpusids = list(needed_corpusid2senids.keys())

                for cur_top_k_value in args.top_k_values:
                    all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                    proposed_question_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])

                # precision
                for cur_top_k_value in args.top_k_values:
                    precision_key_name = f"Precision@{cur_top_k_value}; full; transformation@{0}"
                    percentage_of_elements_in_real_answer = cal_percentage_of_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                    
                    if precision_key_name not in res_dicts:
                        res_dicts[precision_key_name] = [percentage_of_elements_in_real_answer]
                    else:
                        res_dicts[precision_key_name].append(percentage_of_elements_in_real_answer)
                    
                    precision_key_name = f"Precision@{cur_top_k_value}; part; transformation@{0}"
                    if precision_key_name not in res_dicts:
                        res_dicts[precision_key_name] = [percentage_of_elements_in_real_answer]
                    else:
                        res_dicts[precision_key_name].append(percentage_of_elements_in_real_answer)

                    precision_key_name = f"Precision@{cur_top_k_value}; hybrid; transformation@{0}"
                    if precision_key_name not in res_dicts:
                        res_dicts[precision_key_name] = [percentage_of_elements_in_real_answer]
                    else:
                        res_dicts[precision_key_name].append(percentage_of_elements_in_real_answer)
                
                # precision for rephrased questions
                rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions, start=1):
                    if "top_k_documents" not in rephrased_question_dict:
                        continue
                    top_k_documents = rephrased_question_dict['top_k_documents']

                    for cur_top_k_value in args.top_k_values:
                        precision_key_name = f"Precision@{cur_top_k_value}; full; transformation@{rephrased_question_type}"
                        percentage_of_elements_in_real_answer = cal_percentage_of_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                        if precision_key_name not in res_dicts:
                            res_dicts[precision_key_name] = [percentage_of_elements_in_real_answer]
                        else:
                            res_dicts[precision_key_name].append(percentage_of_elements_in_real_answer)
                
                # precision for rephrased questions part
                rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                for rephrased_question_part_type, rephrased_question_part_dict in enumerate(rephrased_questions_part):
                    if "top_k_documents" not in rephrased_question_part_dict:
                        continue
                    top_k_documents = rephrased_question_part_dict['top_k_documents']

                    for cur_top_k_value in args.top_k_values:
                        precision_key_name = f"Precision@{cur_top_k_value}; part; transformation@{rephrased_question_part_type+1}"
                        percentage_of_elements_in_real_answer = cal_percentage_of_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                        if precision_key_name not in res_dicts:
                            res_dicts[precision_key_name] = [percentage_of_elements_in_real_answer]
                        else:
                            res_dicts[precision_key_name].append(percentage_of_elements_in_real_answer)
                
                # precision for rephrased questions hybrid
                rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid, start=1):
                    if "top_k_documents" not in rephrased_question_hybrid_dict:
                        continue
                    top_k_documents = rephrased_question_hybrid_dict['top_k_documents']

                    for cur_top_k_value in args.top_k_values:
                        precision_key_name = f"Precision@{cur_top_k_value}; hybrid; transformation@{rephrased_question_hybrid_type}"
                        percentage_of_elements_in_real_answer = cal_percentage_of_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                        if precision_key_name not in res_dicts:
                            res_dicts[precision_key_name] = [percentage_of_elements_in_real_answer]
                        else:
                            res_dicts[precision_key_name].append(percentage_of_elements_in_real_answer)
                
                # nDCG
                # grouped_sums = idcg_calculator_with_weight(args.difference_alpha, proposed_question_dict, needed_corpusids, top_k_documents)
                grouped_sums = idcg_calculator(needed_corpusids)
                
                relevances = [grouped_sums[cur_doc_id] if cur_doc_id in needed_corpusids else 0 for cur_doc_id in top_k_documents]
                idea_relevances = list(sorted(grouped_sums.values(), reverse=True)) + [0] * (len(top_k_documents) - len(grouped_sums))
                for cur_top_k_value in args.top_k_values:
                    tmp_dcg = dcg(relevances, cur_top_k_value)
                    tmp_idcg = idcg(idea_relevances, cur_top_k_value)
                    tmp_ndcg = tmp_dcg / tmp_idcg if tmp_idcg != 0 else 0.0
                    
                    nDCG_key_name = f"nDCG@{cur_top_k_value}; full; transformation@{0}"
                    if nDCG_key_name not in res_dicts:
                        res_dicts[nDCG_key_name] = [tmp_ndcg]
                    else:
                        res_dicts[nDCG_key_name].append(tmp_ndcg)
                    
                    nDCG_key_name = f"nDCG@{cur_top_k_value}; part; transformation@{0}"
                    if nDCG_key_name not in res_dicts:
                        res_dicts[nDCG_key_name] = [tmp_ndcg]
                    else:
                        res_dicts[nDCG_key_name].append(tmp_ndcg)

                    nDCG_key_name = f"nDCG@{cur_top_k_value}; hybrid; transformation@{0}"
                    if nDCG_key_name not in res_dicts:
                        res_dicts[nDCG_key_name] = [tmp_ndcg]
                    else:
                        res_dicts[nDCG_key_name].append(tmp_ndcg)
                
                rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                for rephrased_question_type, rephrased_question_dict in enumerate(rephrased_questions, start=1):
                    if "top_k_documents" not in rephrased_question_dict:
                        continue
                    top_k_documents = rephrased_question_dict['top_k_documents']

                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                    
                    relevances = [grouped_sums[cur_doc_id] if cur_doc_id in needed_corpusids else 0 for cur_doc_id in top_k_documents]
                    idea_relevances = list(sorted(grouped_sums.values(), reverse=True)) + [0] * (len(top_k_documents) - len(grouped_sums))
                    for cur_top_k_value in args.top_k_values:
                        tmp_dcg = dcg(relevances, cur_top_k_value)
                        tmp_idcg = idcg(idea_relevances, cur_top_k_value)
                        tmp_ndcg = tmp_dcg / tmp_idcg if tmp_idcg != 0 else 0.0
                        nDCG_key_name = f"nDCG@{cur_top_k_value}; full; transformation@{rephrased_question_type}"
                        if nDCG_key_name not in res_dicts:
                            res_dicts[nDCG_key_name] = [tmp_ndcg]
                        else:
                            res_dicts[nDCG_key_name].append(tmp_ndcg)
            
                rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                for rephrased_question_part_type, rephrased_question_part_dict in enumerate(rephrased_questions_part, start=1):
                    if "top_k_documents" not in rephrased_question_part_dict:
                        continue
                    top_k_documents = rephrased_question_part_dict['top_k_documents']

                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_part_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                    
                    relevances = [grouped_sums[cur_doc_id] if cur_doc_id in needed_corpusids else 0 for cur_doc_id in top_k_documents]
                    idea_relevances = list(sorted(grouped_sums.values(), reverse=True)) + [0] * (len(top_k_documents) - len(grouped_sums))
                    for cur_top_k_value in args.top_k_values:
                        tmp_dcg = dcg(relevances, cur_top_k_value)
                        tmp_idcg = idcg(idea_relevances, cur_top_k_value)
                        tmp_ndcg = tmp_dcg / tmp_idcg if tmp_idcg != 0 else 0.0
                        nDCG_key_name = f"nDCG@{cur_top_k_value}; part; transformation@{rephrased_question_part_type}"
                        if nDCG_key_name not in res_dicts:
                            res_dicts[nDCG_key_name] = [tmp_ndcg]
                        else:
                            res_dicts[nDCG_key_name].append(tmp_ndcg)
                
                rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid, start=1):
                    if "top_k_documents" not in rephrased_question_hybrid_dict:
                        continue
                    top_k_documents = rephrased_question_hybrid_dict['top_k_documents']

                    for cur_top_k_value in args.top_k_values:
                        all_in_top_k_key_name = f"all_in_top_{cur_top_k_value}"
                        rephrased_question_hybrid_dict[all_in_top_k_key_name] = are_all_elements_in_list(needed_corpusids, top_k_documents[:cur_top_k_value])
                    
                    relevances = [grouped_sums[cur_doc_id] if cur_doc_id in needed_corpusids else 0 for cur_doc_id in top_k_documents]
                    idea_relevances = list(sorted(grouped_sums.values(), reverse=True)) + [0] * (len(top_k_documents) - len(grouped_sums))
                    for cur_top_k_value in args.top_k_values:
                        tmp_dcg = dcg(relevances, cur_top_k_value)
                        tmp_idcg = idcg(idea_relevances, cur_top_k_value)
                        tmp_ndcg = tmp_dcg / tmp_idcg if tmp_idcg != 0 else 0.0
                        nDCG_key_name = f"nDCG@{cur_top_k_value}; hybrid; transformation@{rephrased_question_hybrid_type}"
                        if nDCG_key_name not in res_dicts:
                            res_dicts[nDCG_key_name] = [tmp_ndcg]
                        else:
                            res_dicts[nDCG_key_name].append(tmp_ndcg)

    metric_dicts = {}
    for key, value in res_dicts.items():
        metric_dicts[key] = f"{sum(value) / len(value) * 100:.2f}"

    metric_dicts = dict(sorted(metric_dicts.items(), key=lambda x: x[0]))

    return metric_dicts

def traverse_directory(args, cur_dir, file_path_2_result):

    for item in os.listdir(cur_dir):
        full_path = os.path.abspath(os.path.join(cur_dir, item))

        if os.path.isdir(full_path):
            traverse_directory(args, full_path, file_path_2_result)
        else:
            file_name = os.path.basename(full_path)
            relative_path = os.path.relpath(full_path, CUSTOM_CORPUS_HOME)

            if not ("content" in file_name and "contents" not in file_name or "entity_graph" in file_name):
                continue
            if not "rephrase_evaluator_content" in file_name:
                continue

            print(f"Processing file {relative_path}")

            rel_path = os.path.relpath(full_path, args.input_root_dir)
            if args.output_root_dir:
                output_path = os.path.join(args.output_root_dir, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            else:
                output_path = None
            if output_path and os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    data = json.load(f)
            else:
                with open(full_path, 'r') as f:
                    data = json.load(f)
            
            input_file_dir = os.path.dirname(full_path)
            input_file_name = os.path.basename(full_path)
            input_file_name_suffix = input_file_name.split('.')[-2]
            chunk_path = os.path.join(input_file_dir, input_file_name.replace(input_file_name_suffix, "chunk_contents"))

            if output_path:
                output_dir = os.path.dirname(output_path)
                chunk_output_path = os.path.join(output_dir, os.path.basename(chunk_path))
                if not os.path.exists(chunk_output_path):
                    os.system(f"cp {chunk_path} {chunk_output_path}")
            
            with open(chunk_path, 'r') as f:
                chunks_data = json.load(f)
            corpusid_2_context = {cur_dict['id']: cur_dict['context'] for cur_dict in chunks_data}
            
            if "content" in file_name and "contents" not in file_name: # content is a file type while contents is just the source chunk file
                if args.eval_retrieval:
                    retrieval_metric_dicts = eval_retrieval_results_for_file_content(args, data, max_process_num=args.max_process_num, max_workers=args.max_workers)
                
                if args.eval_generation:
                    generation_metric_dicts = eval_generation_results_for_file_content(args, data, corpusid_2_context, max_process_num=args.max_process_num, max_workers=args.max_workers, output_path=output_path)
                
                if args.eval_retrieval and args.eval_generation:
                    metric_dicts = {**retrieval_metric_dicts, **generation_metric_dicts}
                elif args.eval_retrieval:
                    metric_dicts = retrieval_metric_dicts
                elif args.eval_generation:
                    metric_dicts = generation_metric_dicts
                
                max_key_len = max([len(key) for key in metric_dicts.keys()])
                for key, value in metric_dicts.items():
                    print(f"{key.ljust(max_key_len)}: {value}")
                
                file_path_2_result[relative_path] = metric_dicts
                
                print("-" * 50)
            
            elif "entity_graph" in file_name:

                if args.eval_retrieval:
                    retrieval_metric_dicts = eval_retrieval_results_for_file_entity_graph(args, data, max_process_num=args.max_process_num, max_workers=args.max_workers)

                if args.eval_generation:
                    generation_metric_dicts = eval_generation_results_for_file_entity_graph(args, data, corpusid_2_context, max_process_num=args.max_process_num, max_workers=args.max_workers, output_path=output_path)
                
                if args.eval_retrieval and args.eval_generation:
                    metric_dicts = {**retrieval_metric_dicts, **generation_metric_dicts}
                elif args.eval_retrieval:
                    metric_dicts = retrieval_metric_dicts
                elif args.eval_generation:
                    metric_dicts = generation_metric_dicts
                
                max_key_len = max([len(key) for key in metric_dicts.keys()])
                for key, value in metric_dicts.items():
                    print(f"{key.ljust(max_key_len)}: {value}")
                
                file_path_2_result[relative_path] = metric_dicts
                
                print("-" * 50)

def sort_dict_recursively(d):
    """
    Recursively sorts all nested dictionaries by key.
    
    :param d: The dictionary to sort
    :return: The sorted dictionary
    """
    if not isinstance(d, dict):
        return d
    
    sorted_dict = {}
    for key in sorted(d):
        value = d[key]
        if isinstance(value, dict):
            sorted_dict[key] = sort_dict_recursively(value)
        else:
            sorted_dict[key] = value
    return sorted_dict

def main(args):

    file_path_2_result = {}
    traverse_directory(args, args.input_root_dir, file_path_2_result)

    file_path_2_result = sort_dict_recursively(file_path_2_result)

    result_dir = os.path.dirname(args.result_path)
    os.makedirs(result_dir, exist_ok=True)
    with open(args.result_path, 'w') as f:
        json.dump(file_path_2_result, f, indent=2, ensure_ascii=False)

    return file_path_2_result

if __name__ == '__main__':

    print("-" * 50)
    parser = argparse.ArgumentParser(description="Calculate metrics for query results.")
    
    # for both precision, nDCG and generation metrics
    parser.add_argument('--input_root_dir', type=str, help="Root folder containing query results.")
    parser.add_argument('--result_path', type=str, help="Path to save the result.")
    parser.add_argument('--eval_retrieval', action='store_true', help="Whether to evaluate retrieval metrics.")
    
    parser.add_argument('--eval_generation', action='store_true', help="Whether to evaluate generation metrics.")
    parser.add_argument('--not_gen_for_original', action='store_true', help="Generate for original questions.")
    parser.add_argument('--eval_answerable', action='store_true', help="Whether to evaluate answerable part.") # eval answerable part
    parser.add_argument('--only_eval_F1', action='store_true', help="Whether to only eval F1.")
    parser.add_argument('--eval_partanswerable', action='store_true', help="Whether to evaluate unanswerable part.")
    parser.add_argument('--eval_unanswerable', action='store_true', help="Whether to evaluate unanswerable part.")
    parser.add_argument('--only_eval_at_rephrased_poses', type=int, nargs='+', default=[3], help="List of rephrased positions to evaluate.")
    parser.add_argument('--only_eval_at_rephrased_poses_part', type=int, nargs='+', default=[2], help="List of rephrased positions part to evaluate.")
    parser.add_argument('--only_eval_at_rephrased_poses_hybrid', type=int, nargs='+', default=[6], help="List of rephrased positions hybrid to evaluate.")
    
    # for precision and nDCG only
    parser.add_argument('--top_k_values', type=int, nargs='+', default=[3, 5], help="List of top_k_value values for which precision will be calculated.")
    # parser.add_argument('--difference_alpha', type=float, default=1.0, help="The difference_alpha parameter for the softmax function in nDCG calculation.")

    # for generation metrics only
    parser.add_argument('--output_root_dir', type=str, default=None, help="Root folder to save the results.")
    parser.add_argument('--save_interval', type=int, default=100, help="The interval at which to save the results.")
    parser.add_argument('--max_workers', type=int, default=8, help="Maximum number of concurrent requests.")
    parser.add_argument('--max_process_num', type=int, default=-1, help="Maximum number of process data.")
    args = parser.parse_args()

    args.input_root_dir = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.input_root_dir))
    if args.output_root_dir:
        args.output_root_dir = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.output_root_dir))
    args.result_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.result_path))

    # args.difference_alpha = float(args.difference_alpha)

    main(args)