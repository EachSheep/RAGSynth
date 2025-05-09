import os
import json
import math
import re
import numpy as np
import argparse
from tqdm import tqdm
from openai import OpenAI
from scipy.stats import entropy
from collections import defaultdict
from scipy.special import rel_entr
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import re

from rag.utils.request_openai_utils import OpenAIModel

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
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

def scoring_by_points_once_give_one(client, transformed_question, rephrased_type: Optional[str] = ['rephrased', 'rephrased_part', 'rephrased_hybrid', 'rephrased_hybrid_part']):
    
    if rephrased_type == 'rephrased':
        cur_prompt = """I will provide you with a question, referred to as the "original question", an answer to the original question, and a rephrased "new question" derived from the original question. Your task is to help me generate detailed scoring criteria for this "new question". 
------
Examples:

Original Question: How much has Apple's revenue increased its revenue compared to last year?
Answer: Apple's revenue increased by 6% year-over-year, reaching $94.9 billion.
Rephrased Question:
    <transformed-action>Near-synonym Replacement</transformed-action>
    <transformed-explanation>Replace "Apple" with "the company led by Tim Cook"</transformed-explanation> 
    <transformed-question>How much has the company led by Tim Cook grown its revenue compared to last year?</transformed-question>
    <transformed-answer>The company led by Tim Cook experienced a 6% growth in revenue year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].</transformed-answer>
Scoring Criteria: If the response shows that the respondent understands "the company led by Tim Cook" refers to Apple and correctly associates it with revenue growth, award 1 point.

Original Question: How much has the company led by Tim Cook grown its revenue compared to last year?
Answer: The company led by Tim Cook experienced a 6% growth in revenue year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].
Rephrased Question:
    <transformed-action>Semantic Ambiguity</transformed-action>
    <transformed-explanation>Introduce ambiguity by asking about revenue change without specifying the direction explicitly</transformed-explanation> 
    <transformed-question>What is the change in revenue for the company led by Tim Cook compared to the previous year?</transformed-question>
    <transformed-answer>The company led by Tim Cook saw its revenue rise by 6% year-over-year, totaling $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].</transformed-answer>
Scoring Criteria: If the response demonstrates that the respondent knows the correct 6% year-over-year revenue increase and the total revenue of $94.9 billion, despite any ambiguity, award 1 point.

Original Question: What is the change in revenue for the company led by Tim Cook compared to the previous year?
Answer: The company led by Tim Cook saw its revenue rise by 6% year-over-year, totaling $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].
Rephrased Question: 
    <transformed-action>Perspective Shift</transformed-action>
    <transformed-explanation>Pose the question from the perspective of a financial analyst</transformed-explanation> 
    <transformed-question>As a financial analyst, do you know the revenue change of the company led by Tim Cook compared to the previous fiscal year?</transformed-question>
    <transformed-answer>From a financial analyst's perspective, the company led by Tim Cook demonstrated a 6% increase in revenue year-over-year, amounting to $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].</transformed-answer>
Scoring Criteria: If the response shows that the respondent recognizes "as a financial analyst" as irrelevant information and provides the correct revenue change, award 1 point.

Original Question: As a financial analyst, do you know the revenue change of the company led by Tim Cook compared to the previous fiscal year?
Answer: From a financial analyst's perspective, the company led by Tim Cook demonstrated a 6% increase in revenue year-over-year, amounting to $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].
Rephrased Question:
    <transformed-action>Conditional Addition</transformed-action>
    <transformed-explanation>Add a condition related to stable market conditions while maintaining the financial analyst perspective</transformed-explanation> 
    <transformed-question>As a financial analyst, do you know the revenue change of the company led by Tim Cook compared to the previous fiscal year, assuming stable market conditions?</transformed-question>
    <transformed-answer>Assuming stable market conditions, from a financial analyst's perspective, the company led by Tim Cook achieved a 6% revenue growth year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].</transformed-answer>
Scoring Criteria: If the response shows that the respondent recognizes "the condition of assuming stable market conditions" as irrelevant information and provides the correct revenue change, award 1 point.

------
Please note:  
(1) The scoring criteria should be formulated by comparing the original question with the transformed question, considering the answer, and reflecting on the transformation that led to the "new question".
(2) The scoring criteria should be detailed and comprehensive, meaning they should be clear enough to allow scoring based solely on the criteria, question, and answer, without needing additional knowledge.
(3) The scoring criteria should begin with "If the response shows that the respondent recognizes/knows/understands ..." to indicate the expected behavior for awarding points.

Your output should strictly follow the format below.
Scoring Criteria: (Here goes the scoring criterion)
------
[[TRANSFORMED QUESTION]]

Begin!
------
"""

    elif rephrased_type == 'rephrased_part':
        cur_prompt = """I will provide you with a question, referred to as the "original question", an answer to the original question, and a rephrased "new question" derived from the original question. Your task is to help me generate detailed scoring criteria for this "new question".
------
Examples:

Original Question: How much has Apple's revenue increased its revenue compared to last year?
Answer: Apple's revenue increased by 6% year-over-year, reaching $94.9 billion.
Rephrased Question:
    <transformed-action>Temporal Expansion</transformed-action>
    <transformed-explanation>Extend the time frame to inquire about revenue growth over multiple years.</transformed-explanation>
    <transformed-question>What has been the revenue growth of Apple over the past three years compared to each preceding year?</transformed-question>
    <transformed-answer>Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. I didn't get any information about the revenue growth in the previous two years, so a complete analysis over the past three years cannot be conducted based on the given clues.</transformed-answer>
Scoring Criteria: If the response demonstrates the respondent's awareness of missing information for the two preceding years, award 1 point.

Original Question: What has been the revenue growth of Apple over the past three years compared to each preceding year?
Answer: Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. I didn't get any information about the revenue growth in the previous two years, so a complete analysis over the past three years cannot be conducted based on the given clues.
Rephrased Question:
    <transformed-action>Comparison Addition</transformed-action>
    <transformed-explanation>Introduce a comparison with the revenue growth of key competitors during the same period.</transformed-explanation>
    <transformed-question>What has been the revenue growth of Apple over the past three years compared to each preceding year, and how does this growth compare to that of its main competitors?</transformed-question>
    <transformed-answer>Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. However, I didn't get any information about the revenue in the previous two years or the growth of Apple's main competitors, so a comparison cannot be made based on the available clues.</transformed-answer>
Scoring Criteria: If the response indicates the respondent's awareness that there is no available information regarding the growth of Apple's main competitors, award 1 point.

Original Question: What has been the revenue growth of Apple over the past three years compared to each preceding year, and how does this growth compare to that of its main competitors?
Answer: Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. However, I didn't get any information about the revenue in the previous two years or the growth of Apple's main competitors, so a comparison cannot be made based on the available clues.
Rephrased Question:
    <transformed-action>Metric Segmentation</transformed-action>
    <transformed-explanation>Break down the revenue growth by specific product categories, such as hardware and services.</transformed-explanation>
    <transformed-question>How does the revenue growth of Apple's overall revenue, along with hardware and services divisions over the past three years compare to each preceding year, and how does this growth compare to that of its main competitors?</transformed-question>
    <transformed-answer>Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. However, I don't have specific data on the revenue growth of Apple's hardware and services divisions over the past three years, and I don't know anything about its competitors, so a detailed breakdown cannot be provided based on the given clues.</transformed-answer>
Scoring Criteria: If the response demonstrates the respondent's understanding of the inability to provide detailed information about hardware and services divisions due to missing information, award 1 point.
------
Please note:  
(1) The scoring criteria should be formulated by comparing the original question with the transformed question, considering the answer, and reflecting on the transformation that led to the "new question".
(2) Your scoring criteria should be detailed and comprehensive, meaning they should be clear enough to allow scoring based solely on the criteria, question, and answer, without needing additional knowledge.
(3) The scoring criteria should begin with "If the response shows that the respondent recognizes/knows/understands ..." to indicate the expected behavior for awarding points.
(4) The scoring criteria should focus on the respondent's understanding that the new sub-question about the change cannot be answered.

Your output should strictly follow the format below.
Scoring Criteria: (Here goes the scoring criterion)
------
[[TRANSFORMED QUESTION]]

Begin! Please note that the maximum score for your evaluation criteria is only 1 point. It is crucial that you strictly generate the scoring criteria based on the differences between the transformed question and the original question, indicating that the generated answer should reflect its awareness of this "difference" and thus cannot be answered.
------
"""
    elif rephrased_type == 'rephrased_hybrid':
        cur_prompt = """I will provide you with a question, referred to as the "original question", an answer to the original question, and a rephrased "new question" derived from the original question. Your task is to help me generate detailed scoring criteria for this "new question".
------
Example:

Original Question: How much has Apple's revenue increased compared to last year?
Answer: Apple's revenue increased by 6% year-over-year, reaching $94.9 billion.
Rephrased Question:
    <transformed-action>Equivalent Transformation: Near-synonym Replacement</transformed-action>
    <transformed-explanation>Replace "Apple's" with "the company led by Tim Cook"</transformed-explanation> 
    <transformed-question>How much has the company led by Tim Cook increased its revenue compared to last year?</transformed-question>
    <transformed-answer>The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].</transformed-answer>
Scoring Criteria: If the response shows that the respondent recognizes "the company led by Tim Cook" as referring to Apple and correctly relates it to the revenue increase, award 1 point.

Original Question: How much has the company led by Tim Cook increased its revenue compared to last year?
Answer: The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].
Rephrased Question:
    <transformed-action>Partial Transformation: Temporal Expansion</transformed-action>
    <transformed-explanation>Extend the time frame to inquire about revenue growth over multiple years.</transformed-explanation>
    <transformed-question>What has been the revenue growth of the company led by Tim Cook over the past three years compared to each preceding year?</transformed-question>
    <transformed-answer>The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. I don't have information about the revenue growth in the previous two years, so a complete analysis over the past three years cannot be conducted based on the given clues.</transformed-answer>
Scoring Criteria: If the response demonstrates that the respondent knows Apple's revenue growth over the previous year beyond this year cannot be answered, award 1 point.

Original Question: What has been the revenue growth of the company led by Tim Cook over the past three years compared to each preceding year?
Answer: The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. I don't have information about the revenue growth in the previous two years, so a complete analysis over the past three years cannot be conducted based on the given clues.
Rephrased Question:
    <transformed-action>Equivalent Transformation: Semantic Ambiguity</transformed-action>
    <transformed-explanation>Introduce ambiguity by generalizing the time periods mentioned.</transformed-explanation>
    <transformed-question>How does the revenue growth of the company led by Tim Cook over recent years compared to preceding periods?</transformed-question>
    <transformed-answer>The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. There is insufficient information to compare revenue growth across multiple years based on the given clues.</transformed-answer>
Scoring Criteria: If the response shows that the respondent understands "recent years" and "past three years" express similar meanings, and successfully connects these time periods with this year's revenue increase, award 1 point.

Original Question: How does the revenue growth of the company led by Tim Cook over recent years compared to preceding periods?
Answer: The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. There is insufficient information to compare revenue growth across multiple years based on the given clues.
Rephrased Question:
    <transformed-action>Partial Transformation: Comparison Addition</transformed-action>
    <transformed-explanation>Introduce a comparison with the revenue growth of key competitors during the same period, while retaining the focus on multiple years.</transformed-explanation>
    <transformed-question>How does the revenue growth of the company led by Tim Cook over recent years compared to preceding periods, and how does this growth compare to that of its main competitors?</transformed-question>
    <transformed-answer>The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. However, there is no information about the revenue growth about other years and any information about Apple's main competitors, so a comparison cannot be made based on the available clues.</transformed-answer>
Scoring Criteria: If the response demonstrates that the respondent knows they cannot answer the revenue increase compared to that of its main competitors, award 1 point.

Original Question: How does the revenue growth of the company led by Tim Cook over recent years compared to preceding periods, and how does this growth compare to that of its main competitors?
Answer: The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. However, there is no information about the revenue growth about other years and any information about Apple's main competitors, so a comparison cannot be made based on the available clues.
Rephrased Question:
    <transformed-action>Equivalent Transformation: Perspective Shift</transformed-action>
    <transformed-explanation>Pose the question from an investor's perspective, considering both Apple’s and its competitors' performance.</transformed-explanation>
    <transformed-question>As an investor, how would you assess the revenue growth of the company led by Tim Cook over recent years compared to preceding periods and in relation to its main competitors?</transformed-question>
    <transformed-answer>As an investor, the company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. There is no information about the revenue growth about other years of Apple's main competitors, so a comprehensive assessment cannot be made based on the available clues.</transformed-answer>
Scoring Criteria: If the response demonstrates that the respondent understands "As an investor" is irrelevant information, either by directly ignoring it when answering the question or by explicitly stating that they know it is irrelevant, award 1 point.

Original Question: As an investor, how would you assess the revenue growth of the company led by Tim Cook over recent years compared to preceding periods and in relation to its main competitors?
Answer: As an investor, the company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. There is no information about the revenue growth about other years of Apple's main competitors, so a comprehensive assessment cannot be made based on the available clues.
Rephrased Question:
    <transformed-action>Partial Transformation: Metric Segmentation</transformed-action>
    <transformed-explanation>Break down the revenue growth by specific product categories while retaining the multi-year and comparison aspects.</transformed-explanation>
    <transformed-question>As an investor, how does the revenue growth of the company led by Tim Cook's hardware and services divisions over recent years compare to each preceding year and to its main competitors?</transformed-question>
    <transformed-answer>As an investor, the company led by Tim Cook's overall revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. Specific data on the other years of revenue growth, and revenue growth of the hardware and services divisions, as well as comparisons to main competitors, are not available based on the given clues.</transformed-answer>
Scoring Criteria: If the response demonstrates that the respondent knows there is missing information regarding the revenue increase of hardware and services divisions compared to each preceding year, award 1 point.
------
Please note:
(1) The scoring criteria should be formulated by comparing the original question with the transformed question, considering the answer, and reflecting on the transformation that led to the "new question".
(2) Your scoring criteria should be detailed and comprehensive, meaning they should be clear enough to allow scoring based solely on the criteria, question, and answer, without needing additional knowledge.
(3) The scoring criteria should begin with "If the response shows that the respondent recognizes/knows/understands ..." to indicate the expected behavior for awarding points.
(4) If the transformed type is Equivalent Transformation, the scoring criteria should focus on the respondent's understanding that the question before and after transformation expresses the same meaning. If the transformed type is Partial Transformation, the scoring criteria should focus on the respondent's understanding that the new sub-question about the change cannot be answered.

Your output should strictly follow the format below.
Scoring Criteria: (Here goes the scoring criterion)
------

[[TRANSFORMED QUESTION]]

Begin! Please note that the maximum score for your evaluation criteria is only 1 point. It is crucial that you strictly generate the scoring criteria based on the differences between the transformed question and the original question, indicating that the generated answer should reflect its awareness of this "difference" and thus cannot be answered.
------
"""
    elif rephrased_type == 'rephrased_hybrid_part':
        cur_prompt = """I will provide you with a question, referred to as the "original question", an answer to the original question, and a rephrased "new question" derived from the original question. Your task is to help me generate detailed scoring criteria for this "new question".
------
Example:

Original Question: How much has the company led by Tim Cook increased its revenue compared to last year?
Answer: The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5].
Rephrased Question:
    <transformed-action>Partial Transformation: Temporal Expansion</transformed-action>
    <transformed-explanation>Extend the time frame to inquire about revenue growth over multiple years.</transformed-explanation>
    <transformed-question>What has been the revenue growth of the company led by Tim Cook over the past three years compared to each preceding year?</transformed-question>
    <transformed-answer>The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. I don't have information about the revenue growth in the previous two years, so a complete analysis over the past three years cannot be conducted based on the given clues.</transformed-answer>
Scoring Criteria: If the response demonstrates that the respondent knows Apple's revenue growth over the previous year beyond this year cannot be answered, award 1 point.

Original Question: How does the revenue growth of the company led by Tim Cook over recent years compared to preceding periods?
Answer: The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. There is insufficient information to compare revenue growth across multiple years based on the given clues.
Rephrased Question:
    <transformed-action>Partial Transformation: Comparison Addition</transformed-action>
    <transformed-explanation>Introduce a comparison with the revenue growth of key competitors during the same period, while retaining the focus on multiple years.</transformed-explanation>
    <transformed-question>How does the revenue growth of the company led by Tim Cook over recent years compared to preceding periods, and how does this growth compare to that of its main competitors?</transformed-question>
    <transformed-answer>The company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. However, there is no information about the revenue growth about other years and any information about Apple's main competitors, so a comparison cannot be made based on the available clues.</transformed-answer>
Scoring Criteria: If the response demonstrates that the respondent knows they cannot answer the revenue increase compared to that of its main competitors, award 1 point.

Original Question: As an investor, how would you assess the revenue growth of the company led by Tim Cook over recent years compared to preceding periods and in relation to its main competitors?
Answer: As an investor, the company led by Tim Cook is Apple, Apple's revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. There is no information about the revenue growth about other years of Apple's main competitors, so a comprehensive assessment cannot be made based on the available clues.
Rephrased Question:
    <transformed-action>Partial Transformation: Metric Segmentation</transformed-action>
    <transformed-explanation>Break down the revenue growth by specific product categories while retaining the multi-year and comparison aspects.</transformed-explanation>
    <transformed-question>As an investor, how does the revenue growth of the company led by Tim Cook's hardware and services divisions over recent years compare to each preceding year and to its main competitors?</transformed-question>
    <transformed-answer>As an investor, the company led by Tim Cook's overall revenue increased by 6% year-over-year, reaching $94.9 billion [Doc https_www.trendforce.com_news_2024_11_01_news-apple-reports-6, Sen 2, 5]. Specific data on the other years of revenue growth, and revenue growth of the hardware and services divisions, as well as comparisons to main competitors, are not available based on the given clues.</transformed-answer>
Scoring Criteria: If the response demonstrates that the respondent knows there is missing information regarding the revenue increase of hardware and services divisions compared to each preceding year, award 1 point.
------
Please note:
(1) The scoring criteria should be formulated by comparing the original question with the transformed question, considering the answer, and reflecting on the transformation that led to the "new question".
(2) Your scoring criteria should be detailed and comprehensive, meaning they should be clear enough to allow scoring based solely on the criteria, question, and answer, without needing additional knowledge.
(3) The scoring criteria should begin with "If the response shows that the respondent recognizes/knows/understands ..." to indicate the expected behavior for awarding points.
(4) The transformed type is Partial Transformation. The scoring criteria should focus on the respondent's understanding that the new sub-question about the change cannot be answered.

Your output should strictly follow the format below.
Scoring Criteria: (Here goes the scoring criterion)
------
[[TRANSFORMED QUESTION]]

Begin! Please note that the maximum score for your evaluation criteria is only 1 point. It is crucial that you strictly generate the scoring criteria based on the differences between the transformed question and the original question, indicating that the generated answer should reflect its awareness of this "difference" and thus cannot be answered.
------
"""
    else:
        raise ValueError("Invalid rephrased_type")

    cur_prompt = cur_prompt.replace("[[TRANSFORMED QUESTION]]", transformed_question)
    
    response, _ = openai_model.generate(client, cur_prompt, TEMPERATURE)
    return response

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

def get_transformed_question(last_question, last_answer, cur_rephrased_question):
    ret_str = ""
    ret_str += f"Original Question: {last_question}\n"
    ret_str += f"Answer: {last_answer}\n\n"
    ret_str += "Rephrased Question:\n"
    ret_str += f"    <transformed-action>{cur_rephrased_question['transformation']}</transformed-action>\n"
    ret_str += f"    <transformed-explanation>{cur_rephrased_question['explanation']}</transformed-explanation>\n"
    ret_str += f"    <transformed-question>{cur_rephrased_question['result']}</transformed-question>\n"
    ret_str += f"    <transformed-answer>{cur_rephrased_question['answer']}</transformed-answer>\n"
    ret_str += "Scoring Criteria:\n"
    return ret_str

def process_file_content(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        with open(input_path, 'r') as f:
            data = json.load(f)

    if rephrase_evaluator_max_gen_times == -1:
        rephrase_evaluator_max_gen_times = len(data)

    all_num, new_gen_num = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_data = {}
        for cur_dict in data[:rephrase_evaluator_max_gen_times]:
            
            if 'proposed-questions' not in cur_dict:
                continue
            proposed_questions = cur_dict['proposed-questions']

            if_already_generated = False
            for proposed_question_type, proposed_question_dict in proposed_questions.items():

                question = proposed_question_dict['question']
                answer = proposed_question_dict['answer']
                
                # cur_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                # last_question = question
                # last_answer = answer
                # for cur_rephrased_question in cur_rephrased_questions:
                #     if 'scoring-criteria' in cur_rephrased_question:
                #         continue
                #     transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question)
                #     future = executor.submit(scoring_by_points_once_give_one, CLIENT, transformed_question, rephrased_type='rephrased')
                #     futures_to_data[future] = (cur_rephrased_question, 'criteria')
                #     last_question = cur_rephrased_question['result']
                #     last_answer = cur_rephrased_question['answer']
                
                cur_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                last_question = question
                last_answer = answer
                for cur_rephrased_question_part in cur_rephrased_questions_part:
                    if 'scoring-criteria' in cur_rephrased_question_part:
                        continue
                    transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question_part)
                    # print("transformed_question:", transformed_question)
                    # input("Press Enter to continue...")
                    future = executor.submit(scoring_by_points_once_give_one, CLIENT, transformed_question, rephrased_type='rephrased_part')
                    futures_to_data[future] = (cur_rephrased_question_part, 'criteria')
                    last_question = cur_rephrased_question_part['result']
                    last_answer = cur_rephrased_question_part['answer']

                cur_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                last_question = question
                last_answer = answer
                for cur_rephrased_question_hybrid in cur_rephrased_questions_hybrid:
                    if 'scoring-criteria' in cur_rephrased_question_hybrid:
                        continue
                    transformation_type = cur_rephrased_question_hybrid['transformation']
                    if "Partial Transformation" in transformation_type:
                        transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question_hybrid)
                        future = executor.submit(scoring_by_points_once_give_one, CLIENT, transformed_question, rephrased_type='rephrased_hybrid_part')
                        futures_to_data[future] = (cur_rephrased_question_hybrid, 'criteria')
                    last_question = cur_rephrased_question_hybrid['result']
                    last_answer = cur_rephrased_question_hybrid['answer']

            if if_already_generated:
                continue

        all_num = len(futures_to_data)
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
            cur_rephrased_question, score_type = futures_to_data[future]
            score_response = future.result(timeout=5*60)
            if score_type == 'criteria':
                score_response = score_response.replace("Scoring Criteria:", "")
                cur_rephrased_question['scoring-criteria'] = score_response
            else:
                raise ValueError("Invalid score_type")
            
            new_gen_num += 1
            if (new_gen_num + 1) % save_interval == 0:
                print(f"Saving results to {output_path}")
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

    if new_gen_num or not os.path.exists(output_path):
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

    return new_gen_num, all_num

def process_file_entity_graph(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times):
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        with open(input_path, 'r') as f:
            data = json.load(f)

    if rephrase_evaluator_max_gen_times == -1:
        rephrase_evaluator_max_gen_times = len(data)

    all_num, new_gen_num = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures_to_data = {}

        for entity_id, entity_dict in list(data.items())[:rephrase_evaluator_max_gen_times]:
            
            proposed_questions = entity_dict['proposed-questions']
            
            if_already_generated = False
            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                question = proposed_question_dict['question']
                answer = proposed_question_dict['positive']

                # cur_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
                # last_question = question
                # last_answer = answer
                # for cur_rephrased_question in cur_rephrased_questions:
                #     if 'scoring-criteria' in cur_rephrased_question:
                #         continue
                #     transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question)
                #     future = executor.submit(scoring_by_points_once_give_one, CLIENT, transformed_question, rephrased_type='rephrased')
                #     futures_to_data[future] = (cur_rephrased_question, 'criteria')
                #     last_question = cur_rephrased_question['result']
                #     last_answer = cur_rephrased_question['answer']
                
                cur_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
                last_question = question
                last_answer = answer
                for cur_rephrased_question_part in cur_rephrased_questions_part:
                    if 'scoring-criteria' in cur_rephrased_question_part:
                        continue
                    transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question_part)
                    future = executor.submit(scoring_by_points_once_give_one, CLIENT, transformed_question, rephrased_type='rephrased_part')
                    futures_to_data[future] = (cur_rephrased_question_part, 'criteria')
                    last_question = cur_rephrased_question_part['result']
                    last_answer = cur_rephrased_question_part['answer']
                
                cur_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
                last_question = question
                last_answer = answer
                for cur_rephrased_question_hybrid in cur_rephrased_questions_hybrid:
                    if 'scoring-criteria' in cur_rephrased_question_hybrid:
                        continue
                    transformed_question = get_transformed_question(last_question, last_answer, cur_rephrased_question_hybrid)
                    future = executor.submit(scoring_by_points_once_give_one, CLIENT, transformed_question, rephrased_type='rephrased_hybrid_part')
                    futures_to_data[future] = (cur_rephrased_question_hybrid, 'criteria')
                    last_question = cur_rephrased_question_hybrid['result']
                    last_answer = cur_rephrased_question_hybrid['answer']

            if if_already_generated:
                continue

        all_num = len(futures_to_data)
        for future in tqdm(as_completed(futures_to_data), total=len(futures_to_data), desc="Processing Futures", dynamic_ncols=True):
            proposed_question_dict, score_type = futures_to_data[future]
            cur_rephrased_question = future.result(timeout=10*60)
            if score_type == 'criteria':
                score_response = cur_rephrased_question.replace("Scoring Criteria:", "")
                proposed_question_dict['scoring-criteria'] = score_response
            else:
                raise ValueError("Invalid score_type")

            new_gen_num += 1
            if (new_gen_num + 1) % save_interval == 0:
                print(f"Saving results to {output_path}")
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"Processed {new_gen_num}/{all_num} scoring tasks.")

    if new_gen_num or not os.path.exists(output_path):
        print(f"Saving results to {output_path}")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Processed {new_gen_num}/{all_num} scoring tasks.")
    
    return new_gen_num, all_num
    
def rephrase_evaluator(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times):
    file_name = os.path.basename(input_path)
    relative_path = os.path.relpath(input_path, CUSTOM_CORPUS_HOME)
    print(f"Processing file {relative_path}")

    if "content" in file_name:
        return process_file_content(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times)
    elif "entity_graph" in file_name:
        return process_file_entity_graph(input_path, output_path, save_interval, max_workers, rephrase_evaluator_max_gen_times)
    else:
        raise ValueError(f"Unknown file type: {file_name}")
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Calculate metrics for answer.")
    parser.add_argument('--input_path', type=str, help="Input file containing query results.")
    parser.add_argument('--output_path', type=str, help="Output file to save the results.")
    parser.add_argument('--save_interval', type=int, help="The interval at which to save the results.")
    parser.add_argument('--max_workers', type=int, default=8, help="Maximum number of concurrent requests.")
    parser.add_argument('--rephrase_evaluator_max_gen_times', type=int, default=-1, help="Maximum number of generations to process.")
    args = parser.parse_args()

    args.input_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.input_path))
    args.output_path = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.output_path))

    rephrase_evaluator(args.input_path, args.output_path, args.save_interval, args.max_workers, args.rephrase_evaluator_max_gen_times)