import os
import json
import random
import argparse

from datasets import Dataset
from tqdm import tqdm
from transformers import set_seed
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag.utils import (
    list_to_docided_string,
)

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for the DPO training script."
    )

    # Optional integer with default
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed that will be set at the beginning of training."
    )

    # Required string arguments (default=None implies they are optional)
    parser.add_argument(
        '--dir_model_name',
        type=str,
        default=None,
        help="use which model to generate the negative answer."
    )
    parser.add_argument(
        '--corpuspath_2_inputpaths',
        type=str,
        default=None,
        help="Path to the corpus 2 input map JSON file."
    )
    parser.add_argument(
        '--prompt_path',
        type=str,
        default=None,
        help="The path to the prompt_template file."
    )

    # List of integers with default values
    parser.add_argument(
        '--only_gen_at_rephrased_poses',
        type=int,
        nargs='+',
        default=[3],
        help="List of rephrased positions to generate."
    )
    parser.add_argument(
        '--only_gen_at_rephrased_poses_part',
        type=int,
        nargs='+',
        default=[2],
        help="List of rephrased positions part to generate."
    )
    parser.add_argument(
        '--only_gen_at_rephrased_poses_hybrid',
        type=int,
        nargs='+',
        default=[6],
        help="List of rephrased positions hybrid to generate."
    )
    parser.add_argument(
        '--top_k_values',
        type=int,
        nargs='+',
        default=[3, 5],
        help="The top k values to consider."
    )

    # Integer arguments with default values
    parser.add_argument(
        '--save_interval',
        type=int,
        default=500,
        help="The interval to save the dataset."
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=8,
        help="The number of workers to use."
    )

    # Optional string for output directory
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="The output directory."
    )

    args = parser.parse_args()
    return args

rejection_statements_to_choose_from = [
    "I'm sorry, but I can't find any information on this issue at the moment.",
    "I apologize, but I currently have no leads on this matter.",
    "Unfortunately, I have no insights on this topic right now.",
    "Regrettably, I am unable to uncover any details related to this problem.",
    "I’m sorry, but I’m unable to locate any relevant information at present.",
    "I’m afraid I don’t have any clues about this issue for now.",
    "I apologize, but I can't find any data on this problem at the moment.",
    "Unfortunately, I have no information available regarding this issue.",
    "I regret to inform you that I have no findings on this matter currently.",
    "I’m sorry, but no clues are available to me regarding this issue at this time.",
    "I apologize, but I can't seem to find any evidence concerning this problem.",
    "Sadly, I have no insights to offer on this topic at this moment.",
    "I’m sorry, but nothing relevant has come up regarding this issue so far.",
    "Unfortunately, there's no trace of any information on this issue right now.",
    "I regret that I have no details to provide on this subject at the moment.",
    "I'm unable to locate any pertinent information on this issue right now.",
    "I apologize, but there are no indications of any clues on this matter currently.",
    "I’m sorry, but I have yet to find any relevant leads on this issue.",
    "I regret to say that there's no information I can access about this problem now.",
    "Unfortunately, I'm without any clues on this topic at this time.",
    "I'm sorry, but I haven't found any useful information about this issue yet.",
    "Regrettably, I have no details to share on this matter at present.",
    "I apologize, but there are no signs of information on this issue right now.",
    "I'm afraid I'm at a loss for any clues on this topic currently.",
    "Unfortunately, no relevant information is available to me at the moment.",
    "I’m sorry, but there’s nothing I can find on this issue at present.",
    "Regrettably, I can't uncover any information about this matter right now.",
    "I apologize, but there's no data available to me on this topic currently.",
    "I’m sorry, but I haven’t been able to gather any clues regarding this issue.",
    "Unfortunately, I have no information to provide on this subject now.",
    "I regret that I can't find any relevant details about this problem at the moment.",
    "I apologize, but I'm unable to access any clues on this matter currently.",
    "I'm sorry, but I haven't located any information regarding this issue yet.",
    "Regrettably, I am without any leads on this topic at this time.",
    "I apologize, but there are no available insights on this issue right now.",
    "I’m afraid I’ve found no information on this problem so far.",
    "Unfortunately, I’m unable to trace any details about this matter at this moment.",
    "I’m sorry, but there’s no relevant information I can find on this issue right now.",
    "Regrettably, I have no clues to offer about this topic at the moment.",
    "I apologize, but there’s no data indicating any information on this issue currently.",
    "I'm sorry, but I have yet to come across any insights on this problem.",
    "Unfortunately, no useful information is accessible to me on this topic at present.",
    "I regret that I can't provide any leads on this matter right now.",
    "I apologize, but I have not discovered any pertinent information about this issue.",
    "I’m sorry, but I have been unable to locate any details on this topic at this time.",
    "Regrettably, no information is available to me regarding this problem at the moment.",
    "I apologize, but I haven’t found any clues on this issue so far.",
    "I'm sorry, but I have no evidence to present on this matter right now.",
    "Unfortunately, I am unable to find any relevant insights about this issue currently.",
    "I regret that I have no information to share on this topic at the moment."
]

def retreival_true_positive_are_standard_negative_are_generated(
        dir_model_name, 
        top_k_documents, 
        top_k_value, 
        corpusid_2_context, 
        prompt_template, 
        question, 
        proposed_question_dict, 
        # all_clueid2docid2senidlist, 
        question_type="original"
    ):
    
    top_k_corpusid_2_context = {cur_id: corpusid_2_context[cur_id] for cur_id in top_k_documents[:top_k_value]}
    clue_str = list_to_docided_string(top_k_corpusid_2_context)
    
    cur_prompt = prompt_template.replace('[[QUESTION]]', question)
    cur_prompt = cur_prompt.replace('[[CLUES]]', clue_str)
        
    # Question, Positive Answer, Negative Answer
    question_prompt = cur_prompt # question
    if "positive" in proposed_question_dict:
        positive_answer = proposed_question_dict['positive'] # original question
    else:
        positive_answer = proposed_question_dict['answer'] # rephrased question
    # positive_answer = proposed_question_dict['answer'] # positive answer
    # positive_answer = replace_clue_with_doc_and_sen(all_clueid2docid2senidlist, positive_answer) # Replace [Clue xx] with the actual Sen like [Doc xx, Sen xx, xx]

    answer_key_name = dir_model_name + "-top_k_value-" + str(top_k_value) + "-answer"
    if answer_key_name not in proposed_question_dict:
        return None
    negative_answer = proposed_question_dict[answer_key_name] # negative answer

    # negative_answer, _ = openai_model.generate(CLIENT, question_prompt) # negative answer, TODO

    # retrieval_status = "Retrieval is correct; positive samples are standard answers, and negative samples are generated answers with incorrect citations."
    if question_prompt and positive_answer and negative_answer:
        return {
            "prompt": question_prompt,
            "chosen": positive_answer,
            "rejected": negative_answer,
            "data_gen_type": f"{question_type}-retrieval_true-positive_are_standard-negative_are_generated",
            "top_k_value": top_k_value
        }
    return None

def retreival_false_positive_are_reject_negative_are_generated(
        dir_model_name,
        top_k_documents, 
        top_k_value, 
        corpusid_2_context, 
        prompt_template, 
        question, 
        proposed_question_dict, 
        # all_clueid2docid2senidlist, 
        question_type="original"
    ):

    top_k_corpusid_2_context = {cur_id: corpusid_2_context[cur_id] for cur_id in top_k_documents[:top_k_value]}
    clue_str = list_to_docided_string(top_k_corpusid_2_context)

    cur_prompt = prompt_template.replace('[[QUESTION]]', question)
    cur_prompt = cur_prompt.replace('[[CLUES]]', clue_str)

    # Question, Positive Answer, Negative Answer
    question_prompt = cur_prompt # question
    positive_answer = random.choice(rejection_statements_to_choose_from) # reject or without any citation
    
    answer_key_name = dir_model_name + "-top_k_value-" + str(top_k_value) + "-answer"
    if answer_key_name not in proposed_question_dict:
        return None
    negative_answer = proposed_question_dict[answer_key_name] # negative answer

    # negative_answer, _ = openai_model.generate(CLIENT, question_prompt) # negative answer, TODO
    
    if question_prompt and positive_answer and negative_answer:
        return {
            "prompt": question_prompt,
            "chosen": positive_answer,
            "rejected": negative_answer,
            "data_gen_type": f"{question_type}-retreival_false-positive_are_reject-negative_are_generated",
            "top_k_value": top_k_value
        }
    return None

def get_train_data(args, train_dataset_checkpoint_path) -> Dataset:
    """Load the dataset and convert it to the necessary format with checkpointing."""

    all_datasets = []
    if os.path.exists(train_dataset_checkpoint_path):
        rel_train_dataset_checkpoint_path = os.path.relpath(train_dataset_checkpoint_path, CUSTOM_CORPUS_HOME)
        print(f"Resuming from checkpoint: {rel_train_dataset_checkpoint_path}")
        with open(train_dataset_checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
        all_datasets, checkpoint_state = checkpoint_data["data"], checkpoint_data["state"]
    else:
        checkpoint_state = {
            "train_file_idx": 0,
            "data_idx": 0,
            "rel_inputpath": "",
        }
    processed_proposed_questions = 0

    with open(args.prompt_path, 'r') as f:
        prompt_template = f.read()

    for i, (corpuspath, inputpaths) in enumerate(args.corpuspath_2_inputpaths.items()):
        
        with open(corpuspath, 'r') as f:
            corpus_data = json.load(f)
        corpusid_2_context = {cur_dict['id']: cur_dict['context'] for cur_dict in corpus_data}
        
        for j, inputpath in enumerate(inputpaths):
            train_file_idx = i * len(inputpaths) + j
            if train_file_idx < checkpoint_state.get("train_file_idx", 0):
                continue

            with open(inputpath, "r") as f:
                json_data = json.load(f)
            rel_inputpath = os.path.relpath(inputpath, CUSTOM_CORPUS_HOME)
            checkpoint_state['inputpath'] = rel_inputpath

            if "content" in rel_inputpath:
                data_list = json_data
            elif "entity_graph" in rel_inputpath:
                data_list = list(json_data.values())
            else:
                raise ValueError(f"Unknown data file: {rel_inputpath}")

            for data_idx, data_item in enumerate(tqdm(data_list, desc="Processing File", total=len(data_list), dynamic_ncols=True)):
                if data_idx < checkpoint_state.get("data_idx", 0):
                    continue

                try:
                    if "content" in rel_inputpath:
                        if 'proposed-questions' not in data_item:
                            continue
                        proposed_questions = data_item['proposed-questions']
                        chunk_id = data_item['id']  # admission.stanford.edu.filter_index.htm.md_chunk_0

                        # all_clueid2docid2senidlist = {}
                        # objective_facts = data_item["objective-facts"]
                        # sens = data_item["sens"]
                        # for (fact_id, objective_fact), sen in zip(enumerate(objective_facts, start=1), sens):
                        #     sen_ids = re.findall(r'\d+-\d+|\d+', sen)
                        #     sen_ids = expand_numbers_and_ranges(sen_ids)
                        #     all_clueid2docid2senidlist[fact_id] = {
                        #         chunk_id: sen_ids
                        #     }

                    elif "entity_graph" in rel_inputpath:
                        if 'proposed-questions' not in data_item:
                            continue
                        proposed_questions = data_item['proposed-questions']
                        objective_relationships = data_item['selected-relationships']['objective-relationships']

                        # all_clueid2docid2senidlist = {}
                        # for (relationship_id, objective_relationship_dict) in enumerate(objective_relationships, start=1):
                        #     docid = objective_relationship_dict['id']
                        #     sen_ids = re.findall(r'\d+-\d+|\d+', objective_relationship_dict["sentences_used"])
                        #     sen_ids = expand_numbers_and_ranges(sen_ids)
                        #     all_clueid2docid2senidlist[relationship_id] = {
                        #         docid: sen_ids
                        #     }
                    else:
                        raise ValueError(f"Unknown data file: {rel_inputpath}")

                    processing_args = []
                    for proposed_question_type, proposed_question_dict in proposed_questions.items():

                        processing_args.append((
                            args.dir_model_name,
                            proposed_question_dict,
                            rel_inputpath,
                            chunk_id if "content" in rel_inputpath else None,
                            # all_clueid2docid2senidlist,
                            corpusid_2_context,
                            prompt_template
                        ))
                    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                        futures_to_args = {
                            executor.submit(
                                process_each_proposed_question,
                                args_tuple
                            ): args_tuple for args_tuple in processing_args
                        }

                        # for future in tqdm(as_completed(futures_to_args), total=len(futures_to_args), desc="Processing Futures", dynamic_ncols=True):
                        for future in as_completed(futures_to_args):

                            results = future.result(timeout=10*60)
                            all_datasets.extend(results)
                            processed_proposed_questions += 1
                            if processed_proposed_questions % args.save_interval == 0:
                                with open(train_dataset_checkpoint_path, "w") as f:
                                    json.dump({"data": all_datasets, "state": checkpoint_state}, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    with open(train_dataset_checkpoint_path, "w") as f:
                        json.dump({"data": all_datasets, "state": checkpoint_state}, f, indent=2, ensure_ascii=False)
                    print(f"An error occurred: {e}")
                    raise  # Re-raise the exception after saving the checkpoint

                checkpoint_state["data_idx"] = data_idx + 1

            # After processing each corpus, reset data_idx
            checkpoint_state["data_idx"] = 0
            # Update train_file_idx for the next corpus
            checkpoint_state["train_file_idx"] = train_file_idx + 1

    return all_datasets

def process_each_proposed_question(args_tuple):
    (
        dir_model_name,
        proposed_question_dict,
        rel_inputpath,
        chunk_id,
        # all_clueid2docid2senidlist,
        corpusid_2_context,
        prompt_template
    ) = args_tuple

    results = []

    if 'top_k_documents' in proposed_question_dict:
        top_k_documents = proposed_question_dict['top_k_documents']
        original_question = proposed_question_dict['question']
        for top_k_value in args.top_k_values:
            all_in_top_k_key_name = f"all_in_top_{top_k_value}"
            if proposed_question_dict[all_in_top_k_key_name]:
                tmp_dict = retreival_true_positive_are_standard_negative_are_generated(
                    dir_model_name,
                    top_k_documents, 
                    top_k_value, 
                    corpusid_2_context, 
                    prompt_template, 
                    original_question, 
                    proposed_question_dict, 
                    # all_clueid2docid2senidlist, 
                    question_type="original"
                )
                if tmp_dict:
                    results.append(tmp_dict)

            elif not proposed_question_dict[all_in_top_k_key_name]:
                if "content" in rel_inputpath:
                    tmp_dict = retreival_false_positive_are_reject_negative_are_generated(
                        dir_model_name,
                        top_k_documents, 
                        top_k_value, 
                        corpusid_2_context, 
                        prompt_template, 
                        original_question, 
                        proposed_question_dict, 
                        # all_clueid2docid2senidlist, 
                        question_type="original"
                    )
                    if tmp_dict:
                        results.append(tmp_dict)
                elif "entity_graph" in rel_inputpath:
                    # needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'])
                    # needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                    # needed_corpusids = []
                    # for relationship_id in needed_objective_relationship_ids:
                    #     if relationship_id in all_clueid2docid2senidlist:
                    #         needed_corpusids.extend(list(all_clueid2docid2senidlist[relationship_id].keys()))
                    # needed_corpusids = list(sorted(list(set(needed_corpusids))))
                    # needed_corpusids = [needed_corpusid for needed_corpusid in needed_corpusids if needed_corpusid in top_k_documents[:top_k_value]]
                    
                    # if not needed_corpusids: # omit when only part standard answer intersects with top_k_documents
                    tmp_dict = retreival_false_positive_are_reject_negative_are_generated(
                        dir_model_name,
                        top_k_documents, 
                        top_k_value, 
                        corpusid_2_context, 
                        prompt_template, 
                        original_question, 
                        proposed_question_dict, 
                        # all_clueid2docid2senidlist, 
                        question_type="original"
                    )
                    if tmp_dict:
                        results.append(tmp_dict)
                else:
                    raise ValueError(f"Unknown data_gen_type: {proposed_question_dict[all_in_top_k_key_name]}")
                
            else:
                raise ValueError(f"Unknown data_gen_type: {proposed_question_dict[all_in_top_k_key_name]}")

    rephrased_question_type_list = ['rephrased-questions', 'rephrased-questions-part', 'rephrased-questions-hybrid']
    only_gen_at_rephrased_poses_list = [args.only_gen_at_rephrased_poses, args.only_gen_at_rephrased_poses_part, args.only_gen_at_rephrased_poses_hybrid]
    for rephrased_question_type, only_gen_at_rephrased_poses in zip(rephrased_question_type_list, only_gen_at_rephrased_poses_list):
        rephrased_questions = proposed_question_dict.get(rephrased_question_type, [])
        rephrased_questions_filtered = []
        for only_eval_at_rephrased_pos in only_gen_at_rephrased_poses:
            if only_eval_at_rephrased_pos < len(rephrased_questions):
                rephrased_questions_filtered.append(rephrased_questions[only_eval_at_rephrased_pos])
        for rephrased_question_dict in rephrased_questions_filtered:

            if "top_k_documents" not in rephrased_question_dict:
                continue
            top_k_documents = rephrased_question_dict['top_k_documents']
            if 'reordered-question' in rephrased_question_dict:
                question_str = rephrased_question_dict['reordered-question']
            else:
                question_str = rephrased_question_dict['result']

            for top_k_value in args.top_k_values:

                all_in_top_k_key_name = f"all_in_top_{top_k_value}"

                if rephrased_question_dict[all_in_top_k_key_name]:
                    tmp_dict = retreival_true_positive_are_standard_negative_are_generated(
                        dir_model_name,
                        top_k_documents, 
                        top_k_value, 
                        corpusid_2_context, 
                        prompt_template, 
                        question_str, 
                        rephrased_question_dict, 
                        # all_clueid2docid2senidlist, 
                        question_type="rephrased_question_type"
                    )
                    if tmp_dict:
                        results.append(tmp_dict)

                elif not rephrased_question_dict[all_in_top_k_key_name]:
                    if "content" in rel_inputpath:
                        needed_corpusids = [chunk_id]
                    elif "entity_graph" in rel_inputpath:
                        # needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'])
                        # needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                        # needed_corpusids = list(
                        #     sorted(list(set([list(all_clueid2docid2senidlist[relationship_id].keys())[0] for relationship_id in needed_objective_relationship_ids]))))
                        # needed_corpusids = [needed_corpusid for needed_corpusid in needed_corpusids if needed_corpusid in top_k_documents[:top_k_value]]

                        # if not needed_corpusids: # omit when only part standard answer intersects with top_k_documents
                        tmp_dict = retreival_false_positive_are_reject_negative_are_generated(
                            dir_model_name,
                            top_k_documents, 
                            top_k_value, 
                            corpusid_2_context, 
                            prompt_template, 
                            question_str, 
                            rephrased_question_dict, 
                            # all_clueid2docid2senidlist, 
                            question_type="rephrased_question_type"
                        )
                        if tmp_dict:
                            results.append(tmp_dict)
                    else:
                        raise ValueError(f"Unknown data_gen_type: {rephrased_question_dict[all_in_top_k_key_name]}")
                else:
                    raise ValueError(f"Unknown data_gen_type: {rephrased_question_dict[all_in_top_k_key_name]}")

    return results

def config_args(args):

    # time_str = datetime.now().strftime('%Y_%b%d_%H-%M-%S')
    
    # input path
    args.corpuspath_2_inputpaths = json.loads(args.corpuspath_2_inputpaths.replace("'", "\""))
    new_corpuspath_2_inputpaths = {}
    for chunk_path_key in args.corpuspath_2_inputpaths:
        new_corpuspath_2_inputpaths[os.path.join(CUSTOM_CORPUS_HOME, chunk_path_key)] = [
            os.path.join(CUSTOM_CORPUS_HOME, inputpath) for inputpath in args.corpuspath_2_inputpaths[chunk_path_key]
        ]
    args.corpuspath_2_inputpaths = new_corpuspath_2_inputpaths

    # prompt_template
    args.prompt_path = os.path.join(CUSTOM_CORPUS_HOME, args.prompt_path)

    # output path
    args.output_dir = os.path.join(CUSTOM_CORPUS_HOME, args.output_dir)
    # args.output_dir = os.path.join(CUSTOM_CORPUS_HOME, args.output_dir, time_str)
    os.makedirs(args.output_dir, exist_ok=True)

    # config
    config_path = os.path.join(args.output_dir, "gen_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    return args

if __name__ == "__main__":

    args = parse_args()
    args = config_args(args)

    set_seed(args.seed)

    train_dataset_checkpoint_path = os.path.join(args.output_dir, "checkpoint.json")
    train_dataset = get_train_data(args, train_dataset_checkpoint_path)
    
    rel_output_dir = os.path.relpath(args.output_dir, CUSTOM_CORPUS_HOME)
    print(f"Saving the dataset to {rel_output_dir}")
    train_dataset_output_path = os.path.join(args.output_dir, "train_dataset.json")
    with open(train_dataset_output_path, "w") as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)
    if os.path.exists(train_dataset_checkpoint_path):
        os.remove(train_dataset_checkpoint_path)