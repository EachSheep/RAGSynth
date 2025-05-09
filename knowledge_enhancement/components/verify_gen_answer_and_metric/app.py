# app.py

import os
import json
import re
from flask import Flask, request, jsonify, render_template
from argparse import ArgumentParser

from rag.utils import extract_doc_to_sen


CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME is None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")

app = Flask(__name__)

def eval_generation_results_for_file_content(args, data, corpusid_2_context):
    """
    A generator that processes the data and yields task dictionaries.
    Replace yield statements with returning dictionaries.
    """
    for cur_dict in data[:args.max_process_num if args.max_process_num != -1 else len(data)]:
        if 'proposed-questions' not in cur_dict:
            continue
        proposed_questions = cur_dict['proposed-questions']

        """calculate generation metrics"""
        chunk_id = cur_dict['id'] # admission.stanford.edu.filter_index.htm.md_chunk_0
        for proposed_question_type, proposed_question_dict in proposed_questions.items():
            
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
            needed_corpusid2senids = extract_doc_to_sen(positive)
            needed_corpusids = list(needed_corpusid2senids.keys())
            if chunk_id not in needed_corpusid2senids or not needed_corpusid2senids[chunk_id]:
                continue

            if not args.not_eval_for_original:

                original_question = proposed_question_dict['question']
                standard_answer = proposed_question_dict['positive']

                if 'check-question-tagged' in proposed_question_dict and proposed_question_dict['check-question-tagged']:
                    pass
                else:
                    task_dict = {
                        'needed_corpusid2corpus': needed_corpusid2corpus,
                        'needed_corpusid2senids': needed_corpusid2senids,
                        'original_question': original_question,
                        'rephrased_question_str': None,
                        'standard_answer': standard_answer,
                        'source_dict': proposed_question_dict,
                        'check-question-tagged': False
                    }
                    yield task_dict
                            
            tmp_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
            rephrased_questions = []
            rephrased_questions_indexes = []
            for only_eval_at_rephrased_pos in args.only_eval_at_rephrased_poses:
                if len(tmp_rephrased_questions) > only_eval_at_rephrased_pos:
                    rephrased_questions.append(tmp_rephrased_questions[only_eval_at_rephrased_pos])
                    rephrased_questions_indexes.append(only_eval_at_rephrased_pos + 1)
            for rephrased_question_type, rephrased_question_dict in zip(rephrased_questions_indexes, rephrased_questions):
                
                if "reordered-question" in rephrased_question_dict:
                    rephrased_question_str = rephrased_question_dict['reordered-question']
                else:
                    rephrased_question_str = rephrased_question_dict['result']
                standard_answer = rephrased_question_dict['answer']

                if 'check-question-tagged' in rephrased_question_dict and rephrased_question_dict['check-question-tagged']:
                    pass
                else:
                    task_dict = {
                        'needed_corpusid2corpus': needed_corpusid2corpus,
                        'needed_corpusid2senids': needed_corpusid2senids,
                        'original_question': original_question,
                        'rephrased_question_str': rephrased_question_str,
                        'standard_answer': standard_answer,
                        'source_dict': rephrased_question_dict,
                        'check-question-tagged': False
                    }
                    yield task_dict

            tmp_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
            rephrased_questions_part = []
            rephrased_questions_indexes_part = []
            for only_eval_at_rephrased_pos_part in args.only_eval_at_rephrased_poses_part:
                if len(tmp_rephrased_questions_part) > only_eval_at_rephrased_pos_part:
                    rephrased_questions_part.append(tmp_rephrased_questions_part[only_eval_at_rephrased_pos_part])
                    rephrased_questions_indexes_part.append(only_eval_at_rephrased_pos_part + 1)
            for rephrased_question_part_type, rephrased_question_part_dict in zip(rephrased_questions_indexes_part, rephrased_questions_part):

                if "reordered-question" in rephrased_question_part_dict:
                    rephrased_question_part_str = rephrased_question_part_dict['reordered-question']
                else:
                    rephrased_question_part_str = rephrased_question_part_dict['result']
                standard_answer = rephrased_question_part_dict['answer']

                if 'check-question-tagged' in rephrased_question_part_dict and rephrased_question_part_dict['check-question-tagged']:
                    pass
                else:
                    task_dict = {
                        'needed_corpusid2corpus': needed_corpusid2corpus,
                        'needed_corpusid2senids': needed_corpusid2senids,
                        'original_question': original_question,
                        'rephrased_question_str': rephrased_question_part_str,
                        'standard_answer': standard_answer,
                        'source_dict': rephrased_question_part_dict,
                        'check-question-tagged': False
                    }
                    yield task_dict

            tmp_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
            rephrased_questions_hybrid = []
            rephrased_questions_indexes_hybrid = []
            for only_eval_at_rephrased_pos_hybrid in args.only_eval_at_rephrased_poses_hybrid:
                if len(tmp_rephrased_questions_hybrid) > only_eval_at_rephrased_pos_hybrid:
                    rephrased_questions_hybrid.append(tmp_rephrased_questions_hybrid[only_eval_at_rephrased_pos_hybrid])
                    rephrased_questions_indexes_hybrid.append(only_eval_at_rephrased_pos_hybrid + 1)
            for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in zip(rephrased_questions_indexes_hybrid, rephrased_questions_hybrid):

                if "reordered-question" in rephrased_question_hybrid_dict:
                    rephrased_question_hybrid_str = rephrased_question_hybrid_dict['reordered-question']
                else:
                    rephrased_question_hybrid_str = rephrased_question_hybrid_dict['result']
                standard_answer = rephrased_question_hybrid_dict['answer']

                if 'check-question-tagged' in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict['check-question-tagged']:
                    pass
                else:
                    task_dict = {
                        'needed_corpusid2corpus': needed_corpusid2corpus,
                        'needed_corpusid2senids': needed_corpusid2senids,
                        'original_question': original_question,
                        'rephrased_question_str': rephrased_question_hybrid_str,
                        'standard_answer': standard_answer,
                        'source_dict': rephrased_question_hybrid_dict,
                        'check-question-tagged': False
                    }
                    yield task_dict

def eval_generation_results_for_file_entity_graph(args, data, corpusid_2_context):
    """
    Similar to eval_generation_results_for_file_content, but for entity_graph files.
    Implement accordingly.
    """
    for entity_dict in list(data.values())[:args.max_process_num if args.max_process_num != -1 else len(data)]:
        if 'proposed-questions' not in entity_dict:
            continue
        proposed_questions = entity_dict['proposed-questions']
        objective_relationships = entity_dict['selected-relationships']['objective-relationships']
        objective_relationship_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}

        """calculate generation metrics"""
        for proposed_question_type, proposed_question_dict in proposed_questions.items():

            # needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'])
            # needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
            # needed_related_relationships = [objective_relationship_id_2_objective_relationship[int(clue_id)] for clue_id in needed_objective_relationship_ids if clue_id and int(clue_id) in objective_relationship_id_2_objective_relationship]
            # needed_corpusids = list(set([cur_related_relationship['id'] for cur_related_relationship in needed_related_relationships if 'id' in cur_related_relationship]))
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

            if not args.not_eval_for_original:

                original_question = proposed_question_dict['question']
                standard_answer = proposed_question_dict['positive']

                if 'check-question-tagged' in proposed_question_dict and proposed_question_dict['check-question-tagged']:
                    pass
                else:
                    task_dict = {
                        'needed_corpusid2corpus': needed_corpusid2corpus,
                        'needed_corpusid2senids': needed_corpusid2senids,
                        'original_question': original_question,
                        'rephrased_question_str': None,
                        'standard_answer': standard_answer,
                        'source_dict': proposed_question_dict,
                        'check-question-tagged': False
                    }
                    yield task_dict

            tmp_rephrased_questions = proposed_question_dict.get('rephrased-questions', [])
            rephrased_questions = []
            rephrased_questions_indexes = []
            for only_eval_at_rephrased_pos in args.only_eval_at_rephrased_poses:
                if len(tmp_rephrased_questions) > only_eval_at_rephrased_pos:
                    rephrased_questions.append(tmp_rephrased_questions[only_eval_at_rephrased_pos])
                    rephrased_questions_indexes.append(only_eval_at_rephrased_pos + 1)
            for rephrased_question_type, rephrased_question_dict in zip(rephrased_questions_indexes, rephrased_questions):

                if "reordered-question" in rephrased_question_dict:
                    rephrased_question_str = rephrased_question_dict['reordered-question']
                else:
                    rephrased_question_str = rephrased_question_dict['result']
                standard_answer = rephrased_question_dict['answer']

                if 'check-question-tagged' in rephrased_question_dict and rephrased_question_dict['check-question-tagged']:
                    pass
                else:
                    task_dict = {
                        'needed_corpusid2corpus': needed_corpusid2corpus,
                        'needed_corpusid2senids': needed_corpusid2senids,
                        'original_question': original_question,
                        'rephrased_question_str': rephrased_question_str,
                        'standard_answer': standard_answer,
                        'source_dict': rephrased_question_dict,
                        'check-question-tagged': False
                    }
                    yield task_dict
                   
            tmp_rephrased_questions_part = proposed_question_dict.get('rephrased-questions-part', [])
            rephrased_questions_part = []
            rephrased_questions_indexes_part = []
            for only_eval_at_rephrased_pos_part in args.only_eval_at_rephrased_poses_part:
                if len(tmp_rephrased_questions_part) > only_eval_at_rephrased_pos_part:
                    rephrased_questions_part.append(tmp_rephrased_questions_part[only_eval_at_rephrased_pos_part])
                    rephrased_questions_indexes_part.append(only_eval_at_rephrased_pos_part + 1)
            for rephrased_question_part_type, rephrased_question_part_dict in zip(rephrased_questions_indexes_part, rephrased_questions_part):

                if "reordered-question" in rephrased_question_part_dict:
                    rephrased_question_part_str = rephrased_question_part_dict['reordered-question']
                else:
                    rephrased_question_part_str = rephrased_question_part_dict['result']
                standard_answer = rephrased_question_part_dict['answer']

                if 'check-question-tagged' in rephrased_question_part_dict and rephrased_question_part_dict['check-question-tagged']:
                    pass
                else:
                    task_dict = {
                        'needed_corpusid2corpus': needed_corpusid2corpus,
                        'needed_corpusid2senids': needed_corpusid2senids,
                        'original_question': original_question,
                        'rephrased_question_str': rephrased_question_part_str,
                        'standard_answer': standard_answer,
                        'source_dict': rephrased_question_part_dict,
                        'check-question-tagged': False
                    }
                    yield task_dict
            
            tmp_rephrased_questions_hybrid = proposed_question_dict.get('rephrased-questions-hybrid', [])
            rephrased_questions_hybrid = []
            rephrased_questions_indexes_hybrid = []
            for only_eval_at_rephrased_pos_hybrid in args.only_eval_at_rephrased_poses_hybrid:
                if len(tmp_rephrased_questions_hybrid) > only_eval_at_rephrased_pos_hybrid:
                    rephrased_questions_hybrid.append(tmp_rephrased_questions_hybrid[only_eval_at_rephrased_pos_hybrid])
                    rephrased_questions_indexes_hybrid.append(only_eval_at_rephrased_pos_hybrid + 1)
            for rephrased_question_hybrid_type, rephrased_question_hybrid_dict in zip(rephrased_questions_indexes_hybrid, rephrased_questions_hybrid):
                
                if "reordered-question" in rephrased_question_hybrid_dict:
                    rephrased_question_hybrid_str = rephrased_question_hybrid_dict['reordered-question']
                else:
                    rephrased_question_hybrid_str = rephrased_question_hybrid_dict['result']
                standard_answer = rephrased_question_hybrid_dict['answer']

                if 'check-question-tagged' in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict['check-question-tagged']:
                    pass
                else:
                    task_dict = {
                        'needed_corpusid2corpus': needed_corpusid2corpus,
                        'needed_corpusid2senids': needed_corpusid2senids,
                        'original_question': original_question,
                        'rephrased_question_str': rephrased_question_hybrid_str,
                        'standard_answer': standard_answer,
                        'source_dict': rephrased_question_hybrid_dict,
                        'check-question-tagged': False
                    }
                    yield task_dict

def load_tasks(args):
    traverse_directory(args, args.input_root_dir)
    print("sum(task_dicts):", sum([len(task_list) for task_list in task_dicts.values()]))

# Global variable to store all tasks
task_dicts = {}
task_index_dicts = {}
data_dicts = {}
save_intervals = [10]

def traverse_directory(args, cur_dir):
    
    global task_dicts
    for item in os.listdir(cur_dir):
        full_path = os.path.abspath(os.path.join(cur_dir, item))
        
        if os.path.isdir(full_path):
            traverse_directory(args, full_path)
        else:
            file_name = os.path.basename(full_path)
            relative_path = os.path.relpath(full_path, args.input_root_dir)

            if "rephrase_evaluator" not in file_name:
                continue

            print(f"Processing file {relative_path}")

            rel_path = os.path.relpath(full_path, args.input_root_dir)
            output_path = os.path.join(args.output_root_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    data = json.load(f)
            else:
                with open(full_path, 'r') as f:
                    data = json.load(f)

            input_file_dir = os.path.dirname(full_path)
            input_file_name = os.path.basename(full_path)
            input_file_name_suffix = input_file_name.split('.')[-2]
            chunk_path = os.path.join(input_file_dir, input_file_name.replace(input_file_name_suffix, "chunk_contents"))
            with open(chunk_path, 'r') as f:
                chunks_data = json.load(f)
            corpusid_2_context = {cur_dict['id']: cur_dict['context'] for cur_dict in chunks_data}
            
            data_dicts[output_path] = data
            if "content" in file_name and "contents" not in file_name:
                for result in eval_generation_results_for_file_content(args, data, corpusid_2_context):
                    # print(json.dumps(result, indent=2))
                    # input("Press Enter to continue...")
                    if output_path not in task_dicts:
                        task_dicts[output_path] = [result]
                        task_index_dicts[output_path] = 0
                    else:
                        task_dicts[output_path].append(result)
            elif "entity_graph" in file_name:
                for result in eval_generation_results_for_file_entity_graph(args, data, corpusid_2_context):
                    # print(json.dumps(result, indent=2))
                    # input("Press Enter to continue...")
                    if output_path not in task_dicts:
                        task_dicts[output_path] = [result]
                        task_index_dicts[output_path] = 0
                    else:
                        task_dicts[output_path].append(result)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/get_task', methods=['GET'])
def get_task():
    global task_index_dicts, task_dicts
    for output_path, cur_task_index in task_index_dicts.items():
        if cur_task_index < len(task_dicts[output_path]):
            task = task_dicts[output_path][cur_task_index]
            task_data = {
                'task_id': [output_path, cur_task_index],
                'original_question': task.get('original_question'),
                'rephrased_question_str': task.get('rephrased_question_str'),
                'standard_answer': task.get('standard_answer'),
                'needed_corpusid2corpus': task.get('needed_corpusid2corpus'),
                'needed_corpusid2senids': task.get('needed_corpusid2senids')
            }
            return jsonify(task_data), 200
    return jsonify({'status': 'no_more_tasks'}), 200

@app.route('/api/update_task', methods=['POST'])
def update_task_endpoint():
    global task_index_dicts, save_intervals, task_dicts, data_dicts
    data = request.json
    if not data:
        return jsonify({'status': 'failed', 'reason': 'No data provided.'}), 400
    
    task_id = data.get('task_id')
    if not task_id or len(task_id) != 2:
        return jsonify({'status': 'failed', 'reason': 'Invalid task_id format.'}), 400
    
    output_path, cur_task_index = task_id
    if output_path not in task_dicts:
        return jsonify({'status': 'failed', 'reason': 'Invalid output_path.'}), 400
    if not (0 <= cur_task_index < len(task_dicts[output_path])):
        return jsonify({'status': 'failed', 'reason': 'task_id out of range.'}), 400
    
    corrected_corpusid2senids = data.get('corrected_corpusid2senids', None)
    modified = data.get('modified', False)

    # Update task
    task = task_dicts[output_path][cur_task_index]
    if modified:
        if corrected_corpusid2senids is not None:
            try:
                corrected_corpusid2senids = json.loads(corrected_corpusid2senids)
                task['corrected_corpusid2senids'] = corrected_corpusid2senids
            except json.JSONDecodeError:
                return jsonify({'status': 'failed', 'reason': 'Invalid JSON format for corrected_corpusid2senids.'}), 400
    
    # Mark as check-question-tagged
    task["source_dict"]['check-question-tagged'] = True
    task['check-question-tagged'] = True

    # Increment task index
    task_index_dicts[output_path] += 1

    if task_index_dicts[output_path] % save_intervals[0] == 0 or task_index_dicts[output_path] >= len(task_dicts[output_path]):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dicts[output_path], f, indent=2, ensure_ascii=False)
    
    return jsonify({'status': 'success'}), 200

def parse_arguments():
    
    parser = ArgumentParser()
    
    parser.add_argument('--input_root_dir', type=str, required=True, help="Root directory containing input files.")
    
    parser.add_argument('--not_eval_for_original', action='store_true', help="Generate for original questions.")
    parser.add_argument('--only_eval_at_rephrased_poses', type=int, nargs='+', default=[], help="List of rephrased positions to evaluate.")
    parser.add_argument('--only_eval_at_rephrased_poses_part', type=int, nargs='+', default=[], help="List of rephrased positions part to evaluate.")
    parser.add_argument('--only_eval_at_rephrased_poses_hybrid', type=int, nargs='+', default=[], help="List of rephrased positions hybrid to evaluate.")

    parser.add_argument('--output_root_dir', type=str, default=None, help="Root folder to save the results.")
    parser.add_argument('--save_interval', type=int, default=100, help="The interval at which to save the results.")
    parser.add_argument('--max_process_num', type=int, default=-1, help="Maximum number of process data.")

    args = parser.parse_args()
    
    args.input_root_dir = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.input_root_dir))
    if args.output_root_dir:
        args.output_root_dir = os.path.abspath(os.path.join(CUSTOM_CORPUS_HOME, args.output_root_dir))

    return args

if __name__ == '__main__':
    args = parse_arguments()
    save_intervals[0] = args.save_interval
    load_tasks(args)
    app.run(debug=True, host='0.0.0.0', port=5000)
