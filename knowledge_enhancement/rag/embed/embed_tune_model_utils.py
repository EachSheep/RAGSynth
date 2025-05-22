import re
import random

from rag.utils import (
    expand_numbers_and_ranges,
    extract_doc_to_sen
)

def add_eos_func(input_examples, model):
    input_examples = [input_example + model.tokenizer.eos_token for input_example in input_examples]
    return input_examples

def get_query_positive_map_from_contents(question_type, chunk_dicts, corpuspath_2_query2positive, corpuspath, args, model):
    train_example_num = 0
    if question_type == "content":
        for chunk_idx, chunk_dict in enumerate(chunk_dicts):
            proposed_questions = chunk_dict.get("proposed-questions", {})
            if not proposed_questions:
                continue
            positive_doc_ids = [chunk_dict["id"]]

            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                # if "question" in proposed_question_dict and proposed_question_dict["question"]:
                #     original_question = proposed_question_dict["question"]
                #     if args.query_prefix:
                #         original_question = args.query_prefix + original_question
                #     if args.add_eos:
                #         original_question = add_eos_func([original_question], model)[0]
                #     if corpuspath not in corpuspath_2_query2positive:
                #         corpuspath_2_query2positive[corpuspath] = {
                #             original_question: {
                #                 "positive_indices": positive_doc_ids,
                #                 "question_type": question_type,
                #                 "rephrased_question_type": "original"
                #             }
                #         }
                #     else:
                #         corpuspath_2_query2positive[corpuspath][original_question] = {
                #                 "positive_indices": positive_doc_ids,
                #                 "question_type": question_type,
                #                 "rephrased_question_type": "original"
                #             }
                #     train_example_num += 1

                if not args.not_use_rephrased:
                    rephrased_questions = proposed_question_dict.get("rephrased-questions", [])
                    for i, rephrased_question_dict in enumerate(rephrased_questions):
                        if i < len(rephrased_questions) - 1:
                            continue
                        if "result" in rephrased_question_dict and rephrased_question_dict["result"]:
                            rephrased_question = rephrased_question_dict["result"]
                            if args.query_prefix:
                                rephrased_question = args.query_prefix + rephrased_question
                            if args.add_eos:
                                rephrased_question = add_eos_func([rephrased_question], model)[0]
                            if corpuspath not in corpuspath_2_query2positive:
                                corpuspath_2_query2positive[corpuspath] = {
                                    rephrased_question: {
                                        "positive_indices": positive_doc_ids,
                                        "question_type": question_type,
                                        "rephrased_question_type": "rephrased"
                                    }
                                }
                            else:
                                corpuspath_2_query2positive[corpuspath][rephrased_question] = {
                                    "positive_indices": positive_doc_ids,
                                    "question_type": question_type,
                                    "rephrased_question_type": "rephrased"
                                }
                            train_example_num += 1

                if not args.not_use_rephrased_part:
                    rephrased_questions_part = proposed_question_dict.get("rephrased-questions-part", [])
                    for i, rephrased_question_part_dict in enumerate(rephrased_questions_part):
                        if i < len(rephrased_questions_part) - 1:
                            continue
                        # if "result" in rephrased_question_part_dict and rephrased_question_part_dict["result"]:
                        #     rephrased_question_part = rephrased_question_part_dict["result"]
                        #     if args.query_prefix:
                        #         rephrased_question_part = args.query_prefix + rephrased_question_part
                        #     if args.add_eos:
                        #         rephrased_question_part = add_eos_func([rephrased_question_part], model)[0]
                        #     if corpuspath not in corpuspath_2_query2positive:
                        #         corpuspath_2_query2positive[corpuspath] = {
                        #             rephrased_question_part : {
                        #                 "positive_indices": positive_doc_ids, 
                        #                 "question_type": question_type, 
                        #                 "rephrased_question_type": "rephrased-part"
                        #             }
                        #         }
                        #     else:
                        #         corpuspath_2_query2positive[corpuspath][rephrased_question_part] = {
                        #                 "positive_indices": positive_doc_ids, 
                        #                 "question_type": question_type, 
                        #                 "rephrased_question_type": "rephrased-part"
                        #             }
                        #     train_example_num += 1

                        if "reordered-question" in rephrased_question_part_dict and rephrased_question_part_dict["reordered-question"]:
                            rephrased_question_part_reordered = rephrased_question_part_dict["reordered-question"]
                            if args.query_prefix:
                                rephrased_question_part_reordered = args.query_prefix + rephrased_question_part_reordered
                            if args.add_eos:
                                rephrased_question_part_reordered = add_eos_func([rephrased_question_part_reordered], model)[0]
                            if corpuspath not in corpuspath_2_query2positive:
                                corpuspath_2_query2positive[corpuspath] = {
                                    rephrased_question_part_reordered: {
                                        "positive_indices": positive_doc_ids,
                                        "question_type": question_type,
                                        "rephrased_question_type": "rephrased-part-reordered"
                                    }
                                }
                            else:
                                corpuspath_2_query2positive[corpuspath][rephrased_question_part_reordered] = {
                                        "positive_indices": positive_doc_ids, 
                                        "question_type": question_type, 
                                        "rephrased_question_type": "rephrased-part-reordered"
                                    }
                            train_example_num += 1

                if not args.not_use_rephrased_hybrid:
                    rephrased_questions_hybrid = proposed_question_dict.get("rephrased-questions-hybrid", [])
                    for i, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid):
                        if i < len(rephrased_questions_hybrid) - 1:
                            continue
                        # if "result" in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict["result"]:
                        #     rephrased_question_hybrid = rephrased_question_hybrid_dict["result"]
                        #     if args.query_prefix:
                        #         rephrased_question_hybrid = args.query_prefix + rephrased_question_hybrid
                        #     if args.add_eos:
                        #         rephrased_question_hybrid = add_eos_func([rephrased_question_hybrid], model)[0]
                        #     if corpuspath not in corpuspath_2_query2positive:
                        #         corpuspath_2_query2positive[corpuspath] = {
                        #             rephrased_question_hybrid: {
                        #                 "positive_indices": positive_doc_ids, 
                        #                 "question_type": question_type, 
                        #                 "rephrased_question_type": "rephrased-hybrid"
                        #             }
                        #         }
                        #     else:
                        #         corpuspath_2_query2positive[corpuspath][rephrased_question_hybrid] = {
                        #                 "positive_indices": positive_doc_ids, 
                        #                 "question_type": question_type, 
                        #                 "rephrased_question_type": "rephrased-hybrid"
                        #             }
                        #     train_example_num += 1

                        if "reordered-question" in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict["reordered-question"]:
                            rephrased_question_hybrid_reordered = rephrased_question_hybrid_dict["reordered-question"]
                            if args.query_prefix:
                                rephrased_question_hybrid_reordered = args.query_prefix + rephrased_question_hybrid_reordered
                            if args.add_eos:
                                rephrased_question_hybrid_reordered = add_eos_func([rephrased_question_hybrid_reordered], model)[0]
                            if corpuspath not in corpuspath_2_query2positive:
                                corpuspath_2_query2positive[corpuspath] = {
                                    rephrased_question_hybrid_reordered: {
                                        "positive_indices": positive_doc_ids, 
                                        "question_type": question_type, 
                                        "rephrased_question_type": "rephrased-hybrid-reordered"
                                    }
                                }
                            else:
                                corpuspath_2_query2positive[corpuspath][rephrased_question_hybrid_reordered] = {
                                        "positive_indices": positive_doc_ids, 
                                        "question_type": question_type, 
                                        "rephrased_question_type": "rephrased-hybrid-reordered"
                                    }
                            train_example_num += 1

    elif question_type == "entity_graph":
        for entity_dict in chunk_dicts.values():
            proposed_questions = entity_dict.get("proposed-questions", {})
            if not proposed_questions:
                continue

            proposed_questions = entity_dict['proposed-questions']
            # objective_relationships = entity_dict['selected-relationships']['objective-relationships']
            # objective_relationship_id_2_objective_relationship = {idx: fact for idx, fact in enumerate(objective_relationships, start=1)}

            for proposed_question_type, proposed_question_dict in proposed_questions.items():
                # needed_objective_relationship_ids = re.findall(r'\d+-\d+|\d+', proposed_question_dict['objective-relationship-id'])
                # needed_objective_relationship_ids = expand_numbers_and_ranges(needed_objective_relationship_ids)
                # needed_related_relationships = [objective_relationship_id_2_objective_relationship[int(clue_id)] for clue_id in needed_objective_relationship_ids if clue_id and int(clue_id) in objective_relationship_id_2_objective_relationship]
                # needed_corpusids = list(set([cur_related_relationship['id'] for cur_related_relationship in needed_related_relationships if 'id' in cur_related_relationship]))
                if not "positive" in proposed_question_dict or not proposed_question_dict["positive"]:
                    continue
                positive = proposed_question_dict["positive"]
                needed_corpusid_2_sens = extract_doc_to_sen(positive)
                needed_corpusids = list(needed_corpusid_2_sens.keys())

                # if "question" in proposed_question_dict and proposed_question_dict["question"]:
                #     original_question = proposed_question_dict["question"]
                #     if args.query_prefix:
                #         original_question = args.query_prefix + original_question
                #     if args.add_eos:
                #         original_question = add_eos_func([original_question], model)[0]
                #     if corpuspath not in corpuspath_2_query2positive:
                #         corpuspath_2_query2positive[corpuspath] = {
                #             original_question: {
                #                 "positive_indices": needed_corpusids, 
                #                 "question_type": question_type, 
                #                 "rephrased_question_type": "original"
                #             }
                #         }
                #     else:
                #         corpuspath_2_query2positive[corpuspath][original_question] ={
                #                 "positive_indices": needed_corpusids, 
                #                 "question_type": question_type, 
                #                 "rephrased_question_type": "original"
                #             }
                #     train_example_num += 1
                
                if not args.not_use_rephrased:
                    rephrased_questions = proposed_question_dict.get("rephrased-questions", [])
                    for i, rephrased_question_dict in enumerate(rephrased_questions):
                        if i < len(rephrased_questions) - 1:
                            continue
                        if "result" in rephrased_question_dict and rephrased_question_dict["result"]:
                            rephrased_question = rephrased_question_dict["result"]
                            if args.query_prefix:
                                rephrased_question = args.query_prefix + rephrased_question
                            if args.add_eos:
                                rephrased_question = add_eos_func([rephrased_question], model)[0]
                            if corpuspath not in corpuspath_2_query2positive:
                                corpuspath_2_query2positive[corpuspath] = {
                                    rephrased_question: {
                                        "positive_indices": needed_corpusids, 
                                        "question_type": question_type, 
                                        "rephrased_question_type": "rephrased"
                                    }
                                }
                            else:
                                corpuspath_2_query2positive[corpuspath][rephrased_question] = {
                                        "positive_indices": needed_corpusids, 
                                        "question_type": question_type, 
                                        "rephrased_question_type": "rephrased"
                                    }
                            train_example_num += 1
                
                if not args.not_use_rephrased_part:
                    rephrased_questions_part = proposed_question_dict.get("rephrased-questions-part", [])
                    for i, rephrased_question_part_dict in enumerate(rephrased_questions_part):
                        if i < len(rephrased_questions_part) - 1:
                            continue
                        # if "result" in rephrased_question_part_dict and rephrased_question_part_dict["result"]:
                        #     rephrased_question_part = rephrased_question_part_dict["result"]
                        #     if args.query_prefix:
                        #         rephrased_question_part = args.query_prefix + rephrased_question_part
                        #     if args.add_eos:
                        #         rephrased_question_part = add_eos_func([rephrased_question_part], model)[0]
                        #     if corpuspath not in corpuspath_2_query2positive:
                        #         corpuspath_2_query2positive[corpuspath] = {
                        #             rephrased_question_part: {
                        #                 "positive_indices": needed_corpusids, 
                        #                 "question_type": question_type, 
                        #                 "rephrased_question_type": "rephrased-part"
                        #             }
                        #         }
                        #     else:
                        #         corpuspath_2_query2positive[corpuspath][rephrased_question_part] = {
                        #                 "positive_indices": needed_corpusids, 
                        #                 "question_type": question_type, 
                        #                 "rephrased_question_type": "rephrased-part"
                        #             }
                        #     train_example_num += 1

                        if "reordered-question" in rephrased_question_part_dict and rephrased_question_part_dict["reordered-question"]:
                            rephrased_question_part_reordered = rephrased_question_part_dict["reordered-question"]
                            if args.query_prefix:
                                rephrased_question_part_reordered = args.query_prefix + rephrased_question_part_reordered
                            if args.add_eos:
                                rephrased_question_part_reordered = add_eos_func([], model)[0]
                            if corpuspath not in corpuspath_2_query2positive:
                                corpuspath_2_query2positive[corpuspath] = {
                                    rephrased_question_part_reordered: {
                                        "positive_indices": needed_corpusids, 
                                        "question_type": question_type, 
                                        "rephrased_question_type": "rephrased-part-reordered"
                                    }
                                }
                            else:
                                corpuspath_2_query2positive[corpuspath][rephrased_question_part_reordered] = {
                                        "positive_indices": needed_corpusids, 
                                        "question_type": question_type, 
                                        "rephrased_question_type": "rephrased-part-reordered"
                                    }
                            train_example_num += 1
                
                if not args.not_use_rephrased_hybrid:
                    rephrased_questions_hybrid = proposed_question_dict.get("rephrased-questions-hybrid", [])
                    for i, rephrased_question_hybrid_dict in enumerate(rephrased_questions_hybrid):
                        if i < len(rephrased_questions_hybrid) - 1:
                            continue
                        # if "result" in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict["result"]:
                        #     rephrased_question_hybrid = rephrased_question_hybrid_dict["result"]
                        #     if args.query_prefix:
                        #         rephrased_question_hybrid = args.query_prefix + rephrased_question_hybrid
                        #     if args.add_eos:
                        #         rephrased_question_hybrid = add_eos_func([rephrased_question_hybrid], model)[0]
                        #     if corpuspath not in corpuspath_2_query2positive:
                        #         corpuspath_2_query2positive[corpuspath] = {
                        #             rephrased_question_hybrid: {
                        #                 "positive_indices": needed_corpusids, 
                        #                 "question_type": question_type, 
                        #                 "rephrased_question_type": "rephrased-hybrid"
                        #             }
                        #         }
                        #     else:
                        #         corpuspath_2_query2positive[corpuspath][rephrased_question_hybrid] = {
                        #                 "positive_indices": needed_corpusids, 
                        #                 "question_type": question_type, 
                        #                 "rephrased_question_type": "rephrased-hybrid"
                        #             }
                        #     train_example_num += 1

                        if "reordered-question" in rephrased_question_hybrid_dict and rephrased_question_hybrid_dict["reordered-question"]:
                            rephrased_question_hybrid_reordered = rephrased_question_hybrid_dict["reordered-question"]
                            if args.query_prefix:
                                rephrased_question_hybrid_reordered = args.query_prefix + rephrased_question_hybrid_reordered
                            if args.add_eos:
                                rephrased_question_hybrid_reordered = add_eos_func([rephrased_question_hybrid_reordered], model)[0]
                            if corpuspath not in corpuspath_2_query2positive:
                                corpuspath_2_query2positive[corpuspath] = {
                                    rephrased_question_hybrid_reordered: {
                                        "positive_indices": needed_corpusids,
                                        "question_type": question_type, 
                                        "rephrased_question_type": "rephrased-hybrid-reordered"
                                    }
                                }
                            else:
                                corpuspath_2_query2positive[corpuspath][rephrased_question_hybrid_reordered] = {
                                        "positive_indices": needed_corpusids,
                                        "question_type": question_type, 
                                        "rephrased_question_type": "rephrased-hybrid-reordered"
                                    }
                            train_example_num += 1
    else:
        raise ValueError(f"Invalid question type: {args.question_type}")
    
    # limit the number of training examples for each dataset
    if args.max_train_num and args.max_train_num > 0:
        train_example_num = 0
        for corpuspath in corpuspath_2_query2positive:
            query2positive = corpuspath_2_query2positive[corpuspath]
            if len(query2positive) > args.max_train_num:
                query2positive = dict(random.sample(list(query2positive.items()), args.max_train_num))
                corpuspath_2_query2positive[corpuspath] = query2positive
                train_example_num += len(query2positive)

    return train_example_num

def calculate_all_embeddings(contexts_to_process, model, normalize_embeddings=False, add_eos=False, batch_size=2):
    """
    Encode contents and save all embeddings into a single .npy file, preserving directory structure, with batch processing.
    """
    if add_eos:
        contexts_to_process = add_eos_func(contexts_to_process, model)
    embeddings = model.encode(contexts_to_process, batch_size=batch_size, normalize_embeddings=normalize_embeddings, show_progress_bar=False)
    return embeddings
