import os
import sys
import argparse  
from dotenv import load_dotenv

parser = argparse.ArgumentParser(description='Load specific .env files for propose and rephrase.')
parser.add_argument('--llm', default="llm.env", help='Path to llm.env')
parser.add_argument('--embed', default="embed.env", help='Path to embed.env')
parser.add_argument('--preprocess_env_path', default=None, help='Path to preprocess.env')
parser.add_argument('--postprocess_env_path', default=None, help='Path to postprocess.env')
parser.add_argument('--dataset_name', type=str, default="admission.stanford.edu.filter", help='Name of the dataset')
parser.add_argument('--tier', type=str, default="1", help='Tier of the dataset')
args = parser.parse_args()

load_dotenv(args.llm)
load_dotenv(args.embed)

from components.fact_extractor.fact_extractor import FactExtractor
from components.entity_extractor.entity_extractor import EntityExtractor
from components.entity_eliminator.entity_eliminator_simple import EntityEliminator
from components.entity_eliminator.entity_eliminator_evaluator import EntityEliminatorEvaluator
from components.add_entity_id.add_entity_id import AddEntityId
from components.propose_generator.propose_generator import ProposeGenerator
from components.rephrase_generator.rephrase_generator import RephraseGenerator
from components.rephrase_generator.rephrase_generator_part import RephraseGeneratorPart
from components.rephrase_generator.rephrase_generator_hybrid import RephraseGeneratorHybrid
from components.sentence_order_changer.sentence_order_changer import SentenceOrderChanger
from components.final_answer_generator.final_answer_generator import FinalAnswerGenerator

CUSTOM_CORPUS_HOME = os.getenv("CUSTOM_CORPUS_HOME", None)
if CUSTOM_CORPUS_HOME == None:
    raise ValueError("CUSTOM_CORPUS_HOME environment variable is not set")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.6))

def load_env_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            key, value = line.strip().split("=")
            replaced_value = value.replace("admission.stanford.edu.filter", args.dataset_name).replace(f"docs_tier_1", f"docs_tier_{args.tier}")
            os.environ[key] = replaced_value

def load_envs(args):  
    # Load always-present env files from command line args  
    additional_files = [  
        args.preprocess_env_path,
        args.postprocess_env_path,
    ]  
    
    for file_path in additional_files:  
        if file_path != None:
            print(f"Loading env file: {file_path}")
            load_env_file(file_path)

if __name__ == '__main__':
    load_envs(args)
    print("-" * 50)
    
    if os.getenv("FACT_EXTRACTOR_INPUT_PATH", None) != None:
        print("Running Fact Extractor")
        FACT_EXTRACTOR_SAVE_INTERVAL = int(os.getenv("FACT_EXTRACTOR_SAVE_INTERVAL", None))
        fact_extractor = FactExtractor(save_interval=FACT_EXTRACTOR_SAVE_INTERVAL)

        success_num, all_num = 0, 1
        max_tries = 3
        while success_num < all_num and max_tries > 0:
            success_num, all_num = fact_extractor.run()
            max_tries -= 1
        print("-" * 50)

    if os.getenv("ENTITY_EXTRACTOR_INPUT_PATH", None) != None:
        print("Running Entity Extractor")
        ENTITY_EXTRACTOR_SAVE_INTERVAL = int(os.getenv("ENTITY_EXTRACTOR_SAVE_INTERVAL", None))
        entity_extractor = EntityExtractor(save_interval=ENTITY_EXTRACTOR_SAVE_INTERVAL)

        success_num, all_num = 0, 1
        max_tries = 3
        while success_num < all_num and max_tries > 0:
            success_num, all_num = entity_extractor.run()
            max_tries -= 1
        print("-" * 50)

    if os.getenv("ADD_ENTITY_ID_INPUT_PATH", None) != None:
        print("Running Add Entity Id")
        add_entity_id = AddEntityId()
        add_entity_id.run()
        print("-" * 50)

    if os.getenv("ENTITY_ELIMINATOR_INPUT_PATH", None) != None:
        print("Running Entity Eliminator")
        entity_eliminator = EntityEliminator()
        entity_eliminator.run()
        print("-" * 50)

    if os.getenv("PROPOSE_GENERATOR_CONTENT_INPUT_PATH", None) != None or os.getenv("PROPOSE_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
        print("Running Propose Generator")
        PROPOSE_GENERATOR_SAVE_INTERVAL = int(os.getenv("PROPOSE_GENERATOR_SAVE_INTERVAL", None))
        propose_generator = ProposeGenerator(save_interval=PROPOSE_GENERATOR_SAVE_INTERVAL)

        success_num, all_num = 0, 1
        max_tries = 3
        while success_num < all_num and max_tries > 0:
            success_num, all_num = propose_generator.run()
            max_tries -= 1
        print("-" * 50)
    
    if os.getenv("FINAL_ANSWER_GENERATOR_CONTENT_INPUT_PATH", None) != None or os.getenv("FINAL_ANSWER_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
        print("Running Final Answer Generator")
        FINAL_ANSWER_GENERATOR_SAVE_INTERVAL = int(os.getenv("FINAL_ANSWER_GENERATOR_SAVE_INTERVAL", None))
        final_answer_generator = FinalAnswerGenerator(save_interval=FINAL_ANSWER_GENERATOR_SAVE_INTERVAL)
        success_num, all_num = 0, 1
        
        max_tries = 3
        while success_num < all_num and max_tries > 0:
            success_num, all_num = final_answer_generator.run()
            max_tries -= 1
        print("-" * 50)

    if os.getenv("REPHRASE_GENERATOR_CONTENT_INPUT_PATH", None) != None or os.getenv("REPHRASE_GENERATOR_ENTITYGRAPH_INPUT_PATH", None) != None:
        print("Running Rephrase Generator")
        REPHRASE_GENERATOR_SAVE_INTERVAL = int(os.getenv("REPHRASE_GENERATOR_SAVE_INTERVAL", None))
        rephrase_generator = RephraseGenerator(save_interval=REPHRASE_GENERATOR_SAVE_INTERVAL)

        success_num, all_num = 0, 1
        max_tries = 3
        while success_num < all_num and max_tries > 0:
            success_num, all_num = rephrase_generator.run()
            max_tries -= 1
        print("-" * 50)

    if os.getenv("REPHRASE_GENERATOR_PART_CONTENT_INPUT_PATH", None) != None or os.getenv("REPHRASE_GENERATOR_PART_ENTITYGRAPH_INPUT_PATH", None) != None:
        print("Running Rephrase Generator Part")
        REPHRASE_GENERATOR_SAVE_INTERVAL = int(os.getenv("REPHRASE_GENERATOR_SAVE_INTERVAL", None))
        rephrase_generator_part = RephraseGeneratorPart(save_interval=REPHRASE_GENERATOR_SAVE_INTERVAL)

        success_num, all_num = 0, 1
        max_tries = 3
        while success_num < all_num and max_tries > 0:
            success_num, all_num = rephrase_generator_part.run()
            max_tries -= 1
        print("-" * 50)

    if os.getenv("REPHRASE_GENERATOR_HYBRID_CONTENT_INPUT_PATH", None) != None or os.getenv("REPHRASE_GENERATOR_HYBRID_ENTITYGRAPH_INPUT_PATH", None) != None:
        print("Running Rephrase Generator Hybrid")
        REPHRASE_GENERATOR_SAVE_INTERVAL = int(os.getenv("REPHRASE_GENERATOR_SAVE_INTERVAL", None))
        rephrase_generator_hybrid = RephraseGeneratorHybrid(save_interval=REPHRASE_GENERATOR_SAVE_INTERVAL)

        success_num, all_num = 0, 1
        max_tries = 3
        while success_num < all_num and max_tries > 0:
            success_num, all_num = rephrase_generator_hybrid.run()
            max_tries -= 1
        print("-" * 50)

    if os.getenv("SENTENCE_ORDER_CHANGER_CONTENT_INPUT_PATH", None) != None or os.getenv("SENTENCE_ORDER_CHANGER_ENTITYGRAPH_INPUT_PATH", None) != None:
        print("Running Sentence Order Change")
        SENTENCE_ORDER_CHANGER_SAVE_INTERVAL = int(os.getenv("SENTENCE_ORDER_CHANGER_SAVE_INTERVAL", None))
        sentence_order_changer_hybrid = SentenceOrderChanger(save_interval=SENTENCE_ORDER_CHANGER_SAVE_INTERVAL)

        success_num, all_num = 0, 1
        max_tries = 3
        while success_num < all_num and max_tries > 0:
            success_num, all_num = sentence_order_changer_hybrid.run()
            max_tries -= 1
        print("-" * 50)