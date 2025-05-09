#!/bin/bash

python src/hierarchical_summary.py --source_folder docs/admission.stanford.edu.summary_and_extract_by_llm --target_folder wash_data_scripts/hierarchical_summary_output/
python src/hierarchical_summary.py --source_folder docs/berkeley.edu.admission.summary_and_extract_by_llm --target_folder wash_data_scripts/hierarchical_summary_output/

python src/hierarchical_summary.py --source_folder docs/docs.cyotek.com.summary_and_extract_by_llm --target_folder wash_data_scripts/hierarchical_summary_output/
python src/hierarchical_summary.py --source_folder docs/www.notion.so.help.summary_and_extract_by_llm --target_folder wash_data_scripts/hierarchical_summary_output/

python src/hierarchical_summary.py --source_folder docs/hearthstone.fandom.com.wiki.summary_and_extract_by_llm --target_folder wash_data_scripts/hierarchical_summary_output/
python src/hierarchical_summary.py --source_folder docs/zelda.fandom.com.wiki.summary_and_extract_by_llm --target_folder wash_data_scripts/hierarchical_summary_output/

python src/hierarchical_summary.py --source_folder docs/www.drugs.com.summary_and_extract_by_llm --target_folder wash_data_scripts/hierarchical_summary_output/
python src/hierarchical_summary.py --source_folder docs/www.mayoclinic.org.summary_and_extract_by_llm --target_folder wash_data_scripts/hierarchical_summary_output/