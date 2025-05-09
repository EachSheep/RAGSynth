#!/bin/bash

python src/wash_by_llm.py --source_folder docs/admission.stanford.edu.new --target_folder docs/admission.stanford.edu.wash_by_llm --num_processes 1
python src/wash_by_llm.py --source_folder docs/berkeley.edu.admission.new --target_folder docs/berkeley.edu.admission.wash_by_llm --num_processes 1

python src/wash_by_llm.py --source_folder docs/docs.cyotek.com.new --target_folder docs/docs.cyotek.com.wash_by_llm --num_processes 8
python src/wash_by_llm.py --source_folder docs/www.notion.so.help.new --target_folder docs/www.notion.so.help.wash_by_llm --num_processes 1

python src/wash_by_llm.py --source_folder docs/hearthstone.fandom.com.wiki.new --target_folder docs/hearthstone.fandom.com.wiki.wash_by_llm --num_processes 8
python src/wash_by_llm.py --source_folder docs/zelda.fandom.com.wiki.new --target_folder docs/zelda.fandom.com.wiki.wash_by_llm --num_processes 8

python src/wash_by_llm.py --source_folder docs/www.drugs.com.new --target_folder docs/www.drugs.com.wash_by_llm --num_processes 32
python src/wash_by_llm.py --source_folder docs/www.mayoclinic.org.new --target_folder docs/www.mayoclinic.org.wash_by_llm --num_processes 8
