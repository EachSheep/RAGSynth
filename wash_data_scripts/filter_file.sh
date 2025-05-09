#!/bin/bash

python src/filter_file.py --source_folder docs/admission.stanford.edu.wash_by_llm \
    --summary_folder docs/admission.stanford.edu.summary_and_extract_by_llm \
    --target_folder docs/admission.stanford.edu.filter
python src/filter_file.py --source_folder docs/berkeley.edu.admission.wash_by_llm \
    --summary_folder docs/berkeley.edu.admission.summary_and_extract_by_llm \
    --target_folder docs/berkeley.edu.admission.filter

python src/filter_file.py --source_folder docs/docs.cyotek.com.wash_by_llm \
    --summary_folder docs/docs.cyotek.com.summary_and_extract_by_llm \
    --target_folder docs/docs.cyotek.com.filter
python src/filter_file.py --source_folder docs/www.notion.so.help.wash_by_llm \
    --summary_folder docs/www.notion.so.help.summary_and_extract_by_llm \
    --target_folder docs/www.notion.so.help.filter

python src/filter_file.py --source_folder docs/hearthstone.fandom.com.wiki.wash_by_llm \
    --summary_folder docs/hearthstone.fandom.com.wiki.summary_and_extract_by_llm \
    --target_folder docs/hearthstone.fandom.com.wiki.filter
# Deleted all files and folders starting with "Community" (these files were user profiles).
python src/filter_file.py --source_folder docs/zelda.fandom.com.wiki.wash_by_llm \
    --summary_folder docs/zelda.fandom.com.wiki.summary_and_extract_by_llm \
    --target_folder docs/zelda.fandom.com.wiki.filter

python src/filter_file.py --source_folder docs/www.drugs.com.wash_by_llm \
    --summary_folder docs/www.drugs.com.summary_and_extract_by_llm \
    --target_folder docs/www.drugs.com.filter
python src/filter_file.py --source_folder docs/www.mayoclinic.org.wash_by_llm \
    --summary_folder docs/www.mayoclinic.org.summary_and_extract_by_llm \
    --target_folder docs/www.mayoclinic.org.filter