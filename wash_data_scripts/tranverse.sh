#!/bin/bash

python src/tranverse.py --source_folder docs/admission.stanford.edu \
    --target_folder docs/admission.stanford.edu.cn.new \
    --base_url https://admission.standford.edu # ok, https://admission.stanford.edu
python src/tranverse.py --source_folder docs/berkeley.edu.admission \
    --target_folder docs/berkeley.edu.admission.new \
    --base_url https://berkeley.edu/admission/ # ok, https://berkeley.edu/admission

python src/tranverse.py --source_folder docs/docs.cyotek.com \
    --target_folder docs/docs.cyotek.com.new \
    --base_url https://docs.cyotek.com # ok, https://docs.cyotek.com
python src/tranverse.py --source_folder docs/www.notion.so.help \
    --target_folder docs/www.notion.so.help.new \
    --base_url https://www.notion.so.help # ok, https://www.notion.so.help/help

python src/tranverse.py --source_folder docs/hearthstone.fandom.com.wiki \
    --target_folder docs/hearthstone.fandom.com.wiki.new \
    --base_url https://hearthstone.fandom.com.wiki # ok, https://hearthstone.fandom.com.wiki
python src/tranverse.py --source_folder docs/zelda.fandom.com.wiki \
    --target_folder docs/zelda.fandom.com.wiki.new \
    --base_url https://zelda.fandom.com.wiki # ok, https://zelda.fandom.com.wiki

python src/tranverse.py --source_folder docs/www.drugs.com \
    --target_folder docs/www.drugs.com.new \
    --base_url https://www.drugs.com
python src/tranverse.py --source_folder docs/www.mayoclinic.org \
    --target_folder docs/www.mayoclinic.org.new \
    --base_url https://www.mayoclinic.org
# **Characteristics:** Mayo Clinic is a renowned non-profit medical institution that offers medical information written and reviewed by doctors and healthcare experts. Drugs.com is a commercial website focused on drug-related information.
# **Focus:** Mayo Clinic provides comprehensive medical information, including symptom management and treatment advice, covering various aspects such as diseases, surgeries, and lifestyle. In contrast, Drugs.com specializes in drug side effects, interactions, and dosages, helping users understand how to use medications.
# **Sources of Information:** Mayo Clinic's information is typically provided by its in-house medical experts, based on their clinical experience and research. Drugs.com's drug information is sourced from various entities, including the FDA, pharmaceutical companies, and user feedback.

# python src/tranverse.py --source_folder docs/www.rxlist.com \
#     --target_folder docs/www.rxlist.com.new \
#     --base_url https://www.rxlist.com
# python src/tranverse.py --source_folder docs/www.webmd.com \
#     --target_folder docs/www.webmd.com.new \
#     --base_url https://www.webmd.com