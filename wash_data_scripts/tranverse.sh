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
# 性质：Mayo Clinic 是一家著名的非营利性医疗机构，其网站提供由医生和医疗专家撰写和审核的医疗信息。Drugs.com 是一个商业性网站，专注于药物信息。
# 重点：Mayo Clinic 主要提供广泛的医学信息、症状管理、治疗建议等，涵盖疾病、手术、生活方式等多个方面。Drugs.com 则专注于药物的副作用、相互作用、剂量等，帮助用户了解如何使用药物。
# 信息来源：Mayo Clinic 的信息通常由其内部的医学专家提供，并基于其临床经验和研究。Drugs.com 的药物信息来自多种来源，包括FDA、制药公司以及用户反馈。

# python src/tranverse.py --source_folder docs/www.rxlist.com \
#     --target_folder docs/www.rxlist.com.new \
#     --base_url https://www.rxlist.com
# python src/tranverse.py --source_folder docs/www.webmd.com \
#     --target_folder docs/www.webmd.com.new \
#     --base_url https://www.webmd.com