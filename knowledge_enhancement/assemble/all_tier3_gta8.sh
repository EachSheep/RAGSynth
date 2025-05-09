#!/bin/zsh

trytimes=1
for ((i=1;i<=trytimes;i++));
do

    # ./query_generation_train.sh -d www.drugs.com.filter -t 3 -p content -l llm_gta_7.env -e embed.env
    # ./query_generation_train.sh -d www.drugs.com.filter -t 3 -p entity_graph -l llm_gta_7.env -e embed.env
    # ./query_generation_train.sh -d www.mayoclinic.org.filter -t 3 -p content -l llm_gta_7.env -e embed.env
    # ./query_generation_train.sh -d www.mayoclinic.org.filter -t 3 -p entity_graph -l llm_gta_7.env -e embed.env

    ./query_generation_train.sh -d www.drugs.com.filter -t 3_minus_1 -p content -l llm_gta_7.env -e embed.env
    ./query_generation_train.sh -d www.drugs.com.filter -t 3_minus_1 -p entity_graph -l llm_gta_7.env -e embed.env
    # ./query_generation_train.sh -d www.mayoclinic.org.filter -t 3_minus_1 -p content -l llm_gta_7.env -e embed.env
    # ./query_generation_train.sh -d www.mayoclinic.org.filter -t 3_minus_1 -p entity_graph -l llm_gta_7.env -e embed.env
done
