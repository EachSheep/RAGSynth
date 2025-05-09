#!/bin/zsh

trytimes=3
for ((i=1;i<=trytimes;i++));
do
    # ./query_generation_train.sh -d docs.cyotek.com.filter -t 3 -p content -l llm_gta_10.env -e embed.env
    # ./query_generation_train.sh -d docs.cyotek.com.filter -t 3 -p entity_graph -l llm_gta_10.env -e embed.env
    ./query_generation_train.sh -d hearthstone.fandom.com.wiki.filter -t 3 -p content -l llm_gta_10.env -e embed.env
    ./query_generation_train.sh -d hearthstone.fandom.com.wiki.filter -t 3 -p entity_graph -l llm_gta_10.env -e embed.env

    ./query_generation_train.sh -d docs.cyotek.com.filter -t 3_minus_1 -p content -l llm_gta_10.env -e embed.env
    ./query_generation_train.sh -d docs.cyotek.com.filter -t 3_minus_1 -p entity_graph -l llm_gta_10.env -e embed.env
    ./query_generation_train.sh -d hearthstone.fandom.com.wiki.filter -t 3_minus_1 -p content -l llm_gta_10.env -e embed.env
    ./query_generation_train.sh -d hearthstone.fandom.com.wiki.filter -t 3_minus_1 -p entity_graph -l llm_gta_10.env -e embed.env
done