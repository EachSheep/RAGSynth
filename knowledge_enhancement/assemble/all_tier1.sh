#!/bin/zsh

trytimes=1
for ((i=1;i<=trytimes;i++));
do
    ./query_generation_test.sh -d docs.cyotek.com.filter -t 1 -p content -l llm.env -e embed.env
    ./query_generation_test.sh -d docs.cyotek.com.filter -t 1 -p entity_graph -l llm.env -e embed.env


    ./query_generation_test.sh -d hearthstone.fandom.com.wiki.filter -t 1 -p content -l llm.env -e embed.env
    ./query_generation_test.sh -d hearthstone.fandom.com.wiki.filter -t 1 -p entity_graph -l llm.env -e embed.env


    ./query_generation_test.sh -d www.drugs.com.filter -t 1 -p content -l llm.env -e embed.env
    ./query_generation_test.sh -d www.drugs.com.filter -t 1 -p entity_graph -l llm.env -e embed.env


    ./query_generation_test.sh -d www.mayoclinic.org.filter -t 1 -p content -l llm.env -e embed.env
    ./query_generation_test.sh -d www.mayoclinic.org.filter -t 1 -p entity_graph -l llm.env -e embed.env


    ./query_generation_test.sh -d www.notion.so.help.filter -t 1 -p content -l llm.env -e embed.env
    ./query_generation_test.sh -d www.notion.so.help.filter -t 1 -p entity_graph -l llm.env -e embed.env


    ./query_generation_test.sh -d zelda.fandom.com.wiki.filter -t 1 -p content -l llm.env -e embed.env
    ./query_generation_test.sh -d zelda.fandom.com.wiki.filter -t 1 -p entity_graph -l llm.env -e embed.env


    ./query_generation_test.sh -d admission.stanford.edu.filter -t 1 -p content -l llm.env -e embed.env
    ./query_generation_test.sh -d admission.stanford.edu.filter -t 1 -p entity_graph -l llm.env -e embed.env


    ./query_generation_test.sh -d berkeley.edu.admission.filter -t 1 -p content -l llm.env -e embed.env
    ./query_generation_test.sh -d berkeley.edu.admission.filter -t 1 -p entity_graph -l llm.env -e embed.env
done



trytimes=3
for ((i=1;i<=trytimes;i++));
do
    ./query_generation_train.sh -d docs.cyotek.com.filter -t 3 -p content -l llm.env -e embed.env
    ./query_generation_train.sh -d docs.cyotek.com.filter -t 3 -p entity_graph -l llm.env -e embed.env


    ./query_generation_train.sh -d hearthstone.fandom.com.wiki.filter -t 3 -p content -l llm.env -e embed.env
    ./query_generation_train.sh -d hearthstone.fandom.com.wiki.filter -t 3 -p entity_graph -l llm.env -e embed.env


    ./query_generation_train.sh -d www.drugs.com.filter -t 3 -p content -l llm.env -e embed.env
    ./query_generation_train.sh -d www.drugs.com.filter -t 3 -p entity_graph -l llm.env -e embed.env


    ./query_generation_train.sh -d www.mayoclinic.org.filter -t 3 -p content -l llm.env -e embed.env
    ./query_generation_train.sh -d www.mayoclinic.org.filter -t 3 -p entity_graph -l llm.env -e embed.env


    ./query_generation_train.sh -d www.notion.so.help.filter -t 3 -p content -l llm.env -e embed.env
    ./query_generation_train.sh -d www.notion.so.help.filter -t 3 -p entity_graph -l llm.env -e embed.env


    ./query_generation_train.sh -d zelda.fandom.com.wiki.filter -t 3 -p content -l llm.env -e embed.env
    ./query_generation_train.sh -d zelda.fandom.com.wiki.filter -t 3 -p entity_graph -l llm.env -e embed.env


    ./query_generation_train.sh -d admission.stanford.edu.filter -t 3 -p content -l llm.env -e embed.env
    ./query_generation_train.sh -d admission.stanford.edu.filter -t 3 -p entity_graph -l llm.env -e embed.env


    ./query_generation_train.sh -d berkeley.edu.admission.filter -t 3 -p content -l llm.env -e embed.env
    ./query_generation_train.sh -d berkeley.edu.admission.filter -t 3 -p entity_graph -l llm.env -e embed.env
done