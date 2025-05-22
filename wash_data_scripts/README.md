# Partial Workflow of Data Cleaning:

1. For domain-specific data obtained from websites, we utilize large language models for data cleaning:
   1. `tranverse.sh` => Converts HTML to Markdown;
   2. `wash_by_llm.sh` => Cleans low-quality data using a large language model;
   3. `summary_and_extract_by_llm.sh` => Extracts FAQs.
2. `cal_md_file_num.sh`: Counts the number of files in a directory.
3. `hierarchical_summary.sh`: Provides an overview of the data.
4. `statistics_filter.sh`: Performs statistical analysis on the document collection.