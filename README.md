# RAGSynthetic

1. **wash_data_scripts**: This section illustrates some of the data cleaning processes we employed during data collection.
2. **src**: This section differentiates between datasets, highlighting which datasets are used for evaluation and which serve as corpora.
3. **knowledge_enhancement**: This section demonstrates an implementation of RAGSynthetic.
   1. **assemble**: The process of synthesizing data using the content in the components.
   2. **components**: A specific implementation of RAGSynthetic in the form of a pipeline, where each step produces an output that serves as the input for the next step. This includes the synthesis of both single-hop and multi-hop data.
4. **chunk_by_files**: The logic for chunking documents.