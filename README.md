# Named-Entity-Driven-Mutation-Testing-for-NLI-system
#### This repository stores the experimental code we wrote for the paper "Named-Entity-Driven-Mutation-Testing-for-Natural-Language-Inference-System".
# Datasets
#### We use three NLI datasets: MNLI, SNLI, and MultiNLI. We mainly use test sets of MNLI and SNLI, and we extract 30,000 samples from MultiNLI. We put these three datasets in the dataset folder.
# Models
#### We use two NLI models for testing, RoBERTa and ELMo. We downloaded these two models locally. It is also easy to call models through HuggingFace (https://github.com/huggingface/).
# Approach
#### **Neutral Sentences Extraction**.
#### **Mutation generation**. We leverage NER for our mutation process, primarily by identifying and mutating the three most common named entity categories. We have designed three mutation operators for different scenarios, namely PERSON, LOCATION, and ORGANIZATION, which are the main named entity categories. We use spaCy's NER for entity recognition, with the model version being "en_core_web_md". This is a medium-sized model that is relatively lightweight and does not occupy excessive memory.
#### **Sentence Structure Filter**. 
#### **Errors detection**.
