# Named-Entity-Driven-Mutation-Testing-for-NLI-system
This repository stores the experimental code we wrote for the paper "Named-Entity-Driven-Mutation-Testing-for-Natural-Language-Inference-System".
# Datasets
We use three NLI datasets: MNLI, SNLI, and MultiNLI. We mainly use test sets of MNLI and SNLI, and we extract 30,000 samples from MultiNLI. We put these three datasets in the dataset folder.
# Models
We use two NLI models for testing, RoBERTa and ELMo. We downloaded these two models locally. It is also easy to call models through HuggingFace (https://github.com/huggingface/).
# Approach
**Neutral Sentences Extraction**. We primarily conduct mutations on sentences labeled as neutral in NLI tasks. Therefore, we need to perform a "First-Round Inference" step to obtain sentence pairs labeled as neutral by the model across the three datasets, which will serve as our seed sentences.

**Mutation generation**. We leverage NER for our mutation process, primarily by identifying and mutating the three most common named entity categories. We have designed three mutation operators for different scenarios, namely PERSON, LOCATION, and ORGANIZATION, which are the main named entity categories. We use spaCy's NER for entity recognition, with the model version being "en_core_web_md". This is a medium-sized model that is relatively lightweight and does not occupy excessive memory. We use the person name repository  of WordNet for PERSON entity replacement, Pycountry and manually collected city names from GaWC for LOCATION replacement, and organization names (such as company names and university names) collected from Wikipedia for ORGANIZATION replacement.

**Sentence Structure Filter**. Due to the complexity of natural language, mutated sentences may contain certain errors. Therefore, we use a sentence filter to remove non-compliant mutated sentences. Specifically, we employ the Stanford Parser for this filtering process: by comparing the dependency trees of mutated sentences with those of seed sentences, we select eligible mutated sentences. (Parser Download address: https://nlp.stanford.edu/software/lex-parser.shtml)

**Errors detection**. We argue that replacing person names, location names, or organization names in seed sentences could not alter the original neutral relationship. Therefore, if we find that the inference result of models for a mutated sentence is not neutral (i.e., entailment or contradiction), we consider this an inference bug.
