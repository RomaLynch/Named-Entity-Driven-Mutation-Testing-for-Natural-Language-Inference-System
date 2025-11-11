from spacy import displacy
import numpy as np
from nltk.corpus import wordnet
import random
import nltk
import json
from nltk.corpus import names
from spacy.pipeline import EntityRuler
from collections import defaultdict 
import random
import pandas as pd
from pprint import pprint
import torch
import os
import re
import torch.nn.functional as F
import torch
from matplotlib_venn import venn3
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from spacy.tokens import Token
import spacy
from collections import defaultdict
import pycountry

# wordnet person names rep, for our PERSON-NAME mutation
male_names = names.words('male.txt')
female_names = names.words('female.txt')
# load SpaCy nlp model for NER, you can download model to local or use online model
model_path = r'/en_core_web_md'
nlp = spacy.load(model_path)

def get_random_gpe():
    # Use pycountry generate country names randomly for COprintRY mutation(GPE entities)
    choice = random.choice(["country", "city"])
    if choice == "country":
        countries = list(pycountry.countries)
        return random.choice(countries).name
    else:
        with open("Cities.txt", "r", encoding='utf-8') as f:
            cities = [line.strip() for line in f.readlines()]
        return random.choice(cities)

def get_random_org():
    # We manually collect ORG entities for our ORGANIZATION mutation
    with open("ORG.txt", "r", encoding='utf-8') as f:
        ORG_names = [line.strip() for line in f.readlines()]

    return random.choice(ORG_names)

def get_random_person():
    # For PERSON-NAME mutation
    return random.choice(male_names) if random.choice([True, False]) else random.choice(female_names)


def identify_target_tokens(text):
    # function for recognizing entities(PERSON, ORG, GPE, NORP)
    doc = nlp(text)
    target_tokens = []
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "GPE", "ORG"]:
            target_tokens.append({
                "start": ent.start,
                "end": ent.end,
                "type": ent.label_,
                "text": ent.text
            })
    return target_tokens



def mut_entity(text, entity_type, is_premise=True):
    # function for replace our four target entities
    doc = nlp(text)
    mut_text = []
    target_tokens = identify_target_tokens(text)
    target_idx = 0
    i = 0

    entity_map = {
        "PERSON": get_random_person,
        "GPE": get_random_gpe,
        "ORG": get_random_org,
    }
    tag = entity_type

    while i < len(doc):
        # recognize the index of entity and find the target entity
        if target_idx < len(target_tokens) and i == target_tokens[target_idx]["start"]:
            target = target_tokens[target_idx]
            if target["type"] == entity_type:
                replacement = entity_map[entity_type]()
                # no repetition of entity after mutation
                while replacement.lower() == target["text"].lower():
                    replacement = entity_map[entity_type]()
            else:
                # keep original sentence
                replacement = target["text"]
            mut_text.append(replacement)
            i = target["end"]
            target_idx += 1
        else:
            mut_text.append(doc[i].text)
            i += 1

    return " ".join(mut_text), tag


# mutation operator
mutPremisePerson = lambda x: mut_entity(x, "PERSON", is_premise=True)
mutHypothesisPerson = lambda x: mut_entity(x, "PERSON", is_premise=False)
mutPremiseGPE = lambda x: mut_entity(x, "GPE", is_premise=True)
mutHypothesisGPE = lambda x: mut_entity(x, "GPE", is_premise=False)
mutPremiseORG = lambda x: mut_entity(x, "ORG", is_premise=True)
mutHypothesisORG = lambda x: mut_entity(x, "ORG", is_premise=False)

# mutation
def process_sentence_pair(premise, hypothesis):
    mutated_pairs = []
    original_pair = (premise, hypothesis)
    
    def has_target_type(text, target_type):
        tokens = identify_target_tokens(text)
        return any(token["type"] == target_type for token in tokens)
    
    premise_has_person = has_target_type(premise, "PERSON")
    premise_has_gpe = has_target_type(premise, "GPE")
    premise_has_org = has_target_type(premise, "ORG")
    
    hypo_has_person = has_target_type(hypothesis, "PERSON")
    hypo_has_gpe = has_target_type(hypothesis, "GPE")
    hypo_has_org = has_target_type(hypothesis, "ORG")
    
    # PERSON-NAME mutation
    if premise_has_person:
        mutated_p, tag = mutPremisePerson(premise)
        mutated_pairs.append(((mutated_p, hypothesis), original_pair, tag))
    if hypo_has_person:
        mutated_h, tag = mutHypothesisPerson(hypothesis)
        mutated_pairs.append(((premise, mutated_h), original_pair, tag))
    if premise_has_person and hypo_has_person:
        p_mut, p_tag = mutPremisePerson(premise)
        h_mut, h_tag = mutHypothesisPerson(hypothesis)
        mutated_pairs.append(((p_mut, h_mut), original_pair, tag))
    
    # COUNTRY mutation
    if premise_has_gpe:
        mutated_p, tag = mutPremiseGPE(premise)
        mutated_pairs.append(((mutated_p, hypothesis), original_pair, tag))
    if hypo_has_gpe:
        mutated_h, tag = mutHypothesisGPE(hypothesis)
        mutated_pairs.append(((premise, mutated_h), original_pair, tag))
    if premise_has_gpe and hypo_has_gpe:
        p_mut, p_tag = mutPremiseGPE(premise)
        h_mut, h_tag = mutHypothesisGPE(hypothesis)
        mutated_pairs.append(((p_mut, h_mut), original_pair, tag))
    
    # ORGANIZATION mutation
    if premise_has_org:
        mutated_p, tag = mutPremiseORG(premise)
        mutated_pairs.append(((mutated_p, hypothesis), original_pair, tag))
    if hypo_has_org:
        mutated_h, tag = mutHypothesisORG(hypothesis)
        mutated_pairs.append(((premise, mutated_h), original_pair, tag))
    if premise_has_org and hypo_has_org:
        p_mut, p_tag = mutPremiseORG(premise)
        h_mut, h_tag = mutHypothesisORG(hypothesis)
        mutated_pairs.append(((p_mut, h_mut), original_pair, tag))
    
    # discard the same mutation sentences
    unique_pairs = []
    unique_mut_sentences = set()
    for mutated, original, m_type in mutated_pairs:
        pair_key = (mutated[0], mutated[1])
        if pair_key not in unique_mut_sentences:
            unique_mut_sentences.add(pair_key)
            unique_pairs.append((mutated, original, m_type))
    
    return unique_pairs

def main(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) % 2 != 0:
        print(f"Data format is not correct.")
        lines = lines[:-1]
    # extract every 2 sentences (premise, hypothesis)
    sentence_pairs = [(lines[i], lines[i+1]) for i in range(0, len(lines), 2)]
    print(f"Successfully read {len(sentence_pairs)} sentence pairs.")

    mutated_groups = defaultdict(list)
    total_mutated = 0
    for premise, hypothesis in sentence_pairs:
        pairs_with_type = process_sentence_pair(premise, hypothesis)
        for pair, original_pair, mut_type in pairs_with_type:
            base_type = mut_type
            mutated_groups[base_type].append({
                "original": list(original_pair),
                "mutated": list(pair)
            })
        total_mutated += len(pairs_with_type)

    final_result = dict(mutated_groups)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)

    print(f"Total mutated pairs generated: {total_mutated}")
    print("The quantities of different mutation operators:")
    for mut_type, items in final_result.items():
        print(f"{mut_type}: {len(items)} pairs")
    print(f"Result saved to: {output_file}")

	main("elmoNeutralMNLI.txt", "MNLI_Mut.json")
	main("elmoNeutralSNLI.txt", "SNLI_Mut.json")
	main("elmoNeutralMultiNLI.txt", "MultiNLI_Mut.json")

