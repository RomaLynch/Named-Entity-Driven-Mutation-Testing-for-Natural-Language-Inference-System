# We conduct prediction step, we predict mutation sentences which pass filter step.
import numpy as np
import random
import json
from nltk.corpus import names
from collections import defaultdict 
import random
import pandas as pd
from pprint import pprint
import torch
import os
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import re
import torch.nn.functional as F
import torch

model_path = '\roberta-large-mnli'

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

label_map = {"0": "entailment", "1": "neutral", "2": "contradiction"}

def predictWithGPU(premise, hypothesis, tokenizer, model, label_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    inputs = tokenizer(
        premise, 
        hypothesis, 
        return_tensors="pt", 
        truncation=True, 
        padding=True
    ).to(device) 
    

    with torch.no_grad():
        outputs = model(** inputs)
        logits = outputs.logits  
    
    logits_cpu = logits.cpu()  
    predicted_class = logits_cpu.argmax().item()
    confidence = torch.nn.functional.softmax(logits_cpu, dim=-1).tolist()[0]

    return label_map[str(predicted_class)], confidence 

with open("MNLI_Mut_Filtered.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
        
bug_data = {}
neutral_bug_data = {} 
valid_tags = ['PERSON', 'GPE', 'ORG']
formatted_tags = {
    'PERSON': 'PERSON',
    'GPE': 'GPE',
    'ORG': 'ORG'
}

neutral_formatted_tags = {
    'PERSON': 'PERSON2',
    'GPE': 'GPE2',
    'ORG': 'ORG2'
}
        
for tag in valid_tags:
    if tag not in data:
        print(f"Warning: can't find entity tag: {tag}, skip")
        continue
                
    bug_items = []
    neutral_bug_items = []
    total = len(data[tag])
            
    for i, item in enumerate(data[tag]):
        try:
            original = item['original']
            mutated = item['mutated']
                    
            if len(original) != 2 or len(mutated) != 2:
                print(f"Skip error item:(tag: {tag}, index:{i})")
                continue
            
            original_premise = original[0]
            original_hypothesis = original[1]
            _, original_confidence = predictWithGPU(original_premise, original_hypothesis, tokenizer, model, label_map)
            
            mutated_premise = mutated[0]
            mutated_hypothesis = mutated[1]
            pred_label, mutated_confidence = predictWithGPU(mutated_premise, mutated_hypothesis, tokenizer, model, label_map)         
                    
            new_item = {
                "Original": original,
                "Mutated": mutated,
                "Confidence": original_confidence,  
                "Predicted_label": pred_label,      
                "Predicted_label_confidence": mutated_confidence  
            }

            # our bug detection: if no neutral-->bug
            if pred_label != "neutral":
                bug_items.append(new_item)
            else:
                neutral_bug_items.append(new_item)
                        
        except Exception as e:
            print(f"Error: (tag: {tag}, index: {i}): {e}")
            continue
            
    bug_data[formatted_tags[tag]] = bug_items
    neutral_bug_data[neutral_formatted_tags[tag]] = neutral_bug_items
    error_rate = len(bug_items)/total if total > 0 else 0.0
    print(f"Tag {tag}: original sentences pairs:{total}, Bug:{len(bug_items)}, no-bug: {len(neutral_bug_items)}, bug rate: {error_rate:.2%})")

bug_data.update(neutral_bug_data)

output_path = "MNLI_filtered_bugs.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(bug_data, f, ensure_ascii=False, indent=2)