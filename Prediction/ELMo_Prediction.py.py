from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import json
from allennlp.data import Instance
import random
import pandas as pd
import numpy as np
from pprint import pprint
import torch
from allennlp.data import Batch
import os

predictor = Predictor.from_path("/elmo/decomposable-attention-elmo-2020.04.09.tar.gz")
indexMap = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}

def predictWithElmoGPU(premise, hypothesis, predictor, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if hasattr(predictor, '_model'):
        predictor._model = predictor._model.to(device)
        predictor._model.eval()
    
    myDict = {"premise": premise, "hypothesis": hypothesis}
    instance = predictor._dataset_reader.text_to_instance(** myDict)

    predictor._dataset_reader.apply_token_indexers(instance)
    
    batch = Batch([instance])
    batch.index_instances(predictor._model.vocab)
    batch_tensor = batch.as_tensor_dict()
    
    def move_to_device(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: move_to_device(v, device) for k, v in data.items()}
        return data
    
    batch_tensor = move_to_device(batch_tensor, device)
    
    with torch.no_grad():
        outputs = predictor._model(**batch_tensor)
    
    label_probs = outputs['label_probs'].cpu().numpy()[0]
    maxIndex = label_probs.argmax()
    
    return indexMap[maxIndex], label_probs.tolist()

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
            _, original_confidence = predictWithElmoGPU(original_premise, original_hypothesis, predictor)
            
            mutated_premise = mutated[0]
            mutated_hypothesis = mutated[1]
            pred_label, mutated_confidence = predictWithElmoGPU(mutated_premise, mutated_hypothesis, predictor)         
                    
            new_item = {
                "Original": original,
                "Mutated": mutated,
                "Confidence": original_confidence,  
                "Predicted_label": pred_label,  
                "Predicted_label_confidence": mutated_confidence 
            }

            if pred_label != "neutral":
                bug_items.append(new_item)
            else:
                neutral_bug_items.append(new_item)
                        
        except Exception as e:
            print(f"Error: (tag: {tag}, index: {i}): {e}")
            continue
            
    bug_data[formatted_tags[tag]] = bug_items
    neutral_bug_data[neutral_formatted_tags[tag]] = neutral_bug_items
    print(f"Tag {tag}: original sentences pairs:{total}, Bug:{len(bug_items)}, no-bug: {len(neutral_bug_items)}, bug rate: {len(bug_items)/total:.2%})")

bug_data.update(neutral_bug_data)

output_path = "MNLI_filtered_bugs.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(bug_data, f, ensure_ascii=False, indent=2)
    print(f"Results save in: {os.path.abspath(output_path)}")