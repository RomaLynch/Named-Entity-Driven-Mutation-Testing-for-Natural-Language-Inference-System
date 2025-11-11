# We need to extract the "Neutral" tag sentences for our experiment
# This python file is used for the above task
import pandas as pd
import numpy as np
import json
from pprint import pprint
import random
import torch
import os
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import re
import torch.nn.functional as F
import torch

model_path = '\roberta-large-mnli'

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)

label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}

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
	
    return label_map[predicted_class]

# We need to extract "Neutral" sentences from 30,0000 original multiNLI. We also need to extract sentences from MNLI and SNLI
with open("NeutralMultiNLI.txt", "r", encoding='utf-8') as f:
    all_lines = [line.rstrip("\n") for line in f.readlines()][:90000]

neutral_pairs = []
current_pair = []

for line in all_lines:
    stripped_line = line.strip()
    if stripped_line:
        current_pair.append(stripped_line)
        if len(current_pair) == 2:
            premise = current_pair[0]
            hypothesis = current_pair[1]
            pre_label = predictWithGPU(premise, hypothesis, tokenizer, model, label_map)
            if pre_label == 'neutral':
                neutral_pairs.append(f"{premise}\n{hypothesis}\n\n")
            current_pair = []
    else:
        current_pair = []
        
if neutral_pairs:
	# "robertaNeutralMultiNLI.txt" in Neutral MultiNLI. You need to do the same process on MNLI and SNLI
	# "robertaNeutralMNLI.txt" and ""robertaNeutralSNLI.txt""
    with open("robertaNeutralMultiNLI.txt", "w", encoding='utf-8') as f:
        f.write("".join(neutral_pairs))