# We need to extract the "Neutral" tag sentences for our experiment
# This python file is used for the above task
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import json
from allennlp.data import Instance
import spacy
from spacy import displacy
import en_core_web_sm
import random
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import names
import pandas as pd
import numpy as np
from pprint import pprint
import torch

predictor = Predictor.from_path("/elmo/decomposable-attention-elmo-2020.04.09.tar.gz")
indexMap = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}


from allennlp.data import Batch

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
    
    return indexMap[maxIndex]

# We need to extract "Neutral" sentences from 30,0000 original multiNLI. We also need to extract sentences from MNLI and SNLI
with open("NeutralMultiNLI.txt", "r", encoding='utf-8') as f:
    all_lines = [line.rstrip("\n") for line in f.readlines()][:90000]

neural_pairs = []
current_pair = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for line in all_lines:
    stripped_line = line.strip()
    if stripped_line:
        current_pair.append(stripped_line)
        if len(current_pair) == 2:
            premise = current_pair[0]
            hypothesis = current_pair[1]
            pred_label = predictWithElmoGPU(premise, hypothesis, predictor)
            if pred_label == "neutral":
                neural_pairs.append(f"{premise}\n{hypothesis}\n\n")
            current_pair = []
    else:
        current_pair = []
if neural_pairs:
	# "elmoNeutralMultiNLI.txt" in Neutral MultiNLI. You need to do the same process on MNLI and SNLI
	# "elmoNeutralMNLI.txt" and "elmoNeutralSNLI.txt"
    with open("elmoNeutralMultiNLI.txt", "w", encoding='utf-8') as f:
        f.write("".join(neural_pairs))