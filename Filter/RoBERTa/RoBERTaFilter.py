import os
import json
import tkinter as tk
from stanfordcorenlp import StanfordCoreNLP

class StanfordDependencyFilter:
    def __init__(self, stanford_parser_path):
        self.nlp = StanfordCoreNLP(stanford_parser_path, lang='en')
        self.core_dep_labels = {
            'ROOT', 'nsubj', 'nsubjpass', 'obj', 'dobj', 'iobj', 
            'cop', 'attr', 'ccomp', 'agent'
        }
        self._check_tkinter()

    def _check_tkinter(self):
        try:
            root = tk.Tk()
            root.withdraw()
        except Exception as e:
            raise RuntimeError(
                "Tkinter Error."
            ) from e

    def extract_core_dependencies(self, sentence):
        if not sentence.strip():
            return []
        
        try:
            dependencies = self.nlp.dependency_parse(sentence)
            all_dep_labels = [dep[0] for dep in dependencies]
            dep_skeleton = []
            
            for dep in dependencies:
                dep_label, head_idx, dep_idx = dep
                if dep_label in self.core_dep_labels:
                    head_dep_label = all_dep_labels[head_idx - 1] if (head_idx - 1) < len(all_dep_labels) else 'OTHER'
                    head_dep_label = head_dep_label if head_dep_label in self.core_dep_labels else 'OTHER'
                    dep_skeleton.append((dep_label, head_dep_label))
            
            return sorted(dep_skeleton)
        
        except Exception as e:
            print(f"Parse Error: {e}£¬Sentence: {sentence[:30]}...")
            return []
        
    def check_dependency_consistency(self, original_sentence, mutated_sentence):
        original_skel = self.extract_core_dependencies(original_sentence)
        mutated_skel = self.extract_core_dependencies(mutated_sentence)
        
        if original_skel == mutated_skel:
            reason = "Dependency relation is same."
            return True, reason, original_skel, mutated_skel
        else:
            added = set(mutated_skel) - set(original_skel)
            removed = set(original_skel) - set(mutated_skel)
            change_detail = []
            if added:
                change_detail.append(f"Mut sentence have a new dependency relation: {list(added)}")
            if removed:
                change_detail.append(f"Mut sentence don't have original relation: {list(removed)}")
            reason = "; ".join(change_detail) if change_detail else "Dependency relation changed"
            return False, reason, original_skel, mutated_skel

    def process_json_file(self, input_path, output_pass_path, output_fail_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pass_data = {}
        fail_data = {}
        valid_tags = ['PERSON', 'GPE', 'ORG', 'NORP']
        
        for tag in valid_tags:
            if tag not in data:
                print(f"Warning: can't find tag: {tag}, skip")
                continue
                
            pass_items = []
            fail_items = []
            total = len(data[tag])
            
            for i, item in enumerate(data[tag]):
                try:
                    original = item['original']
                    mutated = item['mutated']
                    
                    if len(original) != 2 or len(mutated) != 2:
                        fail_items.append({**item, 'filter_reason': "Data format error", 'failed_part': 'sentences pairs'})
                        continue
                        
                    premise_ok, p_reason, p_orig, p_mut = self.check_dependency_consistency(original[0], mutated[0])
                    hypo_ok, h_reason, h_orig, h_mut = self.check_dependency_consistency(original[1], mutated[1])
                    
                    if premise_ok and hypo_ok:
                        pass_items.append({
                            **item,
                            'dependency_info': {'premise': (p_orig, p_mut), 'hypothesis': (h_orig, h_mut)}
                        })
                    else:
                        fail_reason = []
                        if not premise_ok:
                            fail_reason.append(f"premise: {p_reason}")
                        if not hypo_ok:
                            fail_reason.append(f"hypothesis: {h_reason}")
                        fail_items.append({
                            **item,
                            'filter_reason': "; ".join(fail_reason),
                            'failed_part': 'premise' if not premise_ok else 'hypothesis',
                            'dependency_info': {'premise': (p_orig, p_mut), 'hypothesis': (h_orig, h_mut)}
                        })
                        
                except Exception as e:
                    fail_items.append({** item, 'filter_reason': f"error: {str(e)}", 'failed_part': 'program'})
                    continue
            
            pass_data[tag] = pass_items
            fail_data[tag] = fail_items
            print(f"\nTag {tag}: Original senetnces{total} ¡ú keep{len(pass_items)}£¨{len(pass_items)/total:.2%}£©")
        
        os.makedirs(os.path.dirname(output_pass_path), exist_ok=True)
        with open(output_pass_path, 'w', encoding='utf-8') as f:
            json.dump(pass_data, f, ensure_ascii=False, indent=2)
        
        os.makedirs(os.path.dirname(output_fail_path), exist_ok=True)
        with open(output_fail_path, 'w', encoding='utf-8') as f:
            json.dump(fail_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nresult: consistency¡ú{output_pass_path}£¬inconsistency¡ú{output_fail_path}")

    def close(self):
        self.nlp.close()

if __name__ == "__main__":
    # Load stanford parse model, you can download model to local.
    STANFORD_PATH = r'\stanfordcorenlp\stanford-corenlp-4.4.0'
    INPUT_JSON = "SNLI_Mut.json"
    OUTPUT_PASS = "SNLI_Mut_Filtered.json" 
    OUTPUT_FAIL = "SNLI_fail.json"  

    filter = StanfordDependencyFilter(STANFORD_PATH)
    filter.process_json_file(INPUT_JSON, OUTPUT_PASS, OUTPUT_FAIL)