import os
import copy
import json 

import torch

# For data read write
def read_json(path):
    
    with open(path, "r", encoding = "UTF-8") as f:
        data = json.load(f)

    return data

def write_json(data, path):

    with open(path, "w", encoding = "UTF-8") as f:
        json.dump(data, f, indent = 4)

# For prompting
def get_logits(scores):

    all_scores = []
    
    for i in range(len(scores)):
        
        current_scores = torch.softmax(scores[i], dim = -1)
        max_score = max(current_scores[0]).detach().cpu().numpy().tolist()
        all_scores.append(max_score)

    return all_scores

def get_model_response(inputs, llm, tokenizer):

    inputs = tokenizer(inputs, return_tensors = "pt")
    outputs = llm.generate(**inputs, return_dict_in_generate = True, output_scores = True)
    predictions = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens = True)
    print(predictions)
    a = z 
    
    logits = get_logits(outputs["scores"])
    
    return predictions, logits

