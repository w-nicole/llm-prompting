import os
import copy
import json 
import numpy as np

import torch

from config import MAX_OUTPUT_LENGTH, DEVICE

# For data read write
def read_json(path):
    
    with open(path, "r", encoding = "UTF-8") as f:
        data = json.load(f)

    return data

def write_json(data, path):

    with open(path, "w", encoding = "UTF-8") as f:
        json.dump(data, f, indent = 4)

# For prompting
def format_MCQ_options(options, add_or = False):
    format = ""
    for i, (k,v) in enumerate(options.items()):
        if i == len(options)-1:
            if add_or:
                format = format + f"or {k} {v}%"
            else:
                format = format + f"{k} {v}%"
        else:
            format = format + f"{k} {v}%\n"

    return format 

def get_avg_prob(scores, tokenizer):

    """
        Get average probability score for sentence 
        Note: implementation is not optimized for speed 
    """

    scores = torch.stack(scores).permute(1, 0, -1).contiguous()
    scores = torch.softmax(scores, dim = -1)
    bs, seq_length, _ = scores.shape
    all_probabilities = []

    for b in range(bs):
        
        p_list = []

        for s in range(seq_length):
            p, index = torch.max(scores[b, s, :], dim = 0)
            token = tokenizer.convert_ids_to_tokens([index])[0]
            if token == tokenizer.eos_token:
                break

            p = p.detach().cpu().item()
            p_list.append(p)

        avg_p = np.mean(p_list)
        all_probabilities.append(avg_p)
    
    return all_probabilities

def get_model_response(inputs, llm, tokenizer, return_prob = False):

    inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors = "pt")
    inputs.to(DEVICE) # Move to GPU / CPU
    outputs = llm.generate(**inputs, return_dict_in_generate = True, output_scores = True, max_new_tokens = MAX_OUTPUT_LENGTH, repetition_penalty = 1.1)
    predictions = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens = True)

    prob = None
    if return_prob: 
        prob = get_avg_prob(outputs["scores"], tokenizer)

    return predictions, prob

def clean_response(qns, ans):

    all_ans = [] 

    for q, a in zip(qns, ans):
        a = list(set([r.replace(q, "").strip() for r in a.split("\n") if q in r]))
        a = " ".join(a)        
        all_ans.append(a)
    
    return all_ans

def clean_self_evaluate_response(qns, ans):

    all_ans = [] 

    for q, a in zip(qns, ans):
        a = list(set([r.replace(q, "").strip() for r in a.split("\n") if q in r]))[0]
        a = a.replace("(A)", "").replace("(B)", "").replace(".", "").strip().lower()

        # We have to give it a default value 
        if "true" in a: all_ans.append("yes")
        if "false" in a: all_ans.append("no")
        else: all_ans.append("no")
    
    return all_ans

def clean_confidence_response(qns, ans):

    all_ans = [] 

    for q, a in zip(qns, ans):
        
        a = list(set([r.replace(q, "").strip() for r in a.split("\n") if q in r]))[0]
        a = a.split(" ")[0] # We get the first answer 

        # We have to give it a default value 
        all_ans.append(a)
    
    return all_ans