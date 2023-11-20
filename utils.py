import os
import copy
import json 
import numpy as np

import torch

from config import MAX_OUTPUT_LENGTH

# For data read write
def read_json(path):
    
    with open(path, "r", encoding = "UTF-8") as f:
        data = json.load(f)

    return data

def write_json(data, path):

    with open(path, "w", encoding = "UTF-8") as f:
        json.dump(data, f, indent = 4)

# For prompting
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
    outputs = llm.generate(**inputs, return_dict_in_generate = True, output_scores = True, max_new_tokens = MAX_OUTPUT_LENGTH)
    predictions = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens = True)

    prob = None
    if return_prob: 
        prob = get_avg_prob(outputs["scores"], tokenizer)

    return predictions, prob