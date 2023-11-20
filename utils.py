import os
import copy
import json 

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
def get_logits(scores, tokenizer):

    """
        Get average probability score for sentence 
        Note: implementation is not optimized for speed 
    """

    scores = torch.stack(scores).permute(1, 0, -1).contiguous()
    scores = torch.softmax(scores, dim = -1)
    bs, seq_length, _ = scores.shape
    all_probabilities = []

    for b in range(bs):
        for s in range(seq_length):
            p, index = torch.max(scores[b, s, :], dim = 0)
            p = p.detach().cpu()
            print(p, index, tokenizer.convert_ids_to_tokens([index]))
            a = z

    #         a = z

    # print(len(scores))
    # print(type(scores))
    # print(scores[0].shape)

    # bs, all_scores = scores[0].shape[0], []
    
    # for pos in range(len(scores)):
        
    #     current_scores = torch.softmax(scores[[pos]], dim = -1)

    #     for b in range(bs):
        
    #     current_scores = 
    #     scores, pos = 
    #     # max_score = max(current_scores[0]).detach().cpu().numpy().tolist()
    #     # all_scores.append(max_score)

    # return all_scores

def get_model_response(inputs, llm, tokenizer, return_logits = False):

    inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors = "pt")
    outputs = llm.generate(**inputs, return_dict_in_generate = True, output_scores = True, max_new_tokens = MAX_OUTPUT_LENGTH)
    predictions = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens = True)

    logits = None
    if return_logits: 
        logits = get_logits(outputs["scores"], tokenizer)

    return predictions, logits