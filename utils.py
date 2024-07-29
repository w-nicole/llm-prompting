import os
import re
import math
import json 
import numpy as np

import torch

from config import *

def get_samples(total_size, sample_size, n_samples):

    chosen_samples = []
    while len(chosen_samples) < n_samples:

        choice = np.random.choice(list(range(total_size)), size = sample_size, replace = False)
        choice = sorted(choice)
        if choice not in chosen_samples:
            chosen_samples.append(choice)
    
    return chosen_samples

# To get the respective folders
def get_folders(dataset_name):

    if dataset_name == "truthfulqa":
        return TRUTHFUL_QA_RAW_FOLDER, TRUTHFUL_QA_PROCESSED_FOLDER, TRUTHFUL_QA_OUTPUT_FOLDER
    elif dataset_name == "sciq":
        return SCIQ_RAW_FOLDER, SCIQ_PROCESSED_FOLDER, SCIQ_OUTPUT_FOLDER
    elif dataset_name == "triviaqa":
        return TRIVIA_QA_RAW_FOLDER, TRIVIA_QA_PROCESSED_FOLDER, TRIVIA_QA_OUTPUT_FOLDER
    elif dataset_name == 'popqa':
        return POPQA_RAW_FOLDER, POPQA_PROCESSED_FOLDER, POPQA_OUTPUT_FOLDER
    else: 
        raise NotImplementedError()

# To clean llama2 weights
def clean_llama2_weights(folder = "./model_weights/llama2/7b"):
    
    """
    Changes the prefix of the state dict of a llama chekpoint to match the names in the BCI architecture
    """

    shards = [
        "pytorch_model-00001-of-00003.bin",
        "pytorch_model-00002-of-00003.bin",
        "pytorch_model-00003-of-00003.bin",
    ]
    shards = [os.path.join(folder, s) for s in shards]

    index_name = os.path.join(folder, "pytorch_model.bin.index.json")

    for shard_name in shards:
        shard = torch.load(shard_name)
        new_shard = {re.sub("decoder\\.transformer", "model", key): shard[key] for key in shard}
        new_shard_2 = {re.sub("decoder\\.lm_head", "lm_head", key): new_shard[key] for key in new_shard}
        torch.save(new_shard_2, shard_name)

    with open(index_name, "r") as ifile:
        index = json.load(ifile)
        new_wmap = {re.sub("decoder\\.transformer", "model", key): index["weight_map"][key] for key in index["weight_map"]}
        new_wmap_2 = {re.sub("decoder\\.lm_head", "lm_head", key): new_wmap[key] for key in new_wmap}
        index["weight_map"] = new_wmap_2

    with open(index_name, "w") as ofile:
        json.dump(index, ofile)

# For data read write
def read_json(path):
    
    with open(path, "r", encoding = "UTF-8") as f:
        data = json.load(f)

    return data

def write_json(data, path):

    with open(path, "w", encoding = "UTF-8") as f:
        json.dump(data, f, indent = 4)

# To process diverse outputs
def unpack_qns(diverse_qns):

    all_qns = []
    start_idx = 0
    idx_list = []
    for q in diverse_qns:
        q = [r for r in q if r != ""]
        all_qns.extend(q)
        idx_list.append((start_idx, start_idx + len(q)))
        start_idx += len(q)

    return idx_list, all_qns

def pack_qns_ans(idx_list, qns_list, ans_list):

    qns_packed, ans_packed = [],[]
    for (s, e) in idx_list:

        qns_packed.append(qns_list[s:e])
        ans_packed.append(ans_list[s:e])
    
    return qns_packed, ans_packed

# To turn chosen confidence option to numerical
def parse_option(option):

    try:
        option = option.replace("%", "")
        option_list = [r.strip() for r in option.split("to")]
        if len(option_list) == 2: 
            l, u = float(option_list[0]), float(option_list[1])
            val = (l + u) / 2.0
        else:
            val = float(option_list[0])
    except:
        val = option 
    return val

# For prompting
def format_MCQ_options(options, add_or = False):
    format = ""
    for i, (k,v) in enumerate(options.items()):
        if i == len(options)-1:
            if add_or:
                format = format + f"or {k} {v}"
            else:
                format = format + f"{k} {v}"
        else:
            format = format + f"{k} {v}\n"

    return format 

def get_model_response(inputs, llm, tokenizer, max_new_tokens = MAX_OUTPUT_LENGTH, return_raw = False):

    inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors = "pt", max_length = MAX_OUTPUT_LENGTH)
    inputs.to(llm.device) # Move to GPU / CPU
    outputs = llm.generate(**inputs, return_dict_in_generate = True, 
                                     output_scores = True, 
                                     max_new_tokens = max_new_tokens, 
                                     repetition_penalty = 1.0)
    predictions = tokenizer.batch_decode(outputs["sequences"], skip_special_tokens = True)

    if return_raw:
        return outputs

    return predictions

# Get probability of true functions 
def get_true_prob(inputs, llm, tokenizer, true_idx, false_idx, llama2 = False):

    if llama2:
        model_output = get_model_response(inputs, llm, tokenizer, max_new_tokens = 2, return_raw = True)
    else: 
        model_output = get_model_response(inputs, llm, tokenizer, max_new_tokens = 1, return_raw = True)

    # Get the scores
    scores = model_output["scores"][-1] # Last token in response
    prob = torch.concat([scores[:, true_idx].unsqueeze(-1), scores[:, false_idx].unsqueeze(-1)], dim = -1)
    # prob = torch.where(torch.isneginf(prob), 1.0, prob) # For stability
    prob = torch.softmax(prob, dim = -1)[:, 0]
    prob = prob.detach().cpu().numpy().tolist()

    return prob

# Cleaning functions
# Flan T5
def clean_abstain_flan_t5(qns, ans):

    all_ans = [] 
    for a in ans:
        a = a.lower().strip()
        if "yes" in a and "no" in a: 
            all_ans.append(1) # Abstain if unsure
        if "yes" in a: 
            all_ans.append(0) # Do not abstain if model knows the answer
        if "no" in a:
            all_ans.append(1) # Abstain if model do not know the answer
        else:
            all_ans.append(a) # To check later

    return all_ans

def clean_answer_flan_t5(qns, ans):

    all_ans = [] 
    for a in ans:
        all_ans.append(a)
    return all_ans

def clean_self_eval_flan_t5(qns, ans):
    all_ans = [] 
    for a in ans:
        a = a.lower()
        if "yes" in a and "no" in a: 
            all_ans.append(0) # Wrong answer if model returns both yes and no
        if "yes" in a: 
            all_ans.append(1) # correct if model answers yes
        if "no" in a:
            all_ans.append(0) # wrong if model answers no
        else:
            all_ans.append(a) # For manual inspection

    return all_ans

def clean_confidence_MCQ_flan_t5(qns, ans, NL = False):

    all_ans = []
    if NL:
        options_template = {k : v for k, v in CONFIDENCE_OPTIONS_NL.items()}
    else:
        options_template = {k : v for k, v in CONFIDENCE_OPTIONS.items()}

    for a in ans:
        a = a.replace("\n", "")
        a = a.strip()
        if a in options_template:
            v = options_template[a]
            if NL: v = CONFIDENCE_SCORE_NL_MAPPING[v]
            all_ans.append(v)
        else:
            all_ans.append(a)
    
    all_ans = [parse_option(a) for a in all_ans]

    return all_ans

def clean_confidence_OE_flan_t5(qns, ans):

    all_ans = []

    for a in ans:

        a = a.replace("\n", "").strip()
        try: 
            a = float(a)
        except:
            print(a)
            a = 0.0
        all_ans.append(a)

    return all_ans

# mistral
def clean_abstain_mistral(qns, ans):

    all_ans = [] 

    for q, a in zip(qns, ans):

        a = a.replace(q, "").strip().lower()
        if "yes" in a and "no" in a:
            all_ans.append(1) # Abstain if unsure 
        elif "yes" in a:
            all_ans.append(0) # Do not abstain if model knows the answer 
        elif "no" in a:
            all_ans.append(1) # Abstain if model do not know the answer
        else:
            all_ans.append(a) # For manual inspection
    return all_ans

def clean_answer_mistral(qns, ans):

    all_ans = [] 

    for q, a in zip(qns, ans):
        a = a.replace(q, "").strip()
        all_ans.append(a)
    
    return all_ans

def clean_self_eval_mistral(qns, ans):

    all_ans = [] 

    for q, a in zip(qns, ans):
        a = a.replace(q, "").strip().lower() 

        if "true" in a and "false" in a: 
            all_ans.append(0) # Not sure of the answer 
        elif "true" in a: 
            all_ans.append(1) # Sure that answer is right 
        elif "false" in a:
            all_ans.append(0) # Not sure that the answer is right 
        else:
            all_ans.append(a) # For manual inspection

    return all_ans

def clean_confidence_MCQ_mistral(qns, ans, NL = False):

    all_ans = [] 

    if NL:
        options_template = {k : v for k, v in CONFIDENCE_OPTIONS_NL.items()}
    else:
        options_template = {k : v for k, v in CONFIDENCE_OPTIONS.items()}

    for q, a in zip(qns, ans):
        
        check = False
        a = a.split("Answer:")[-1].strip()
        for _, v in options_template.items(): 
            if v in a: 
                a = v
                check = True 
                break 
        if check and NL: 
            a = CONFIDENCE_SCORE_NL_MAPPING[a]
        all_ans.append(a)

    all_ans = [parse_option(a) for a in all_ans]

    return all_ans

def clean_confidence_OE_mistral(qns, ans):
    
    all_ans = []

    for q, a in zip(qns, ans):
        
        a = a.replace(q, "").strip().split(".")[0]
        all_ans.append(a)

    all_ans = [parse_option(a) for a in all_ans]
    return all_ans

def clean_answer_and_confidence_MCQ_mistral(qns, ans, NL = False):

    all_ans, all_scores = [], []

    for q, a in zip(qns, ans):
        
        a = a.replace(q, "").strip()
        a_list = a.split(".")
        if NL: 
            a, s = ".".join(a_list[:-2]).strip(), ".".join(a_list[-2:-1]).strip()
        else: 
            a, s = ".".join(a_list[:-1]).strip(), a_list[-1].strip()
        all_ans.append(a)
        all_scores.append(s)

    all_scores = clean_confidence_MCQ_mistral(qns, all_scores, NL = NL)

    return all_ans, all_scores

def clean_answer_and_confidence_OE_mistral(qns, ans):

    all_ans, all_scores = [], []

    for q, a in zip(qns, ans):
        
        try:
            a = a.replace(q, "").strip()
            idx = a.index("Confidence score:")

            s = a[idx:].replace("Confidence score:", "").replace(".", "").strip()
            
            all_ans.append(a[:idx].strip())
            all_scores.append(s)

        except:
            all_ans.append(a)
            all_scores.append(a)

    all_scores = [parse_option(s) for s in all_scores]

    return all_ans, all_scores

# llama2
def clean_abstain_llama2(qns, ans):

    all_ans = [] 

    for q, a in zip(qns, ans):

        a = a.replace(q, "").strip().lower()
        if "yes" in a and "no" in a:
            all_ans.append(1) # Abstain if unsure 
        elif "yes" in a:
            all_ans.append(0) # Do not abstain if model knows the answer 
        elif "no" in a:
            all_ans.append(1) # Abstain if model do not know the answer
        else:
            all_ans.append(a) # For manual inspection
    return all_ans

def clean_answer_llama2(qns, ans):

    all_ans = [] 

    for q, a in zip(qns, ans):
        a = a.replace(q, "").strip()
        all_ans.append(a)
    
    return all_ans

def clean_self_eval_llama2(qns, ans):

    all_ans = [] 

    for q, a in zip(qns, ans):

        a = a.replace(q, "").strip().split(".")[0]

        if "True" in a and "False" in a:
            all_ans.append(0) # Append false if unsure 
        elif "True" in a: 
            all_ans.append(1)
        elif "False" in a: 
            all_ans.append(0)
        else:
            all_ans.append(a) # For manual inspection
    return all_ans

def clean_confidence_MCQ_llama2(qns, ans, NL = False):

    all_ans = [] 
    
    if NL:
        options_template = {k : v for k, v in CONFIDENCE_OPTIONS_NL.items()}
    else:
        options_template = {k : v for k, v in CONFIDENCE_OPTIONS.items()}

    for q, a in zip(qns, ans):
        
        check = False
        a = a.split("Score:")[-1].strip()
        a = a.replace("[/INST]", "")
        a = " ".join(a.split())
        for _, v in options_template.items(): 
            if v in a: 
                a = v
                check = True 
                break 
        
        if not check:
            for k, v in options_template.items():
                k = k.replace("(", "").replace(")", "").strip()
                if k in a:
                    a = v
                    check = True
                    break
        
        if check and NL: 
            a = CONFIDENCE_SCORE_NL_MAPPING[a]
        all_ans.append(a)

    all_ans = [parse_option(a) for a in all_ans]

    return all_ans


def clean_confidence_OE_llama2(qns, ans):
    
    all_ans = []

    for q, a in zip(qns, ans):
        
        a = a.replace(q, "").strip().split("Confidence Level:")[-1].strip()
        a = a.replace(".", "")
        all_ans.append(a)

    all_ans = [parse_option(a) for a in all_ans]

    return all_ans

def clean_answer_and_confidence_MCQ_llama2(qns, ans, NL = False):

    all_ans, all_scores = [], []

    for q, a in zip(qns, ans):
        
        a = a.replace(q, "").strip()
        a_list = a.split(".")

        if NL: 
            a, s = ".".join(a_list[:-2]).strip(), ".".join(a_list[-2:-1]).strip()
        else: 
            a, s = ".".join(a_list[:-1]).strip(), a_list[-1].strip()
        
        if a == "": a = ".".join(a_list)
        all_ans.append(a)
        all_scores.append(s)

    all_scores = clean_confidence_MCQ_llama2(qns, all_scores, NL = NL)

    return all_ans, all_scores

def clean_answer_and_confidence_OE_llama2(qns, ans):

    all_ans, all_scores = [], []

    for q, a in zip(qns, ans):
        
        a = a.replace(q, "").strip()
        try:
            idx = a.index("Confidence score:")
            s = a[idx:].replace("Confidence score:", "").replace(".", "").strip()
            all_ans.append(a[:idx].strip())
            all_scores.append(s)

        except:
            all_ans.append(a)
            all_scores.append(a)

    all_scores = [parse_option(s) for s in all_scores]

    return all_ans, all_scores

# Function to get the appropriate function
def get_clean_abstain_fnc(model):
    if "flan-t5" in model: 
        return clean_abstain_flan_t5
    elif "mistral" in model:
        return clean_abstain_mistral
    elif "llama2" in model:
        return clean_abstain_llama2
    else: 
        NotImplementedError()    

def get_clean_answer_fnc(model):
    if "flan-t5" in model: 
        return clean_answer_flan_t5
    elif "mistral" in model:
        return clean_answer_mistral
    elif "llama2" in model:
        return clean_answer_llama2
    else: 
        NotImplementedError()

def get_clean_self_evaluate_fnc(model):
    if "flan-t5" in model: 
        return clean_self_eval_flan_t5
    elif "mistral" in model:
        return clean_self_eval_mistral
    elif "llama2" in model: 
        return clean_self_eval_llama2
    else: 
        NotImplementedError()

def get_clean_confidence_MCQ_fnc(model):
    if "flan-t5" in model: 
        return clean_confidence_MCQ_flan_t5
    elif "mistral" in model:
        return clean_confidence_MCQ_mistral
    elif "llama2" in model: 
        return clean_confidence_MCQ_llama2
    else: 
        NotImplementedError()

def get_clean_confidence_OE_fnc(model):
    if "flan-t5" in model: 
        return clean_confidence_OE_flan_t5
    elif "mistral" in model:
        return clean_confidence_OE_mistral
    elif "llama2" in model: 
        return clean_confidence_OE_llama2
    else: 
        NotImplementedError()

def get_clean_answer_and_confidence_MCQ_fnc(model):
    if "flan-t5" in model: 
        raise NotImplementedError()
    elif "mistral" in model:
        return clean_answer_and_confidence_MCQ_mistral
    elif "llama2" in model: 
        return clean_answer_and_confidence_MCQ_llama2
    else: 
        NotImplementedError()

def get_clean_answer_and_confidence_OE_fnc(model):
    if "flan-t5" in model: 
        raise NotImplementedError()
    elif "mistral" in model:
        return clean_answer_and_confidence_OE_mistral
    elif "llama2" in model: 
        return clean_answer_and_confidence_OE_llama2
    else: 
        NotImplementedError()
