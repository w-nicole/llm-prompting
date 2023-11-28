def baseline_template_flan_t5(ques_list):
    return [f"Answer the following yes/no question. Do you know the answer to {q} Yes or No." for q in ques_list]

    # return [f"Are you confident that you can answer this question: {que} Say yes or no" for que in ques_list]
    
def trial_prompt_flan(ques_list):
    return [f'Are you able to answer the following question:\n\n{que}\n\nOPTIONS:\n- yes\n- no' for que in ques_list]
    
    
def prompt_with_ans_flan(ques_list, ans_list):
    return [f'Are you confident that \n\n{ans} \n\nis an answer to the following question \n\n{que}\n\nOPTIONS:\n- yes\n- no' for que, ans in zip(ques_list, ans_list)]
    
    
def baseline_template_mistral(ques_list):
    return  [f"{que}" for que in ques_list ]

def trial_prompt_mistral(ques_list):
    return [f'Are you able to answer the following question:\n\n{que}\n\nOPTIONS:\n- yes\n- no' for que in ques_list]

    
def prompt_with_ans_mistral(ques_list, ans_list):
    return [f'Are you confident that \n\n{ans} \n\nis an answer to the following question \n\n{que}\n\nOPTIONS:\n- yes\n- no' for que, ans in zip(ques_list, ans_list)]


    
def baseline_template_sheared_llama(ques_list):
    return  [f"{que}" for que in ques_list ]

def trial_prompt_mistral_sheared_llama(ques_list):
    return [f'Are you able to answer the following question:\n\n{que}\n\nOPTIONS:\n- yes\n- no' for que in ques_list]

    
def prompt_with_ans_sheared_llama(ques_list, ans_list):
    return [f'Are you confident that \n\n{ans} \n\nis an answer to the following question \n\n{que}\n\nOPTIONS:\n- yes\n- no' for que, ans in zip(ques_list, ans_list)]

# print(prompt_with_ans_flan(['are you human?'], ['no'])[0])


def get_baseline_template(model):
    if model == "mistral":
        return baseline_template_mistral
    if "flan" in model:
        return baseline_template_flan_t5
    if "sheared" in model:
        return baseline_template_sheared_llama
    else:
        raise "Not implemented yet"
        
def get_trial_prompt(model):
    if model == "mistral":
        return trial_prompt_mistral
    if "flan" in model:
        return trial_prompt_flan
    if "sheared" in model:
        return baseline_template_sheared_llama
    else:
        raise "Not implemented yet"
        
        
def get_prompt_with_ans_template(model):
    if "flan" in model:
        return prompt_with_ans_flan
    if model == 'mistral':
        return prompt_with_ans_mistral
    if "sheared" in model:
        return prompt_with_ans_sheared_llama
    else:
        raise "Not implemented yet"