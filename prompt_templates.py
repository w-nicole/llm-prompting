from utils import format_MCQ_options
from config import CONFIDENCE_OPTIONS, CONFIDENCE_OPTIONS_NL, N_DIVERSE_QUES

# Flan-T5
def abstain_template_flan_t5(ques_list):

    """
        Template: Please answer the following question. {ques}
    """

    ques_list_formatted = [f"Answer the following yes/no question. Do you know the answer to {q} Yes or No." for q in ques_list]

    return ques_list_formatted

def get_answer_template_flan_t5(ques_list):

    ques_list_formatted = [f"Answer the following question. Question: {q}" for q in ques_list]

    return ques_list_formatted

def self_evaluate_template_flan_t5(ques_list, pred_ans_list):
   
    """
        Template: Answer the following yes/no question. Is '{pred_ans}' the answer to the question {ques}. 
    """ 
    assert len(ques_list) == len(pred_ans_list)

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Answer the following yes/no question. Is '{a}' the answer to the question '{q}'" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_template_flan_t5(ques_list, pred_ans_list):

    """
        Template:
    """ 
    assert len(ques_list) == len(pred_ans_list)

    options_template = format_MCQ_options(CONFIDENCE_OPTIONS)

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"Choose one of the options below. The likelihood that '{a}' is the answer to the question '{q}' is {options_template}. A higher score indicates higher confidence and lower score indicates lower confidence." \
                                    for q, a in ques_pred_ans_list]

    return ques_ans_conf_list_formatted

def confidence_MCQ_NL_template_flan_t5(ques_list, pred_ans_list):

    """
        Template:
    """ 
    assert len(ques_list) == len(pred_ans_list)

    options_template = format_MCQ_options(CONFIDENCE_OPTIONS_NL)

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"Choose one of the options below. How confident are you that '{a}' is the answer to the question '{q}' is {options_template}." \
                                    for q, a in ques_pred_ans_list]

    return ques_ans_conf_list_formatted

def confidence_OE_template_flan_5(ques_list, pred_ans_list):

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"Give a score between 0 to 100. The score indicates the likelihood that '{a}' is the answer to the question '{q}'. A higher score indicates higher confidence and lower score indicates lower confidence. The score is " \
                                    for q, a in ques_pred_ans_list]

    return ques_ans_conf_list_formatted

# Mistral 
def abstain_template_mistral(ques_list):

    """
        Template: Please answer the following question. {ques}
    """

    ques_list_formatted = [f"[INST] Answer only Yes or No. Question: {q} Do you know the answer to the question? [/INST]" for q in ques_list]

    return ques_list_formatted

def get_answer_template_mistral(ques_list):

    """
        Template: Answer the following question as short as possible with no explanation. Question: {q} Answer:
    """

    ques_list_formatted = [f"[INST] Provide a one sentence answer to the following question with no explanation. Question: {q} Answer: [/INST]" for q in ques_list]

    return ques_list_formatted

def self_evaluate_template_mistral(ques_list, pred_ans_list):
   
    """
        Template: "Question: {q} Proposed Answer: {a} Is the proposed answer: A) True or B) False? Proposed answer:"
    """ 
    assert len(ques_list) == len(pred_ans_list)
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"[INST] Answer only True or False. Question: {q}\n Proposed Answer: {a}\n Is the proposed answer True or False? Answer: [/INST]" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_template_mistral(ques_list, pred_ans_list):
   
    """
        Template:
    """ 

    options_template = format_MCQ_options(CONFIDENCE_OPTIONS, add_or = True)
    options_template = options_template.replace("\n", " ")

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"[INST] Question: {q} Proposed Answer: {a} How likely is the proposed answer correct? Choose either {options_template}. A higher score indicates higher confidence and lower score indicates lower confidence. Answer: [/INST]" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_NL_template_mistral(ques_list, pred_ans_list):
   
    """
        Template:
    """ 
    options_template_NL = format_MCQ_options(CONFIDENCE_OPTIONS_NL, add_or = True)
    options_template_NL = options_template_NL.replace("\n", " ")

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"[INST] Question: {q} Proposed Answer: {a} How confident are you that the proposed answer is true? Choose either {options_template_NL}. Answer: [/INST]" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_OE_template_mistral(ques_list, pred_ans_list):

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"[INST] Question: {q} Proposed Answer: {a} Give a score between 0 and 100 indicating the level of confidence that the proposed answer is true. A higher score indicates higher confidence and lower score indicates lower confidence. Score: [/INST]" \
                                    for q, a in ques_pred_ans_list]

    return ques_ans_conf_list_formatted

def get_answer_confidence_MCQ_template_mistral(ques_list):

    options_template = format_MCQ_options(CONFIDENCE_OPTIONS, add_or = True)
    options_template = options_template.replace("\n", " ")

    ques_list_formatted = [f"[INST] Provide a one sentence answer to the following question with no explanation. Then choose from either {options_template} indicating the level of confidence that the proposed answer is right. A higher score indicates higher confidence and lower score indicates lower confidence. Question: {q} Answer and score: [/INST]" for q in ques_list]

    return ques_list_formatted

def get_answer_confidence_MCQ_NL_template_mistral(ques_list):

    options_template_NL = format_MCQ_options(CONFIDENCE_OPTIONS_NL, add_or = True)
    options_template_NL = options_template_NL.replace("\n", " ")

    ques_list_formatted = [f"[INST] Provide a one sentence answer to the following question with no explanation. Then, how confident are you that the answer is right? Choose from either {options_template_NL}. Question: {q} Answer and score: [/INST]" for q in ques_list]

    return ques_list_formatted

def get_answer_confidence_OE_template_mistral(ques_list):

    ques_list_formatted = [f"[INST] Provide a one sentence answer to the following question with no explanation. Then provide a confidence score from 0 to 100 indicating the confidence that the proposed answer is right. A higher score indicates higher confidence and lower score indicates lower confidence. Question: {q} Answer and score: [/INST]" for q in ques_list]

    return ques_list_formatted

# Llama2 
def abstain_template_llama2(ques_list):

    """
        Template: Please answer the following question. {ques}
    """

    ques_list_formatted = [f"[INST] Answer only Yes or No with no explanation. Question: {q} Do you know the answer to the question? [/INST]" for q in ques_list]

    return ques_list_formatted

def get_answer_template_llama2(ques_list):

    """
        Template: Answer the following question as short as possible with no explanation. Question: {q} Answer:
    """
    ques_list_formatted = [f"[INST] Answer the following question with no explanation. Question: {q}\n Answer: [/INST]" for q in ques_list]

    return ques_list_formatted

def self_evaluate_template_llama2(ques_list, pred_ans_list):

    """
        Template: Answer the following question as short as possible with no explanation. Question: {q} Answer:
    """
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"[INST] Answer only True or False. Question: {q}\n Proposed Answer: {a}\n Is the proposed answer True or False? Answer: [/INST]" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_template_llama2(ques_list, pred_ans_list):
   
    """
        Template:
    """ 
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))

    options_template = format_MCQ_options(CONFIDENCE_OPTIONS, add_or = True)
    options_template = options_template.replace("\n", " ")

    ques_ans_list_formatted = [f"[INST] Question: {q} Proposed Answer: {a} How likely is the proposed answer correct? Choose either {options_template}. A higher score indicates higher confidence and lower score indicates lower confidence. Provide just the score without explanation. Score: [/INST]" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_NL_template_llama2(ques_list, pred_ans_list):
   
    """
        Template:
    """ 
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    options_template_NL = format_MCQ_options(CONFIDENCE_OPTIONS_NL)
    options = ""
    for i, k in enumerate(CONFIDENCE_OPTIONS_NL.keys()):
        if i == len(CONFIDENCE_OPTIONS_NL.keys()) -1: 
            options = options + "or " + k
        else:
            options = options + k + ", "
    ques_ans_list_formatted = [f"[INST] Question: {q} Proposed Answer: {a} How confident are you that the proposed answer is true? Choose either {options_template_NL}. A higher score indicates higher confidence and lower score indicates lower confidence. Provide just the score without explanation. Score: [/INST]" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_OE_template_llama2(ques_list, pred_ans_list):

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"[INST] Question: {q} Proposed Answer: {a} Give a score between 0 and 100 indicating the level of confidence that the proposed answer is true. A higher score indicates higher confidence and lower score indicates lower confidence. Provide just the score without explanation. Score: [/INST]" for q, a in ques_pred_ans_list]

    return ques_ans_conf_list_formatted

def get_answer_confidence_MCQ_template_llama2(ques_list):

    options_template = format_MCQ_options(CONFIDENCE_OPTIONS, add_or = True)
    options_template = options_template.replace("\n", " ")

    ques_list_formatted = [f"[INST] Provide a one sentence answer to the following question with no explanation. Then choose from either {options_template} indicating the level of confidence that the proposed answer is right. A higher score indicates higher confidence and lower score indicates lower confidence. Question: {q} Answer and score: [/INST]" for q in ques_list]

    return ques_list_formatted

def get_answer_confidence_MCQ_NL_template_llama2(ques_list):

    options_template_NL = format_MCQ_options(CONFIDENCE_OPTIONS_NL, add_or = True)
    options_template_NL = options_template_NL.replace("\n", " ")

    ques_list_formatted = [f"[INST] Provide a one sentence answer to the following question with no explanation. Then, how confident are you that the answer is right? Choose from either {options_template_NL}. Question: {q} Answer and score: [/INST]" for q in ques_list]

    return ques_list_formatted

def get_answer_confidence_OE_template_llama2(ques_list):

    ques_list_formatted = [f"[INST] Provide a one sentence answer to the following question with no explanation. Then provide a confidence score from 0 to 100 indicating the confidence that the proposed answer is right. A higher score indicates higher confidence and lower score indicates lower confidence. Question: {q} Answer and score: [/INST]" for q in ques_list]

    return ques_list_formatted

# GPT-3.5-turbo / GPT-4
def diverse_ques_GPT4_template(ques_list):

    """
        Template: Can you ask the following question `{ques}` in {k} diverse ways?"
    """ 
    ques_list = [q[0].lower() + q[1:] for q in ques_list]
    ques_list_formatted = [f"Can you ask the following question `{q}` in {N_DIVERSE_QUES} diverse ways?" for q in ques_list]
    ques_list_formatted = [{"role": "user", "content": q} for q in ques_list_formatted]
    return ques_list_formatted

def abstain_template_GPT4(ques_list):

    ques_list_formatted = [f"Do you know the answer to the question '{q}' answer Yes or No only." for q in ques_list]
    ques_list_formatted = [{"role": "user", "content": q} for q in ques_list_formatted]

    return ques_list_formatted

def get_answer_template_GPT4(ques_list):

    """
        Template: Can you ask the following question `{ques}` in {k} diverse ways?"
    """ 
    ques_list = [q[0].lower() + q[1:] for q in ques_list]
    ques_list_formatted = [f"Provide only the answer to the question with no explanations `{q}`" for q in ques_list]
    ques_list_formatted = [{"role": "user", "content": q} for q in ques_list_formatted]
    return ques_list_formatted

def get_self_evaluate_template_GPT4(ques_list):

    """
        Template: Can you ask the following question `{ques}` in {k} diverse ways?"
    """ 
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Is '{a}' the answer to the question '{q}'? Answer Yes no No only." for q, a in ques_pred_ans_list]
    ques_ans_list_formatted = [{"role": "user", "content": q} for q in ques_ans_list_formatted]
    return ques_list_formatted

def confidence_OE_template_GPT4(ques_list, pred_ans_list):

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"Give a score between 0 and 100 indicating the level of confidence that '{a}' is the answer to '{q}'. A higher score indicates higher confidence and lower score indicates lower confidence." for q, a in ques_pred_ans_list]
    ques_ans_conf_list_formatted = [{"role": "user", "content": q} for q in ques_ans_conf_list_formatted]
    
    return ques_ans_conf_list_formatted

# Function to get the correct functions
def get_abstain_template(model):

    if "flan-t5" in model:
        return abstain_template_flan_t5
    elif "mistral" in model:
        return abstain_template_mistral
    elif "llama2" in model: 
        return abstain_template_llama2
    elif "shearedllama" in model:
        return abstain_template_llama2
    else: 
        raise NotImplementedError() 
    
def get_get_answer_template(model):

    if "flan-t5" in model: 
        return get_answer_template_flan_t5
    elif "mistral" in model:
        return get_answer_template_mistral
    elif "llama2" in model: 
        return get_answer_template_llama2
    elif "shearedllama" in model:
        return get_answer_template_llama2
    else: 
        raise NotImplementedError() 

def get_self_evaluate_template(model):

    if "flan-t5" in model: 
        return self_evaluate_template_flan_t5
    elif "mistral" in model:
        return self_evaluate_template_mistral
    elif "llama2" in model: 
        return self_evaluate_template_llama2
    elif "shearedllama" in model:
        return self_evaluate_template_shearedllama
    else: 
        raise NotImplementedError() 

def get_confidence_MCQ_template(model):

    if "flan-t5" in model: 
        return confidence_MCQ_template_flan_t5
    elif "mistral" in model:
        return confidence_MCQ_template_mistral
    elif "llama2" in model: 
        return confidence_MCQ_template_llama2
    elif "shearedllama" in model:
        return confidence_MCQ_template_llama2
    else: 
        raise NotImplementedError() 

def get_confidence_MCQ_NL_template(model):

    if "flan-t5" in model: 
        return confidence_MCQ_NL_template_flan_t5
    elif "mistral" in model:
        return confidence_MCQ_NL_template_mistral
    elif "llama2" in model: 
        return confidence_MCQ_NL_template_llama2
    elif "shearedllama" in model:
        return confidence_MCQ_NL_template_llama2
    else: 
        raise NotImplementedError() 

def get_confidence_OE_template(model):

    if "flan-t5" in model: 
        return confidence_OE_template_flan_5
    elif "mistral" in model:
        return confidence_OE_template_mistral
    elif "llama2" in model: 
        return confidence_OE_template_llama2
    elif "shearedllama" in model:
        return confidence_OE_template_llama2
    else: 
        raise NotImplementedError() 