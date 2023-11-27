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

    ques_list_formatted = [f"Question: {q} Do you know the answer to the question? (A) Yes or (B) No? Answer:" for q in ques_list]

    return ques_list_formatted

def get_answer_template_mistral(ques_list):

    """
        Template: Answer the following question as short as possible with no explanation. Question: {q} Answer:
    """

    ques_list_formatted = [f"Provide an answer to the following question with no explanation. Question: {q} Answer:" for q in ques_list]

    return ques_list_formatted

def self_evaluate_template_mistral(ques_list, pred_ans_list):
   
    """
        Template: "Question: {q} Proposed Answer: {a} Is the proposed answer: A) True or B) False? Proposed answer:"
    """ 
    assert len(ques_list) == len(pred_ans_list)
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Question: {q} Proposed Answer: {a} Is the proposed answer: A) True or B) False? Proposed answer:" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_template_mistral(ques_list, pred_ans_list):
   
    """
        Template:
    """ 
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    options_template = format_MCQ_options(CONFIDENCE_OPTIONS, add_or = True)

    options_template = options_template.replace("\n", " ")
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Question: {q} Proposed Answer: {a} Is the estimated level of confidence that the proposed answer is true: {options_template}. A higher score indicates higher confidence and lower score indicates lower confidence. Confidence:" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_NL_template_mistral(ques_list, pred_ans_list):
   
    """
        Template:
    """ 
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    options_template = format_MCQ_options(CONFIDENCE_OPTIONS_NL, add_or = True)

    options_template = options_template.replace("\n", " ")
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Question: {q} Proposed Answer: {a} How confident are you that the proposed answer is true: {options_template}. Choice:" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_OE_template_mistral(ques_list, pred_ans_list):

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"Question: {q} Proposed Answer: {a} Give a score between 0 and 100 indicating the level of confidence that the proposed answer is true. A higher score indicates higher confidence and lower score indicates lower confidence. Score:" \
                                    for q, a in ques_pred_ans_list]

    return ques_ans_conf_list_formatted

# Llama2 
def abstain_template_llama2(ques_list):

    """
        Template: Please answer the following question. {ques}
    """

    ques_list_formatted = [f"Choose only A) or B). \n Question: {q}\n A) I can answer this factually\n B) I cannot answer this factually\n Answer: " for q in ques_list]

    return ques_list_formatted

def get_answer_template_llama2(ques_list):

    """
        Template: Answer the following question as short as possible with no explanation. Question: {q} Answer:
    """
    ques_list_formatted = [f"Answer in one sentence. Question: {q} Answer:" for q in ques_list]

    return ques_list_formatted

def self_evaluate_template_llama2(ques_list, pred_ans_list):

    """
        Template: Answer the following question as short as possible with no explanation. Question: {q} Answer:
    """
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Choose only A) or B).\n Question: {q}\n Proposed Answer: {a}\n A) The proposed answer is right \n B) The proposed answer is wrong \n Answer:" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_template_llama2(ques_list, pred_ans_list):
   
    """
        Template:
    """ 
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    options_template = format_MCQ_options(CONFIDENCE_OPTIONS)
    options = ""
    for i, k in enumerate(CONFIDENCE_OPTIONS.keys()):
        if i == len(CONFIDENCE_OPTIONS.keys()) -1: 
            options = options + "or " + k
        else:
            options = options + k + ", "
    ques_ans_list_formatted = [f"Choose only {options}. Question: {q}\n Proposed Answer: {a}\n Is the estimated level of confidence that the proposed answer is correct: {options_template}. \n A higher score indicates higher confidence and lower score indicates lower confidence. \n Answer:" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_NL_template_llama2(ques_list, pred_ans_list):
   
    """
        Template:
    """ 
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    options_template = format_MCQ_options(CONFIDENCE_OPTIONS_NL)
    options = ""
    for i, k in enumerate(CONFIDENCE_OPTIONS.keys()):
        if i == len(CONFIDENCE_OPTIONS.keys()) -1: 
            options = options + "or " + k
        else:
            options = options + k + ", "
    ques_ans_list_formatted = [f"Choose only {options}. Question: {q}\n Proposed Answer: {a}\n How confident are you that the proposed answer is correct?\n {options_template}. \n A higher score indicates higher confidence and lower score indicates lower confidence. \n Answer:" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_OE_template_llama2(ques_list, pred_ans_list):

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"Give a score between 0 and 100 indicating the level of confidence that the proposed answer is true.\n A higher score indicates higher confidence and lower score indicates lower confidence.\n Question: {q}\n Proposed Answer: {a}\n Score:" for q, a in ques_pred_ans_list]

    return ques_ans_conf_list_formatted

# shearedllama
# def abstain_template_shearedllama(ques_list):

#     return 

# def get_answer_template_shearedllama(ques_list):

#     """
#         Template: Answer the following question as short as possible with no explanation. Question: {q} Answer:
#     """
#     ques_list_formatted = [f"Answer the following question as factually as possible with no explanation. Question: {q} Answer:" for q in ques_list]

#     return ques_list_formatted

# def self_evaluate_template_shearedllama(ques_list, pred_ans_list):
   
#     """
#         Template: "Choose one of the options below. Question: {q} Proposed Answer: {a} Is the proposed answer: (A) True or (B) False? Proposed answer:"
#     """ 
#     assert len(ques_list) == len(pred_ans_list)

#     ques_list = [q[0].lower() + q[1:] for q in ques_list]
#     ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
#     ques_ans_list_formatted = [f"Choose one of the options below. Question: {q} Proposed Answer: {a} Is the proposed answer: (A) True or (B) False? Proposed answer:" for q, a in ques_pred_ans_list]

#     return ques_ans_list_formatted

# def confidence_MCQ_template_shearedllama(ques_list, pred_ans_list, NL = False):
   
#     """
#         Template: "Choose one of the options below. Question: {q} Proposed Answer: {a} Is the level of confidence that the proposed answer is right: <Options>. Confidence:"
#     """ 
#     assert len(ques_list) == len(pred_ans_list)

#     ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
#     if NL:
#         options_template = CONFIDENCE_OPTIONS_NL
#     else:
#         options_template = CONFIDENCE_OPTIONS

#     options_template = format_MCQ_options(options_template, add_or = True)
#     options_template = options_template.replace("\n", " ")
#     print(options_template)
#     ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
#     ques_ans_list_formatted = [f"Choose one of the options below. Question: {q} Proposed Answer: {a}. Is the level of confidence that the proposed is right: {options_template}? Estimated confidence:" for q, a in ques_pred_ans_list]

#     return ques_ans_list_formatted

# def confidence_MCQ_NL_template_shearedllama(ques_list, pred_ans_list):
#     return 

# def confidence_OE_template_shearedllama(ques_list, pred_ans_list):
#     return

# Diverse prompts 
def diverse_ques_gpt4_template(ques_list):

    """
        Template: Can you ask the following question `{ques}` in {k} diverse ways?"
    """ 
    ques_list = [q[0].lower() + q[1:] for q in ques_list]
    ques_list_formatted = [f"Can you ask the following question `{q}` in {N_DIVERSE_QUES} diverse ways?" for q in ques_list]
    ques_list_formatted = [{"role": "user", "content": q} for q in ques_list_formatted]
    return ques_list_formatted

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
        return self_evaluate_template_llama2
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