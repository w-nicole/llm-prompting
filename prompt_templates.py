from utils import format_MCQ_options
from config import CONFIDENCE_OPTIONS, N_DIVERSE_QUES

# Get model specific templates
# Flan-T5
def baseline_template_flan_t5(ques_list):

    """
        Template: Please answer the following question. {ques}
    """

    ques_list_formatted = [f"Please answer the following question. {q}" for q in ques_list]

    return ques_list_formatted

def self_evaluate_template_flan_t5(ques_list, pred_ans_list):
   
    """
        Template: Answer the following yes/no question. Is '{pred_ans}' the answer to the question {ques}. 
    """ 
    assert len(ques_list) == len(pred_ans_list)

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Answer the following yes/no question. Is '{a}' the answer to the question {q}" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_template_flan_t5(ques_list, pred_ans_list):

    """
        Template: Answer the following multiple choice question. What is the level of confidence that '{pred_ans}' is the answer to the question: {ques}"
    """ 
    assert len(ques_list) == len(pred_ans_list)

    options_template = format_MCQ_options(CONFIDENCE_OPTIONS)
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"What is the level of confidence that '{a}' is the answer to the question: {q} Options:\n {options_template}." \
                                    for q, a in ques_pred_ans_list]

    return ques_ans_conf_list_formatted

# shearedllama
def baseline_template_shearedllama(ques_list):

    """
        Template: Answer the following question as short as possible with no explanation. Question: {q} Answer:
    """

    ques_list_formatted = [f"Answer the following question without any elaboration. Question: {q} Answer:" for q in ques_list]

    return ques_list_formatted

def self_evaluate_template_shearedllama(ques_list, pred_ans_list):
   
    """
        Template: "Question: {q} Proposed Answer: {a} Is the proposed answer: (A) True or (B) False? Proposed answer:"
    """ 
    assert len(ques_list) == len(pred_ans_list)

    ques_list = [q[0].lower() + q[1:] for q in ques_list]
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Question: {q} Proposed Answer: {a} Is the proposed answer: (A) True or (B) False? Proposed answer:" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_template_shearedllama(ques_list, pred_ans_list):
   
    """
        Template: "Question: {q} Proposed Answer: {a} Level of confidence proposed answer is right: <Options>. Confidence:"
    """ 
    assert len(ques_list) == len(pred_ans_list)

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    options_template = format_MCQ_options(CONFIDENCE_OPTIONS, add_or = True)
    options_template = options_template.replace("\n", " ")

    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Question: {q} Proposed Answer: {a} Is the estimated level of confidence that the proposed answer is true: {options_template}. Confidence:" for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

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
def get_baseline_template(model):

    if "flan-t5" in model: 
        return baseline_template_flan_t5

    if "mistral" in model:
        return baseline_template_flan_t5
    
    if "shearedllama" in model:
        return baseline_template_shearedllama
    
    else: 
        raise NotImplementedError() 

def get_self_evaluate_template(model):

    if "flan-t5" in model: 
        return self_evaluate_template_flan_t5

    if "mistral" in model:
        return self_evaluate_template_flan_t5
    
    if "shearedllama" in model:
        return self_evaluate_template_shearedllama
    
    else: 
        raise NotImplementedError() 

def get_confidence_MCQ_template(model):

    if "flan-t5" in model: 
        return confidence_MCQ_template_flan_t5

    if "mistral" in model:
        return confidence_MCQ_template_flan_t5
        
    if "shearedllama" in model:
        return confidence_MCQ_template_shearedllama
    
    else: 
        raise NotImplementedError() 