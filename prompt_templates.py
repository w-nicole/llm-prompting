from config import CONFIDENCE_OPTIONS, N_DIVERSE_QUES

def baseline_template(ques_list):

    """
        Template: {ques} 
    """

    ques_list_formatted = [q for q in ques_list]

    return ques_list_formatted

def instruct_template(ques_list):

    """
        Template: Can you answer the following question: {ques} 
    """

    ques_list_formatted = [f"Can you answer the following question: {q}" for q in ques_list]

    return ques_list_formatted

def self_evaluate_template(ques_list, pred_ans_list):
   
    """
        Template: Is '{pred_ans}' the answer to the question {ques} Answer yes or no only. 
    """ 
    assert len(ques_list) == len(pred_ans_list)

    ques_list = [q[0].lower() + q[1:] for q in ques_list]
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_list_formatted = [f"Is '{a}' the answer to the question {q} Answer yes or no only." for q, a in ques_pred_ans_list]

    return ques_ans_list_formatted

def confidence_MCQ_template(ques_list, pred_ans_list):

    """
        Template: What is the level of confidence that '{pred_ans}' is the answer to the question: {ques} Choose only from {options_template}"
    """ 
    assert len(ques_list) == len(pred_ans_list)

    options_template = str(CONFIDENCE_OPTIONS).replace("'", "").replace("{", "").replace("}", "").replace(":", "")
    ques_list = [q[0].lower() + q[1:] for q in ques_list]
    ques_pred_ans_list = list(zip(ques_list, pred_ans_list))
    ques_ans_conf_list_formatted = [f"What is the level of confidence that '{a}' is the answer to the question: {q} Choose only from {options_template}." \
                                    for q, a in ques_pred_ans_list]

    return ques_ans_conf_list_formatted

def diverse_ques_gpt4_template(ques_list):

    """
        Template: Can you ask the following question `{ques}` in {k} diverse ways?"
    """ 
    ques_list = [q[0].lower() + q[1:] for q in ques_list]
    ques_list_formatted = [f"Can you ask the following question `{q}` in {N_DIVERSE_QUES} diverse ways?" for q in ques_list]
    ques_list_formatted = [{"role": "user", "content": q} for q in ques_list_formatted]
    return ques_list_formatted
