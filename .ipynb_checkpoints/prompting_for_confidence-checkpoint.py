import os 
from tqdm import tqdm 

from dataloaders import get_dataloader
from models import get_model_and_tokenizer

from config import * 
from utils import get_model_response
from prompt_templates import baseline_template, self_evaluate_template, confidence_MCQ_template

def get_response(dataloader, llm, tokenizer):

    all_records = {} 

    for batch in tqdm(dataloader):

        # Unpack the batch 
        id_, qns, ans = batch 

        # Get original question
        qns_formatted = baseline_template(qns)

        # Get responses 
        response, avg_prob = get_model_response(qns_formatted, llm, tokenizer, return_prob = True)

        # Get self evaluation
        qns_response = self_evaluate_template(qns, response)
        self_evaluate_response, _ = get_model_response(qns_response, llm, tokenizer)

        # Get confidence score
        qns_response_conf = confidence_MCQ_template(qns, response)
        conf_response, _ = get_model_response(qns_response_conf, llm, tokenizer)

        # Get self evaluation (with ground truth ans)
        qns_ans = self_evaluate_template(qns, ans)
        gt_ans_response, _ = get_model_response(qns_ans, llm, tokenizer)

        # Get confidence score (with ground truth ans)
        qns_ans_conf = confidence_MCQ_template(qns, ans)
        gt_ans_conf_response, _ = get_model_response(qns_ans_conf, llm, tokenizer)

        # Merge all the responses together
        all_records[id_] = [qns, ans, qns_response, self_evaluate_response, conf_response, gt_ans_response, gt_ans_conf_response]
        
    return all_records


def prompt_trial(ques_list):
    return [f"Question: {que} \n\n Context: None" for que in ques_list]
    


def get_response_with_given_prompt(dataloader, llm, tokenizer):
    all_records = []
    
    for batch in tqdm(dataloader):

        # Unpack the batch 
        id_, qns, ans = batch 

        # Get original question
        qns_formatted = prompt_trial(qns)

        # Get responses 
        response, avg_prob = get_model_response(qns_formatted, llm, tokenizer, return_prob = True)
        
        # Merge all the responses together
        all_records.append((qns, ans, response))
        
    return all_records


if __name__ == "__main__":

    # Load the model and dataloader 
    llm, tokenizer = get_model_and_tokenizer("flan-t5-small")
    dataloader = get_dataloader("truthfulqa", os.path.join(TRUTHFUL_QA_PROCESSED_PATH, "train.json"))    
    outputs = get_response_with_given_prompt(dataloader, llm, tokenizer)
    
    for qns, ans, resp in outputs:
        print(f"question: {qns}")
        print(f"answer: {ans}")
        print(f"response: {resp} \n")
    