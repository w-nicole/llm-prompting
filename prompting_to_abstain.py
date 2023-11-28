import os 
from tqdm import tqdm 
from config import * 

from dataloaders import get_dataloader
from models import get_model_and_tokenizer
from prompt_template_sumiya import * 
from utils import get_model_response, clean_response, clean_self_evaluate_response, clean_confidence_response, write_json

def get_response_with_given_prompt(dataloader, llm, tokenizer):
    
    all_results = []
    
    num_ans_no_baseline = 0
    num_ans_no_trial = 0
    num_ans_no_qwa = 0
    num_que = 0
    for batch in tqdm(dataloader):
        
        # Unpack the batch 
        id_, qns, ans = batch 

        # pass through the prompt 
        qns_formatted = BASELINE_TEMPLATE(qns)
        qns_trial = TRIAL_TEMPLATE(qns)
        qns_w_ans = PROMPT_W_ANS(qns, ans)
        
        num_que += len(qns_formatted)
        
        # Get responses 
        response, avg_prob = get_model_response(qns_formatted, llm, tokenizer, return_prob = True)
        for answer in response:
            num_ans_no_baseline += answer.lower() == 'no' 
        
        response, avg_prob = get_model_response(qns_trial, llm, tokenizer, return_prob = True)
        for answer in response:
            num_ans_no_trial += answer.lower() == 'no'
        
        response, avg_prob = get_model_response(qns_w_ans, llm, tokenizer, return_prob = True)
        for answer in response:
            num_ans_no_qwa += answer.lower() == 'no' 
        
        
        # if "shearedllama" in MODEL: response = clean_response(qns_formatted, response) # Note we do not need to clean this for Flan T5

    print(f'Baseline {num_ans_no_baseline}/{num_que}')
    print(f'Trial {num_ans_no_trial}/{num_que}')
    print(f'Question with Answer {num_ans_no_qwa}/{num_que}')
    
if __name__ == "__main__":
    DATASET = "truthfulqa"
    MODEL = "shearedllama-2.7b"
    DATASET_FOLDER  = TRUTHFUL_QA_PROCESSED_FOLDER
    OUTPUT_FOLDER =  TRUTHFUL_QA_OUTPUT_FOLDER
    
    # Load the prompt templates 
    BASELINE_TEMPLATE = get_baseline_template(MODEL)
    TRIAL_TEMPLATE = get_trial_prompt(MODEL)
    PROMPT_W_ANS = get_prompt_with_ans_template(MODEL)

    # Load the model and dataloader 
    llm, tokenizer = get_model_and_tokenizer(MODEL)
    if "sheared" in MODEL and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        llm.resize_token_embeddings(len(tokenizer))
        
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataloader = get_dataloader("truthfulqa", os.path.join(DATASET_FOLDER, "train.json"))    
    outputs = get_response_with_given_prompt(dataloader, llm, tokenizer)
  