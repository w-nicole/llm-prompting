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

        # Format the according to template (TO BE DONE)
        qns = baseline_template(qns)

        # Get responses 
        response, avg_prob = get_model_response(qns, llm, tokenizer, return_logits = True)

        print(qns_response_conf)
        print(conf_response)

        a = z 

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


        # get_model_response(llm, tokenizer)
        print(batch)    
        print("here")
        a = z 
    
    None

if __name__ == "__main__":

    # Load the model and dataloader 
    llm, tokenizer = get_model_and_tokenizer("flan-t5-small")
    dataloader = get_dataloader("truthfulqa", os.path.join(TRUTHFUL_QA_PROCESSED_PATH, "train.json"))
    outputs = get_response(dataloader, llm, tokenizer)
    
    

    print("okay i am here")
    