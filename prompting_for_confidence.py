import os 
from tqdm import tqdm 

from dataloaders import get_dataloader
from models import get_model_and_tokenizer

from config import * 
from utils import get_model_response, clean_response, clean_self_evaluate_response, clean_confidence_response, write_json
from prompt_templates import get_baseline_template, get_self_evaluate_template, get_confidence_MCQ_template

def get_response(dataloader, llm, tokenizer):

    all_results = []

    for batch in tqdm(dataloader):

        # Unpack the batch 
        id_, qns, ans = batch 

        # Get original question
        qns_formatted = BASELINE_TEMPLATE(qns)

        # Get responses 
        response, avg_prob = get_model_response(qns_formatted, llm, tokenizer, return_prob = True)
        if "shearedllama" in MODEL: response = clean_response(qns_formatted, response) # Note we do not need to clean this for Flan T5

        # Get self evaluation
        qns_response = SELF_EVALUATE_TEMPLATE(qns, response)
        self_evaluate_response, _ = get_model_response(qns_response, llm, tokenizer)
        if "shearedllama" in MODEL: self_evaluate_response = clean_self_evaluate_response(qns_response, self_evaluate_response) # Note we do not need to clean this for Flan T5

        # Get confidence score
        qns_response_conf = CONFIDENCE_MCQ_TEMPLATE(qns, response)
        pred_conf, _ = get_model_response(qns_response_conf, llm, tokenizer)
        if "shearedllama" in MODEL: pred_conf = clean_confidence_response(qns_response_conf, pred_conf) # Note we do not need to clean this for Flan T5

        # Get self evaluation (with ground truth ans)
        qns_ans = SELF_EVALUATE_TEMPLATE(qns, ans)
        self_evaluate_gt_response, _ = get_model_response(qns_ans, llm, tokenizer)
        if "shearedllama" in MODEL: self_evaluate_gt_response = clean_self_evaluate_response(qns_ans, self_evaluate_gt_response) # Note we do not need to clean this for Flan T5

        # Get confidence score (with ground truth ans)
        qns_ans_conf = CONFIDENCE_MCQ_TEMPLATE(qns, ans)
        pred_conf_gt, _ = get_model_response(qns_ans_conf, llm, tokenizer)
        if "shearedllama" in MODEL: pred_conf_gt = clean_confidence_response(qns_ans_conf, pred_conf_gt) # Note we do not need to clean this for Flan T5

        # Merge all the responses together
        for i in range(len(id_)):

            all_results.append({"id_": id_[i],
                                "original_question": qns[i], 
                                "answer": ans[i], 
                                "predicted_ans" : response[i],
                                "avg_prob": avg_prob[i],
                                "self_evaluate": self_evaluate_response[i],
                                "predicted_conf": pred_conf[i],
                                "self_evaluate_gt": self_evaluate_gt_response[i],
                                "predicted_conf_gt": pred_conf_gt[i]})
                                
    return all_results

if __name__ == "__main__":

    # Settings
    DATASET = "truthfulqa"
    MODEL = "shearedllama-1.3b"
    DATASET_FOLDER = TRUTHFUL_QA_PROCESSED_FOLDER
    OUTPUT_FOLDER = TRUTHFUL_QA_OUTPUT_FOLDER

    # Load the prompt templates 
    BASELINE_TEMPLATE = get_baseline_template(MODEL)
    SELF_EVALUATE_TEMPLATE = get_self_evaluate_template(MODEL)
    CONFIDENCE_MCQ_TEMPLATE = get_confidence_MCQ_template(MODEL)

    # Load the model and dataloader 
    llm, tokenizer = get_model_and_tokenizer(MODEL)

    for split in ["train", "val", "test"]:
        current_output_path = os.path.join(OUTPUT_FOLDER, f"{MODEL}_{split}_confidence.json")
        dataloader = get_dataloader(DATASET, os.path.join(DATASET_FOLDER, f"{split}.json"))
        outputs = get_response(dataloader, llm, tokenizer)
        write_json(outputs, current_output_path)