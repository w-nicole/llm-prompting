import os
from config import * 

import argparse 
from tqdm import tqdm 
from functools import partial

from dataloaders import get_dataloader
from models import get_model_and_tokenizer

from utils import *

from prompt_templates import get_abstain_template, get_get_answer_template, get_self_evaluate_template, \
                             get_confidence_MCQ_template, get_confidence_MCQ_NL_template, get_confidence_OE_template, \
                             get_get_answer_and_confidence_MCQ_template, get_get_answer_and_confidence_MCQ_NL_template, get_get_answer_and_confidence_OE_template

@torch.no_grad()
def get_response(llm, tokenizer, dataloader, args):

    for batch in tqdm(dataloader):

        # Unpack the batch 
        id_, qns, ans, diverse_qns = batch

        # Merge the outputs of the diverse questions
        idx_list, diverse_qns = unpack_qns(diverse_qns)

        # Get responses from all the answers
        diverse_qns_formatted = GET_ANSWER_TEMPLATE(diverse_qns)
        pred_diverse_ans = get_model_response(diverse_qns_formatted, llm, tokenizer)
        pred_diverse_ans = get_clean_answer_fnc(args.model)(diverse_qns_formatted, pred_diverse_ans)
        diverse_qns, pred_diverse_ans = pack_qns_ans(idx_list, diverse_qns, pred_diverse_ans)

        print("okay i am here")

        # 3. Get self evaluation
        self_eval_formatted = SELF_EVALUATE_TEMPLATE(qns, pred_ans)
        self_eval = get_model_response(self_eval_formatted, llm, tokenizer)
        self_eval = get_clean_self_evaluate_fnc(args.model)(self_eval_formatted, self_eval)






if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description = "Getting results for model by showing a list of wrong answers")

    # Data settings 
    parser.add_argument("--dataset", type = str, default = "truthfulqa", help = "The dataset")
    parser.add_argument("--model", type = str, default = "llama2-7b-chat", help = "LLM to use")

    # Set defaults 
    args = parser.parse_args()

    # Settings
    _, dataset_folder, output_folder = get_folders(args.dataset)
    args.dataset_folder = dataset_folder
    args.output_folder = output_folder
    args.output_file_path = os.path.join(args.output_folder, f"{args.model}.json")
    args.temp_folder = os.path.join(output_folder, f"{args.model}_temp")
    if not os.path.exists(args.temp_folder):
        os.makedirs(args.temp_folder)
    true_idx, false_idx = TRUE_FALSE_IDX[args.model]["true"], TRUE_FALSE_IDX[args.model]["false"]
    args.true_idx = true_idx 
    args.false_idx = false_idx
    args.llama2_check = "llama2" in args.model

    # Define partial functions 
    get_true_prob_p = partial(get_true_prob, true_idx = args.true_idx, false_idx = args.false_idx)

    # # Get the models
    llm, tokenizer = get_model_and_tokenizer(args.model)

    # Load the prompt templates 
    ABSTAIN_TEMPLATE = get_abstain_template(args.model)
    GET_ANSWER_TEMPLATE = get_get_answer_template(args.model)
    SELF_EVALUATE_TEMPLATE = get_self_evaluate_template(args.model)
    CONFIDENCE_MCQ_TEMPLATE = get_confidence_MCQ_template(args.model)
    CONFIDENCE_MCQ_NL_TEMPLATE = get_confidence_MCQ_NL_template(args.model)
    CONFIDENCE_OE_TEMPLATE = get_confidence_OE_template(args.model)
    GET_ANSWER_CONFIDENCE_MCQ_TEMPLATE = get_get_answer_and_confidence_MCQ_template(args.model)
    GET_ANSWER_CONFIDENCE_MCQ_NL_TEMPLATE = get_get_answer_and_confidence_MCQ_NL_template(args.model)
    GET_ANSWER_CONFIDENCE_OE_TEMPLATE = get_get_answer_and_confidence_OE_template(args.model)

    # Get the dataloader 
    dataloader = get_dataloader(args.dataset, os.path.join(args.dataset_folder, FILENAME), BATCH_SIZE)

    # Run and get response 
    get_response(llm, tokenizer, dataloader, args)
