import os 
from config import * 

import argparse 
from tqdm import tqdm 
from functools import partial

from dataloaders import get_dataloader
from models import get_model_and_tokenizer, get_bert_scorer
from metric import compute_bert_score, get_pairwise_bert_score

from utils import *
from prompt_templates import get_abstain_template, get_get_answer_template, get_self_evaluate_template, \
                             get_confidence_MCQ_template, get_confidence_MCQ_NL_template, get_confidence_OE_template

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

        # 1. Check if we should abstain
        abstain_formatted = ABSTAIN_TEMPLATE(qns)
        abstain = get_model_response(abstain_formatted, llm, tokenizer)
        abstain = get_clean_abstain_fnc(args.model)(abstain_formatted, abstain)

        # 2. Get the answer
        qns_formatted = GET_ANSWER_TEMPLATE(qns)
        pred_ans = get_model_response(qns_formatted, llm, tokenizer)
        pred_ans = get_clean_answer_fnc(args.model)(qns_formatted, pred_ans)

        # 3. Get self evaluation
        self_eval_formatted = SELF_EVALUATE_TEMPLATE(qns, pred_ans)
        self_eval = get_model_response(self_eval_formatted, llm, tokenizer)
        self_eval = get_clean_self_evaluate_fnc(args.model)(self_eval_formatted, self_eval)

        # 4. Get probability of true given answer
        true_prob = get_true_prob_p(self_eval_formatted, llm, tokenizer)

        # 5. Get confidence score (MCQ)
        conf_MCQ_formatted = CONFIDENCE_MCQ_TEMPLATE(qns, pred_ans)
        pred_conf_MCQ = get_model_response(conf_MCQ_formatted, llm, tokenizer)
        pred_conf_MCQ = get_clean_confidence_MCQ_fnc(args.model)(conf_MCQ_formatted, pred_conf_MCQ)

        # 6. Get confidence score (Open-ended)
        conf_OE_formatted = CONFIDENCE_OE_TEMPLATE(qns, pred_ans)
        pred_conf_OE = get_model_response(conf_OE_formatted, llm, tokenizer)
        pred_conf_OE = get_clean_confidence_OE_fnc(args.model)(conf_OE_formatted, pred_conf_OE)

        # 7. Get confidence score (MCQ + NL)
        conf_NL_MCQ_formatted = CONFIDENCE_MCQ_NL_TEMPLATE(qns, pred_ans)
        pred_conf_NL_MCQ = get_model_response(conf_NL_MCQ_formatted, llm, tokenizer)
        pred_conf_NL_MCQ = get_clean_confidence_MCQ_fnc(args.model)(conf_NL_MCQ_formatted, pred_conf_NL_MCQ, NL = True)

        # 8. Get self evaluation (ground truth ans)
        self_eval_GT_formatted = SELF_EVALUATE_TEMPLATE(qns, ans)
        self_eval_GT = get_model_response(self_eval_GT_formatted, llm, tokenizer)
        self_eval_GT = get_clean_self_evaluate_fnc(args.model)(self_eval_GT_formatted, self_eval_GT)

        # 9. Get confidence score (MCQ + ground truth ans)
        conf_GT_MCQ_formatted = CONFIDENCE_MCQ_TEMPLATE(qns, ans)
        pred_conf_GT_MCQ = get_model_response(conf_GT_MCQ_formatted, llm, tokenizer)
        pred_conf_GT_MCQ = get_clean_confidence_MCQ_fnc(args.model)(conf_GT_MCQ_formatted, pred_conf_GT_MCQ)

        # 10. Get confidence score (Open-ended + ground truth ans)
        conf_GT_OE_formatted = CONFIDENCE_OE_TEMPLATE(qns, ans)
        pred_conf_GT_OE = get_model_response(conf_GT_OE_formatted, llm, tokenizer)
        pred_conf_GT_OE = get_clean_confidence_OE_fnc(args.model)(conf_GT_OE_formatted, pred_conf_GT_OE)

        # 11. Get confidence score (MCQ + NL + ground truth ans)
        conf_GT_NL_MCQ_formatted = CONFIDENCE_MCQ_NL_TEMPLATE(qns, ans)
        pred_conf_GT_NL_MCQ = get_model_response(conf_GT_NL_MCQ_formatted, llm, tokenizer)
        pred_conf_GT_NL_MCQ = get_clean_confidence_MCQ_fnc(args.model)(conf_GT_NL_MCQ_formatted, pred_conf_GT_NL_MCQ, NL = True)

        # Merge all the responses together
        for i, idx in enumerate(id_):
            
            res = {"id_": id_[i],
                    "original_question": qns[i], 
                    "diverse_qns" : diverse_qns[i],
                    "answer": ans[i], 
                    "pred_abstain": abstain[i],
                    "pred_ans" : pred_ans[i],
                    "pred_diverse_ans" : pred_diverse_ans[i],
                    "self_eval": self_eval[i],
                    "true_prob": true_prob[i],
                    "pred_conf_MCQ": pred_conf_MCQ[i],
                    "pred_conf_OE": pred_conf_OE[i],
                    "pred_conf_NL_MCQ": pred_conf_NL_MCQ[i],
                    "self_eval_gt": self_eval_GT[i],
                    "pred_conf_gt_MCQ": pred_conf_GT_MCQ[i],
                    "pred_conf_gt_OE": pred_conf_GT_OE[i],
                    "pred_conf_gt_NL_MCQ": pred_conf_GT_NL_MCQ[i]}
            
            write_json(res, os.path.join(args.temp_folder, f"{idx}.json"))

def merge_records(folder, output_file_path):

    all_results = [read_json(os.path.join(folder, f)) for f in os.listdir(folder)]
    pred_ans, ans = [r["pred_ans"] for r in all_results], [r["answer"] for r in all_results]
    pred_ans_bertscore = compute_bert_score(bert_scorer, pred_ans, ans).detach().cpu().numpy()
    for i, r in enumerate(all_results): r["pred_ans_bert_score"] = pred_ans_bertscore[i]

    for r in tqdm(all_results):

        diverse_ques = [r["pred_ans"]] + r["pred_diverse_ans"]
        diverse_ans_bertscore = get_pairwise_bert_score(bert_scorer, diverse_ques)
        r["pred_ans_bert_score"] = diverse_ans_bertscore

    write_json(all_results, output_file_path)

if __name__ == "__main__":

    """
        This set of code consolidates all the results from prompting
    """

    # Parser
    parser = argparse.ArgumentParser(description = "Getting testing results of trained model")

    # Data settings 
    parser.add_argument("--dataset", type = str, default = "triviaqa", help = "The dataset")
    parser.add_argument("--model", type = str, default = "flan-t5-small", help = "LLM to use")

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

    # Define partial functions 
    get_true_prob_p = partial(get_true_prob, true_idx = args.true_idx, false_idx = args.false_idx)

    # # Get the models
    # llm, tokenizer = get_model_and_tokenizer(args.model)
    idx = DEVICE_IDX.split(",")[0]
    device = torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
    bert_scorer = get_bert_scorer(device)

    # Load the prompt templates 
    ABSTAIN_TEMPLATE = get_abstain_template(args.model)
    GET_ANSWER_TEMPLATE = get_get_answer_template(args.model)
    SELF_EVALUATE_TEMPLATE = get_self_evaluate_template(args.model)
    CONFIDENCE_MCQ_TEMPLATE = get_confidence_MCQ_template(args.model)
    CONFIDENCE_MCQ_NL_TEMPLATE = get_confidence_MCQ_NL_template(args.model)
    CONFIDENCE_OE_TEMPLATE = get_confidence_OE_template(args.model)

    # Get the dataloader 
    dataloader = get_dataloader(args.dataset, os.path.join(args.dataset_folder, FILENAME), BATCH_SIZE)

    # Run and get response 
    # get_response(llm, tokenizer, dataloader, args)
    merge_records(args.temp_folder, args.output_file_path)
    os.removedirs(args.temp_folder)