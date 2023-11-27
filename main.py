import os 
from tqdm import tqdm 
from functools import partial

from dataloaders import get_dataloader
from models import get_model_and_tokenizer, get_bert_scorer

from utils import *
from config import * 
from metric import compute_bert_score
from prompt_templates import get_abstain_template, get_get_answer_template, get_self_evaluate_template, \
                             get_confidence_MCQ_template, get_confidence_MCQ_NL_template, get_confidence_OE_template

def get_response(dataloader, llm, tokenizer, bert_scorer):

    all_results = []

    for batch in tqdm(dataloader):

        # Unpack the batch 
        id_, qns, ans = batch 
        ans = list(ans)

        # # 1. Check if we should abstain
        abstain_formatted = ABSTAIN_TEMPLATE(qns)
        abstain = get_model_response(abstain_formatted, llm, tokenizer)
        abstain = get_clean_abstain_fnc(MODEL)(abstain_formatted, abstain)

        # 2. Get the answer
        qns_formatted = GET_ANSWER_TEMPLATE(qns)
        pred_ans = get_model_response(qns_formatted, llm, tokenizer)
        pred_ans = get_clean_answer_fnc(MODEL)(qns_formatted, pred_ans)
        pred_correct = compute_bert_score(bert_scorer, pred_ans, ans, threshold = SCORING_THRESHOLD) # Evaluate if answer is correct

        # 3. Get self evaluation
        self_eval_formatted = SELF_EVALUATE_TEMPLATE(qns, pred_ans)
        self_eval = get_model_response(self_eval_formatted, llm, tokenizer)
        self_eval = get_clean_self_evaluate_fnc(MODEL)(self_eval_formatted, self_eval)

        # 4. Get probability of true given answer
        true_prob = get_true_prob_p(self_eval_formatted, llm, tokenizer)

        # 5. Get confidence score (MCQ)
        conf_MCQ_formatted = CONFIDENCE_MCQ_TEMPLATE(qns, pred_ans)
        pred_conf_MCQ = get_model_response(conf_MCQ_formatted, llm, tokenizer)
        pred_conf_MCQ = get_clean_confidence_MCQ_fnc(MODEL)(conf_MCQ_formatted, pred_conf_MCQ)

        # 6. Get confidence score (Open-ended)
        conf_OE_formatted = CONFIDENCE_OE_TEMPLATE(qns, pred_ans)
        pred_conf_OE, _ = get_model_response(conf_OE_formatted, llm, tokenizer)
        pred_conf_OE = get_clean_confidence_OE_fnc(MODEL)(conf_OE_formatted, pred_conf_OE)

        # 7. Get confidence score (MCQ + NL)
        conf_NL_MCQ_formatted = CONFIDENCE_MCQ_NL_TEMPLATE(qns, pred_ans)
        pred_conf_NL_MCQ = get_model_response(conf_NL_MCQ_formatted, llm, tokenizer)
        pred_conf_NL_MCQ = get_clean_confidence_MCQ_fnc(MODEL)(conf_NL_MCQ_formatted, pred_conf_NL_MCQ, NL = True)

        # 8. Get self evaluation (ground truth ans)
        self_eval_GT_formatted = SELF_EVALUATE_TEMPLATE(qns, ans)
        self_eval_GT = get_model_response(self_eval_GT_formatted, llm, tokenizer)
        self_eval_GT = get_clean_self_evaluate_fnc(MODEL)(self_eval_GT_formatted, self_eval_GT)

        # 9. Get confidence score (MCQ + ground truth ans)
        conf_GT_MCQ_formatted = CONFIDENCE_MCQ_TEMPLATE(qns, ans)
        pred_conf_GT_MCQ = get_model_response(conf_GT_MCQ_formatted, llm, tokenizer)
        pred_conf_GT_MCQ = get_clean_confidence_MCQ_fnc(MODEL)(conf_GT_MCQ_formatted, pred_conf_GT_MCQ)

        # 10. Get confidence score (Open-ended + ground truth ans)
        conf_GT_OE_formatted = CONFIDENCE_OE_TEMPLATE(qns, ans)
        pred_conf_GT_OE = get_model_response(conf_GT_OE_formatted, llm, tokenizer)
        pred_conf_GT_OE = get_clean_confidence_OE_fnc(MODEL)(conf_GT_OE_formatted, pred_conf_GT_OE)

        # 11. Get confidence score (MCQ + NL + ground truth ans)
        conf_GT_NL_MCQ_formatted = CONFIDENCE_MCQ_NL_TEMPLATE(qns, ans)
        pred_conf_GT_NL_MCQ = get_model_response(conf_GT_NL_MCQ_formatted, llm, tokenizer)
        pred_conf_GT_NL_MCQ = get_clean_confidence_MCQ_fnc(MODEL)(conf_GT_NL_MCQ_formatted, pred_conf_GT_NL_MCQ, NL = True)

        # Merge all the responses together
        for i in range(len(id_)):

            all_results.append({"id_": id_[i],
                                "original_question": qns[i], 
                                "answer": ans[i], 
                                "pred_ans" : pred_ans[i],
                                "pred_correct": pred_correct[i],
                                "self_eval": self_eval[i],
                                "true_prob": true_prob[i],
                                "pred_conf_MCQ": pred_conf_MCQ[i],
                                "pred_conf_OE": pred_conf_OE[i],
                                "pred_conf_NL_MCQ": pred_conf_NL_MCQ[i],
                                "self_eval_gt": self_eval_GT[i],
                                "pred_conf_gt_MCQ": pred_conf_GT_MCQ[i],
                                "pred_conf_gt_OE": pred_conf_GT_OE[i],
                                "pred_conf_gt_NL_MCQ": pred_conf_GT_NL_MCQ[i]})

    return all_results

if __name__ == "__main__":

    """
        This set of code consolidates all the results from prompting
    """

    # Settings
    DATASET = "truthfulqa"
    MODEL = 'flan-t5-small'
    _, DATASET_FOLDER, OUTPUT_FOLDER = get_folders(DATASET)
    LLAMA2_CHECK = MODEL == 'llama2-7b'

    # Get the true false index 
    TRUE_IDX, FALSE_IDX = TRUE_FALSE_IDX[MODEL]["true"], TRUE_FALSE_IDX[MODEL]["false"]

    # Load the prompt templates 
    ABSTAIN_TEMPLATE = get_abstain_template(MODEL)
    GET_ANSWER_TEMPLATE = get_get_answer_template(MODEL)
    SELF_EVALUATE_TEMPLATE = get_self_evaluate_template(MODEL)
    CONFIDENCE_MCQ_TEMPLATE = get_confidence_MCQ_template(MODEL)
    CONFIDENCE_MCQ_NL_TEMPLATE = get_confidence_MCQ_NL_template(MODEL)
    CONFIDENCE_OE_TEMPLATE = get_confidence_OE_template(MODEL)

    # Define partial functions 
    get_true_prob_p = partial(get_true_prob, true_idx = TRUE_IDX, false_idx = FALSE_IDX, llama2 = LLAMA2_CHECK)

    # Load the model and dataloader 
    bert_scorer = get_bert_scorer()
    llm, tokenizer = get_model_and_tokenizer(MODEL)

    for split in ["train", "val", "test"]:
        current_output_path = os.path.join(OUTPUT_FOLDER, f"{MODEL}_{split}_confidence.json")
        dataloader = get_dataloader(DATASET, os.path.join(DATASET_FOLDER, f"{split}.json"))
        outputs = get_response(dataloader, llm, tokenizer, bert_scorer)
        write_json(outputs, current_output_path)