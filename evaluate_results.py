
import os
from config import * 
from pprint import pprint
from utils import read_json, write_json

from tqdm import tqdm 

from models import get_bert_scorer
from metric import compute_bert_score, get_pairwise_bert_score, ECE, AUROC, AUC, precision_recall_fscore_support

def get_bert_score(folder):

    all_results = [read_json(os.path.join(folder, f)) for f in os.listdir(folder)]
    pred_ans, ans = [r["pred_ans"] for r in all_results], [r["answer"] for r in all_results]
    # pred_ans_conf_MCQ = [r["pred_ans_conf_MCQ"] for r in all_results]
    # pred_ans_conf_NL_MCQ = [r["pred_ans_conf_NL_MCQ"] for r in all_results]
    # pred_ans_conf_OE = [r["pred_ans_conf_OE"] for r in all_results]

    pred_ans_bertscore = compute_bert_score(bert_scorer, pred_ans, ans)
    pred_ans_conf_MCQ_bertscore = compute_bert_score(bert_scorer, pred_ans_conf_MCQ, ans)
    pred_ans_conf_NL_MCQ_bertscore = compute_bert_score(bert_scorer, pred_ans_conf_NL_MCQ, ans)
    pred_ans_conf_OE_bertscore = compute_bert_score(bert_scorer, pred_ans_conf_OE, ans)

    for i, r in enumerate(all_results): 
        r["pred_ans_bert_score"] = pred_ans_bertscore[i]
        r["pred_ans_conf_MCQ_bert_score"] = pred_ans_conf_MCQ_bertscore[i]
        r["pred_ans_conf_NL_MCQ_bert_score"] = pred_ans_conf_NL_MCQ_bertscore[i]
        r["pred_ans_conf_OE_bert_score"] = pred_ans_conf_OE_bertscore[i]

    for r in tqdm(all_results):

        consistency_score = get_pairwise_bert_score(bert_scorer, r["pred_ans"], r["pred_diverse_ans"])
        consistency_score = np.mean(consistency_score)
        r["consistency_score"] = consistency_score
    
    return all_results

def get_bert_score_flan_t5(folder):

    all_results = [read_json(os.path.join(folder, f)) for f in os.listdir(folder)]
    pred_ans, ans = [r["pred_ans"] for r in all_results], [r["answer"] for r in all_results]
    pred_ans_bertscore = compute_bert_score(bert_scorer, pred_ans, ans)

    for i, r in tqdm(enumerate(all_results)):

        consistency_score = get_pairwise_bert_score(bert_scorer, r["pred_ans"], r["pred_diverse_ans"])
        consistency_score = np.mean(consistency_score)
        r["pred_ans_bert_score"] = pred_ans_bertscore[i]
        r["consistency_score"] = consistency_score
        if r["self_eval"] == "yes": r["self_eval"] = 1
        if r["self_eval_gt"] == "yes": r["self_eval_gt"] = 1
    
    return all_results

def get_results(results):

    n_total = len(results)

    # We get the correct label for each method of getting the answers 
    ans_label = np.array([int(r["pred_ans_bert_score"] >= SCORING_THRESHOLD) for r in results])
    if not CHECK_FLAN_T5:
        ans_conf_MCQ_label = np.array([int(r["pred_ans_conf_MCQ_bert_score"] >= SCORING_THRESHOLD) for r in results])
        ans_conf_NL_MCQ_label = np.array([int(r["pred_ans_conf_NL_MCQ_bert_score"] >= SCORING_THRESHOLD) for r in results])
        ans_conf_OE_label = np.array([int(r["pred_ans_conf_OE_bert_score"] >= SCORING_THRESHOLD) for r in results])

    # We first get the accuracy of each method
    ans_acc = round(sum(ans_label) / n_total * 100, 3)
    ans_conf_MCQ_acc, ans_conf_NL_MCQ_acc, ans_conf_OE_acc = "", "", ""
    if not CHECK_FLAN_T5:
        ans_conf_MCQ_acc = round(sum(ans_conf_MCQ_label) / n_total * 100, 3)
        ans_conf_NL_MCQ_acc = round(sum(ans_conf_NL_MCQ_label) / n_total * 100, 3)
        ans_conf_OE_acc = round(sum(ans_conf_OE_label) / n_total * 100, 3)

    # We now get different calibration score for each method 

    # 1. Probability of true 
    true_prob = np.array([r["true_prob"] for r in results])
    ECE_true_prob = ECE(true_prob, ans_label)
    AUROC_true_prob = AUROC(true_prob, ans_label)
    AUC_true_prob = AUC(true_prob, ans_label)

    # 2. Consistency score 
    conf_cons = np.array([r["consistency_score"] for r in results])
    ECE_conf_cons = ECE(conf_cons, ans_label)
    AUROC_conf_cons = AUROC(conf_cons, ans_label)
    AUC_conf_cons = AUC(conf_cons, ans_label)

    # 3. Confidence (MCQ)
    conf_MCQ = np.array([r["pred_conf_MCQ"] / 100 for r in results])
    ECE_conf_MCQ = ECE(conf_MCQ, ans_label)
    AUROC_conf_MCQ = AUROC(conf_MCQ, ans_label)
    AUC_conf_MCQ = AUC(conf_MCQ, ans_label)

    # 4. Confidence (MCQ + NL)
    conf_NL_MCQ = np.array([r["pred_conf_NL_MCQ"] / 100 for r in results])
    ECE_conf_NL_MCQ = ECE(conf_NL_MCQ, ans_label)
    AUROC_conf_NL_MCQ = AUROC(conf_NL_MCQ, ans_label)
    AUC_conf_NL_MCQ = AUC(conf_NL_MCQ, ans_label)

    # 5. Confidence (OE)
    conf_OE = np.array([r["pred_conf_OE"] / 100 for r in results])
    ECE_conf_OE = ECE(conf_OE, ans_label)
    AUROC_conf_OE = AUROC(conf_OE, ans_label)
    AUC_conf_OE = AUC(conf_OE, ans_label)

    ECE_conf_MCQ_1s, AUROC_conf_MCQ_1s, AUC_conf_MCQ_1s = "", "", ""
    ECE_conf_NL_MCQ_1s, AUROC_conf_NL_MCQ_1s, AUC_conf_NL_MCQ_1s = "", "", ""
    ECE_conf_OE_1s, AUROC_conf_OE_1s, AUC_conf_OE_1s = "", "", ""

    if not CHECK_FLAN_T5:
        # 6. Confidence (MCQ, 1S)
        conf_MCQ_1s = np.array([r["pred_conf_MCQ_1s"] / 100 for r in results])
        ECE_conf_MCQ_1s = ECE(conf_MCQ_1s, ans_conf_MCQ_label)
        AUROC_conf_MCQ_1s = AUROC(conf_MCQ_1s, ans_conf_MCQ_label)
        AUC_conf_MCQ_1s = AUC(conf_MCQ_1s, ans_conf_MCQ_label)

        # 7. Confidence (MCQ + NL, 1S)
        conf_NL_MCQ_1s = np.array([r["pred_conf_NL_MCQ_1s"] / 100 for r in results])
        ECE_conf_NL_MCQ_1s = ECE(conf_NL_MCQ_1s, ans_conf_NL_MCQ_label)
        AUROC_conf_NL_MCQ_1s = AUROC(conf_NL_MCQ_1s, ans_conf_NL_MCQ_label)
        AUC_conf_NL_MCQ_1s = AUC(conf_NL_MCQ_1s, ans_conf_NL_MCQ_label)

        # 8. Confidence (OE, 1S)
        conf_OE_1s = np.array([r["pred_conf_OE_1s"] / 100 for r in results])
        ECE_conf_OE_1s = ECE(conf_OE_1s, ans_conf_OE_label)
        AUROC_conf_OE_1s = AUROC(conf_OE_1s, ans_conf_OE_label)
        AUC_conf_OE_1s = AUC(conf_OE_1s, ans_conf_OE_label)

    # Get precision, recall and F-score of various aspects 
    # 1. Check how well can LLMs evaluate answers 
    self_eval = np.array([r["self_eval"] for r in results])
    P_self_eval, R_self_eval, F1_self_eval, _ = precision_recall_fscore_support(ans_label, self_eval, average = "macro")

    # 2. Check how well can LLMs decide when to answer and when not to 
    abstain = np.array([1 - r["pred_abstain"] for r in results])
    P_abstain, R_abstain, F1_abstain, _ = precision_recall_fscore_support(ans_label, abstain, average = "macro")

    # Format the scores 
    scores = {"Accuracy_ans": ans_acc,
              "Accuracy_ans_conf_MCQ": ans_conf_MCQ_acc,
              "Accuracy_ans_conf_NL_MCQ": ans_conf_NL_MCQ_acc,
              "Accuracy_ans_conf_OE": ans_conf_OE_acc,
              "ECE_true_prob": ECE_true_prob, 
              "AUROC_true_prob": AUROC_true_prob,
              "AUC_true_prob": AUC_true_prob,
              "ECE_conf_cons": ECE_conf_cons, 
              "AUROC_conf_cons": AUROC_conf_cons,
              "AUC_conf_cons": AUC_conf_cons,
              "ECE_conf_MCQ": ECE_conf_MCQ, 
              "AUROC_conf_MCQ": AUROC_conf_MCQ,
              "AUC_conf_MCQ": AUC_conf_MCQ,
              "ECE_conf_OE": ECE_conf_OE, 
              "AUROC_conf_OE": AUROC_conf_OE,
              "AUC_conf_OE": AUC_conf_OE,
              "ECE_conf_NL_MCQ": ECE_conf_NL_MCQ, 
              "AUROC_conf_NL_MCQ": AUROC_conf_NL_MCQ,
              "AUC_conf_NL_MCQ": AUC_conf_NL_MCQ,

              "ECE_conf_MCQ_1s": ECE_conf_MCQ_1s, 
              "AUROC_conf_MCQ_1s": AUROC_conf_MCQ_1s,
              "AUC_conf_MCQ_1s": AUC_conf_MCQ_1s,
              "ECE_conf_OE_1s": ECE_conf_OE_1s, 
              "AUROC_conf_OE_1s": AUROC_conf_OE_1s,
              "AUC_conf_OE_1s": AUC_conf_OE_1s,
              "ECE_conf_NL_MCQ_1s": ECE_conf_NL_MCQ_1s, 
              "AUROC_conf_NL_MCQ_1s": AUROC_conf_NL_MCQ_1s,
              "AUC_conf_NL_MCQ_1s": AUC_conf_NL_MCQ_1s,

              "P_R_F1_self_eval": [P_self_eval, R_self_eval, F1_self_eval],
              "P_R_F1_abstain": [P_abstain, R_abstain, F1_abstain]}

    return scores

if __name__ == "__main__":

    """
        This set of code gather all the results obtained
    """

    # Settings
    DATASET = "sciq"
    LLM = "flan-t5-xl"
    CHECK_FLAN_T5 = "flan-t5" in LLM

    idx = 0
    device = torch.device(f"cuda:{idx}") if torch.cuda.is_available() else torch.device("cpu")
    bert_scorer = get_bert_scorer(device)

    # # Get the folder 
    RECORDS_FOLDER = os.path.join(OUTPUT_FOLDER, DATASET, f"{LLM}_temp_DONE")
    OUTPUT_FILE_PATH = os.path.join(RESULTS_FOLDER, f"{DATASET}_{LLM}_cleaned.json")
    RESULTS_OUTPUT_FILE_PATH = os.path.join(RESULTS_FOLDER, f"{DATASET}_{LLM}_results.json")
    # all_results = read_json(OUTPUT_FILE_PATH)

    # Get the bert score
    if CHECK_FLAN_T5:
        all_results = get_bert_score_flan_t5(RECORDS_FOLDER.replace("_DONE", ""))
    else:
        all_results = get_bert_score(RECORDS_FOLDER)
    write_json(all_results, OUTPUT_FILE_PATH)

    # Get the evaluations now 
    scores = get_results(all_results)
    write_json(scores, RESULTS_OUTPUT_FILE_PATH)
