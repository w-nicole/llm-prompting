import os 
from utils import * 
from metric import * 

def get_results(results):

    # Note we loop through different types of confidence scores 
    correct = np.array([r["pred_correct"] for r in results])
    acc = sum(correct) / len(correct)

    # 1. Probability of true 
    true_prob = np.array([r["true_prob"] for r in results])
    ECE_true_prob = ECE(true_prob, correct)
    AUROC_true_prob = AUROC(true_prob, correct)
    AUC_true_prob = AUC(true_prob, correct)

    # 2. Consistency score 
    conf_cons = np.array([r["pred_conf_cons"] for r in results])
    ECE_conf_cons = ECE(conf_cons, correct)
    AUROC_conf_cons = AUROC(conf_cons, correct)
    AUC_conf_cons = AUC(conf_cons, correct)

    # 3. Confidence (MCQ)
    conf_MCQ = np.array([r["pred_conf_MCQ"] / 100 for r in results])
    ECE_conf_MCQ = ECE(conf_MCQ, correct)
    AUROC_conf_MCQ = AUROC(conf_MCQ, correct)
    AUC_conf_MCQ = AUC(conf_MCQ, correct)

    # 4. Confidence (OE)
    conf_OE = np.array([r["pred_conf_OE"] / 100 for r in results])
    ECE_conf_OE = ECE(conf_OE, correct)
    AUROC_conf_OE = AUROC(conf_OE, correct)
    AUC_conf_OE = AUC(conf_OE, correct)

    # 5. Confidence (MCQ + NL)
    conf_NL_MCQ = np.array([r["pred_conf_NL_MCQ"] / 100 for r in results])
    ECE_conf_NL_MCQ = ECE(conf_NL_MCQ, correct)
    AUROC_conf_NL_MCQ = AUROC(conf_NL_MCQ, correct)
    AUC_conf_NL_MCQ = AUC(conf_NL_MCQ, correct)

    # Get precision, recall and F-score of various aspects 
    # 1. Check how well can LLMs evaluate answers 
    self_eval = np.array([r["self_eval"] for r in results])
    P_self_eval, R_self_eval, F1_self_eval, _ = precision_recall_fscore_support(correct, self_eval, average = "macro")

    # 2. Check how well can LLMs decide when to answer and when not to 
    abstain = np.array([1 - r["pred_abstain"] for r in results])
    P_abstain, R_abstain, F1_abstain, _ = precision_recall_fscore_support(correct, abstain, average = "macro")

    # Format the scores 
    scores = {"Accuracy": acc,
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
              "P_R_F1_self_eval": [P_self_eval, R_self_eval, F1_self_eval],
              "P_R_F1_abstain": [P_abstain, R_abstain, F1_abstain]}
    
    return scores

if __name__ == "__main__":

    """
        This set of code gather all the results obtained
    """

    # Settings
    RESULTS_FOLDER = "outputs"
    
    # Master dictionary to save the results 
    all_results = {d: {} for d in os.listdir(RESULTS_FOLDER)}

    for dataset in os.listdir(RESULTS_FOLDER):

        current_folder = os.path.join(RESULTS_FOLDER, dataset)
        files = [f for f in os.listdir(current_folder)]
        if len(files) == 0: continue 
        else:
            for f in files:

                current_path = os.path.join(current_folder, f)
                results = read_json(current_path)
                all_scores = get_results(results)
                model = f.split("_")[0]
                all_results[dataset][model] = all_scores
                print(all_scores)

        print(all_results)
        a = z
