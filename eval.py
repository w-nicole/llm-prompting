import os, argparse
from config import * 
from pprint import pprint
from utils import read_json, write_json, get_samples

from tqdm import tqdm 

from models import get_bert_scorer
from metric import compute_bert_score, get_pairwise_bert_score, ECE, AUROC, AUC, precision_recall_fscore_support


def get_results(results, ds):
    ans_label = np.array([int(r["pred_ans_bert_score"] >= SCORING_THRESHOLD) for r in results])

    true_prob = np.array([1-r["hallucination_score_paraphrases"] for r in results])
    E = ECE(true_prob, ans_label)
    AURO = AUROC(true_prob, ans_label)
    AU  = AUC(true_prob, ans_label)
    
    print(ds)
    print(f"Paraphrases: ECE: {E} AUC:{AU} AUROC:{AURO}")

    true_prob = np.array([1-r["hallucination_score_samples"] for r in results])
    E = ECE(true_prob, ans_label)
    AURO = AUROC(true_prob, ans_label)
    AU  = AUC(true_prob, ans_label)

    print(f"Samples: ECE: {E} AUC:{AU} AUROC:{AURO}\n")


for ds in ['triviaqa', 'truthfulqa', 'sciq']:
    results = read_json(f"hallucination_result/{ds}_mistral-7b-instruct_cleaned.json")
    get_results(results, ds)

