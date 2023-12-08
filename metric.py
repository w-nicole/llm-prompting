import numpy as np 
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_fscore_support

# For BERT SCORER
def compute_bert_score(scorer, cands: list[str], refs: list[str]):

    # Let's standardize the casing 
    cands = [str(s).lower() for s in cands]
    refs = [str(s).lower() for s in refs]
    
    _, _, F1 = scorer.score(cands, refs)
    F1 = F1.detach().cpu().numpy().tolist()

    # To make sure that it's between 0.0 and 1.0
    F1 = [max(a, 0.0) for a in F1]
    F1 = [min(a, 1.0) for a in F1]
    
    return F1

def get_pairwise_bert_score(scorer, ans, diverse_ans_list):

    ref = [ans for _ in range(len(diverse_ans_list))]
    pred = diverse_ans_list

    F1 = compute_bert_score(scorer, pred, ref)

    return F1

# Getting the ECE score
def ECE(conf, accuracy, n_bins = 10):
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):

        # Determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(conf > bin_lower.item(), conf <= bin_upper.item())
        
        prob_in_bin = in_bin.mean() # Ie, the proportion of records in this bin
        if prob_in_bin.item() > 0:

            accuracy_in_bin = accuracy[in_bin].mean()
            avg_confidence_in_bin = conf[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

    return round(ece[0], 3)

# Getting the AUROC
def AUROC(conf, accuracy):

    fpr, tpr, thresholds = metrics.roc_curve(accuracy, conf, pos_label = 1)
    score = metrics.auc(fpr, tpr)

    return round(score, 3)

# Getting the AUC 
def AUC(conf, accuracy, n_bins = 10):

    sort_order = [b[0] for b in sorted(enumerate(conf), key = lambda i:i[1], reverse = True)]
    conf = [conf[i] for i in sort_order]
    accuracy = [accuracy[i] for i in sort_order]

    A = []
    for c in np.linspace(0, 1, n_bins):
        sample = conf[:int(c*len(conf))]
        if not sample:
            A.append(0)
            continue
        score = accuracy[:int(c*len(conf))]
        total = int(c*len(conf))
        score = sum(score) / total
        A.append(score)

    return round(auc(np.linspace(0, 1, n_bins), A), 3)
