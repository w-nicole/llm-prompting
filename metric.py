import numpy as np 
from sklearn import metrics
from sklearn.metrics import auc, precision_recall_fscore_support

# For BERT SCORER
def compute_bert_score(scorer, cands: list[str], refs: list[str], threshold = 0.8):
    P, R, F1 = scorer.score(cands, refs)
    return [1 if score >= threshold else 0 for score in F1]

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

    return ece[0]

# Getting the AUROC
def AUROC(conf, accuracy):

    fpr, tpr, thresholds = metrics.roc_curve(accuracy, conf, pos_label = 1)
    score = metrics.auc(fpr, tpr)

    return score

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

    return auc(np.linspace(0, 1, n_bins), A)
