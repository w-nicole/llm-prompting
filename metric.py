import numpy as np 

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