import evaluate, os, tqdm, string, re
from evaluate import load

from utils import *
from tqdm import tqdm

def remove_punc_and_lower(str):
    return re.sub('['+string.punctuation+']', '', str.lower())


def exact_match_score(original_answer, answer):
    exact_match = evaluate.load("exact_match")

    # case and punctuation ignored  
    results = exact_match.compute(references=[original_answer], predictions=[answer], ignore_case=True, ignore_punctuation=True) 
    return results['exact_match']


def word_level_f1_score(original_answer, answer):
    def f1_score(precision, recall):
        if precision + recall == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    pred_words = set(remove_punc_and_lower(answer).split())
    ref_words = set(remove_punc_and_lower(original_answer).split())

    true_positives = len(pred_words.intersection(ref_words))
    false_positives = len(pred_words - ref_words)
    false_negatives = len(ref_words - pred_words)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = f1_score(precision, recall)

    return f1

if __name__ == "__main__":

    for file in os.listdir("results/"):
        if "sciq" not in file or "cleaned" not in file: continue

        print("Starting ", file)
        data = read_json("results/"+file)

        for item in tqdm(data):
            original_answer = item['answer']
            exact_match_scores = []
            f1_scores = []
            for answer in item["pred_diverse_ans"]:
                exact_match_scores.append(exact_match_score(original_answer, answer))
                f1_scores.append(word_level_f1_score(original_answer, answer))

            item['exact_match'] = exact_match_scores
            item['word level f1 score'] = f1_scores

        write_json(data, f"new_results/{file}")