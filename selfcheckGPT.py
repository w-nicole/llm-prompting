import numpy as np
import spacy, torch, argparse
from tqdm import tqdm

from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
from utils import *


torch.manual_seed(28)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Getting testing results of trained model")

    # Data settings 
    parser.add_argument("--dataset", type = str, default = "triviaqa", help = "The dataset")
    parser.add_argument("--model", type = str, default = "mistral-7b-instruct", help = "LLM to use")

    # Set defaults 
    args = parser.parse_args()

    ds = read_json(f"sample_result/{args.dataset}_{args.model}_cleaned.json")

    selfcheck_prompt = SelfCheckAPIPrompt(model='gpt-4o')
    nlp = spacy.load("en_core_web_sm")

    results = []
    for item in tqdm(ds):   
        div_ans = item['pred_diverse_ans']

        Response = div_ans[0]
        Samples = div_ans[1:]

        sentences = [sent.text.strip() for sent in nlp(Response).sents]  # spacy sentence tokenization
        sent_scores_prompt = selfcheck_prompt.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=Samples
        )
        item['hallucination_score_paraphrases'] = np.mean(sent_scores_prompt)


        Response = item['reference_answer']
        Samples = item['sample_answer']

        sentences = [sent.text.strip() for sent in nlp(Response).sents]  # spacy sentence tokenization
        sent_scores_prompt = selfcheck_prompt.predict(
            sentences=sentences,  # list of sentences
            sampled_passages=Samples
        )

        item['hallucination_score_samples'] = np.mean(sent_scores_prompt)
        results.append(item.copy())
    
    write_json(results, f"hallucination_result/{args.dataset}_{args.model}_cleaned.json")
        
