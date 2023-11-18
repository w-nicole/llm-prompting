import bert_score 

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, AutoModel

from config import * 

def get_bert_scorer():

    return bert_score.BERTScorer(model_type = BERT_SCORER_MODEL)


def get_model_and_tokenizer(chosen_model):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINTS[chosen_model])
    
    if chosen_model in ["flan-t5-small"]:
        llm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINTS[chosen_model])
    elif chosen_model in ["vicuna-7b", "openllama-3b-v2", "alpaca"]:
        llm = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINTS[chosen_model])
    else:
        raise NotImplementedError() 
        
    return llm, tokenizer

if __name__ == "__main__":

    llm, tokenizer = get_model_and_tokenizer("vicuna-7b")
    get_bert_scorer()
    print("okay i am here")


