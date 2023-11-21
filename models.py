import bert_score 

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer

from config import * 

def get_bert_scorer():

    return bert_score.BERTScorer(model_type = BERT_SCORER_MODEL)

def get_model_and_tokenizer(chosen_model):

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINTS[chosen_model])

    # Add pad token if it does not already exist 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    if chosen_model in ["flan-t5-small", "flan-t5-base", "flan-t5-large"]:
        llm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINTS[chosen_model])
    elif chosen_model in ["shearedllama-1.3b", "shearedllama-2.7b", "mistral-7b"]:
        llm = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINTS[chosen_model])
    else:
        raise NotImplementedError()

    # Move to GPU 
    llm.to(DEVICE)
    
    return llm, tokenizer

if __name__ == "__main__":

    llm, tokenizer = get_model_and_tokenizer("mistral-7b")
    print("Managed to run successfully")


