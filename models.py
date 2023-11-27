import bert_score 

import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from config import * 

def clean_model_keys(model_weights):

    cleaned_model_weights = {} 
    for k,v in model_weights.items():

        k = k.replace("decoder.transformer", "")
        cleaned_model_weights[k] = v 
    
    return cleaned_model_weights

def get_bert_scorer():

    return bert_score.BERTScorer(model_type = BERT_SCORER_MODEL, lang = "en", rescale_with_baseline = True)

def get_model_and_tokenizer(chosen_model):

    if chosen_model in ["llama2-7b"]:
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_CHECKPOINTS[chosen_model])
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINTS[chosen_model])

    # Add pad token if it does not already exist 
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
        
    if chosen_model in ["flan-t5-small", "flan-t5-base", "flan-t5-large", "flan-t5-xl"]:
        llm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINTS[chosen_model])
    elif chosen_model in ["shearedllama-bling-1.3b", "shearedllama-bling-2.7b", "shearedllama-1.3b", "shearedllama-2.7b", "llama2-7b", "mistral-7b"]:
        llm = AutoModelForCausalLM.from_pretrained(MODEL_CHECKPOINTS[chosen_model])
    
    elif chosen_model in ["llama2-7b"]:
        llm = LlamaForCausalLM.from_pretrained(MODEL_CHECKPOINTS[chosen_model])
    else:
        raise NotImplementedError()

    # Move to GPU 
    llm.to(DEVICE)
    
    return llm, tokenizer

if __name__ == "__main__":

    llm, tokenizer = get_model_and_tokenizer("llama2-7b")
    print("Managed to run successfully")


