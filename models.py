
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
    
def get_model_and_tokenizer(chosen_model):
    
    model_checkpoints = {
        "flan-t5" : "google/flan-t5-small",
        "lamma2": "",
        "vicuna": "lmsys/vicuna-13b-v1.1",
        "alpaca": "chavinlo/alpaca-native"
    }
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints[chosen_model])
    
    if chosen_model in ["flan-t5"]:
        llm = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoints[chosen_model])
    elif chosen_model in ["vicuna", "alpaca"]:
        llm = AutoModelForCausalLM.from_pretrained(model_checkpoints[chosen_model])
    else:
        raise notImplementedError() 
        
    return llm, tokenizer

