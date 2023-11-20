import os 
from tqdm import tqdm 

from dataloaders import get_dataloader
from models import get_model_and_tokenizer

from config import * 

def get_perplexity(dataloader):

    for batch in tqdm(dataloader):
        print(batch)    
        print("here")
        a = z 
    
    None

if __name__ == "__main__":

    # Load the model and dataloader 
    llm, tokenizer = get_model_and_tokenizer()
    dataloader = get_dataloader("truthfulqa", os.path.join(TRUTHFUL_QA_PROCESSED_PATH, "train.json"))
    perplexity_score = get_perplexity(dataloader)
    
    

    print("okay i am here")
    