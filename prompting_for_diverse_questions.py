import re
from tqdm import tqdm 
from openai import OpenAI

from dataloaders import get_dataloader

from config import * 
from utils import write_json
from prompt_templates import diverse_ques_gpt4_template

def get_gpt4_diverse_outputs(dataloader):
    
    all_results = []

    for batch in tqdm(dataloader):

        # Unpack the batch 
        id_, qns, ans = batch 

        # Format the question
        diverse_qns = diverse_ques_gpt4_template(qns)

        # Get response 
        response = CLIENT.chat.completions.create(model = "gpt-4",
                                                  messages = diverse_qns)
        
        # Format the response 
        response = response.model_dump()["choices"][0]["message"]["content"].split("\n")
        response = [re.sub("\d+.", "", r).strip() for r in response]
        
        # Add in to all the results 
        id_, qns, ans = id_[0], qns[0], ans[0] # Hack because we know that the batch size is always 0
        all_results.append({"id_": id_,
                            "original_question": qns, 
                            "answer": ans, 
                            "GPT4_diverse_question" : response})

    return all_results 

if __name__ == "__main__":

    # Key motivation: diversity can be achieved 2 ways: we can either get diverse prompts / get GPT-4 to rephrase the question separately
    # This code gets diversity from GPT-4 

    # Getting the client
    DATASET = "sciq"
    DATASET_FOLDER = SCIQ_PROCESSED_FOLDER
    KEY = 
    CLIENT = OpenAI(api_key = KEY)

    # Getting the results
    for split in ["train", "val", "test"]:
        dataloader = get_dataloader(DATASET, os.path.join(DATASET_FOLDER, f"{split}.json"), batch_size = 1) # Seems like the API only supports batch size of 1
        diverse_outputs = get_gpt4_diverse_outputs(dataloader)
        write_json(diverse_outputs, os.path.join(DATASET_FOLDER, f"{split}_w_diverse_qns_GPT4.json"))
