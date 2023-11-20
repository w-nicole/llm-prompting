import re
from tqdm import tqdm 
from openai import OpenAI

from dataloaders import get_dataloader

from config import * 
from prompt_templates import diverse_ques_gpt4_template

def get_gpt4_diverse_outputs(dataloader):

    all_results = {} 

    for batch in tqdm(dataloader):

        # Unpack the batch 
        id_, qns, ans = batch 

        # Format the question
        diverse_qns = diverse_ques_gpt4_template(qns)

        # Get response 
        response = client.chat.completions.create(model = "gpt-4",
                                                  messages = diverse_qns)
        
        # Format the response 
        response = response.model_dump()["choices"][0]["message"]["content"].split("\n")
        response = [re.sub("\d+.", "", r).strip() for r in response]

        # Add in to all the results 
        id_, qns, ans = id_.item(), qns[0], ans[0] # Hack because we know that the batch size is always 0
        all_results["id_"] = {"original_question": qns, 
                              "answer": ans, 
                              "GPT4_diverse_question" : response}
    
    return all_results 

if __name__ == "__main__":

    # Key motivation: diversity can be achieved 2 ways: we can either get diverse prompts / get GPT-4 to rephrase the question separately
    # This code gets diversity from GPT-4 

    key = ""
    client = OpenAI(api_key = key)
    dataloader = get_dataloader("truthfulqa", os.path.join(TRUTHFUL_QA_PROCESSED_PATH, "train.json"), batch_size = 1) # Seems like the API only supports batch size of 1
    diverse_outputs = get_gpt4_diverse_outputs(dataloader)
