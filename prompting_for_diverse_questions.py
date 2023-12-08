import re
import traceback
from tqdm import tqdm 
from openai import OpenAI

from dataloaders import get_dataloader

from config import * 
from utils import write_json
from prompt_templates import diverse_ques_GPT4_template
from cache import is_response_cached, cache_response, multithread_with_tqdm

stop = False

def get_response_from_gpt35(batch, pbar):
    global stop
    # Unpack the batch
    id_, qns, ans = batch
    cache_id = id_[0]
    if stop:
        return
    if is_response_cached(cache_id):
        pbar.update(1)
        return

    # Format the question
    diverse_qns = diverse_ques_GPT4_template(qns)

    # Get response
    try:
        response = CLIENT.chat.completions.create(model="gpt-3.5-turbo", messages=diverse_qns)
    except Exception as e:
        stop = True
        print(e)
        print(traceback.format_exc())
        raise e

    # Get the response
    response = response.model_dump()["choices"][0]["message"]["content"]  #.split("\n")
    # response = [re.sub("\d+.", "", r).strip() for r in response]

    # Cache response
    cache_response(cache_id, response)

    # Update tqdm
    pbar.update(1)

def get_gpt35_diverse_outputs(dataloader):
    multithread_with_tqdm(
        get_response_from_gpt35, len(dataloader.dataset), dataloader, 20
    )

if __name__ == "__main__":

    # Key motivation: diversity can be achieved 2 ways: we can either get diverse prompts / get GPT-3.5 to rephrase the question separately
    # This code gets diversity from GPT-3.5

    # Getting the client
    DATASET = "triviaqa"
    DATASET_FOLDER = TRIVIA_QA_PROCESSED_FOLDER
    
    KEY = ""
    CLIENT = OpenAI(api_key = KEY)

    # Getting the results
    for split in ["train", "val", "test"]:
        dataloader = get_dataloader(DATASET, os.path.join(DATASET_FOLDER, f"{split}.json"), batch_size = 1) # Seems like the API only supports batch size of 1
        # diverse_outputs = get_gpt4_diverse_outputs(dataloader)
        # write_json(diverse_outputs, os.path.join(DATASET_FOLDER, f"{split}_w_diverse_qns_GPT4.json"))
        get_gpt35_diverse_outputs(dataloader)


