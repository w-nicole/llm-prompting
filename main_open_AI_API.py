import re
import argparse 
import traceback
from tqdm import tqdm 
from openai import OpenAI
from functools import partial

from config import *
from utils import *
from dataloaders import get_dataloader
from cache import is_response_cached, cache_response, multithread_with_tqdm

def get_response(batch, pbar):

    global STOP

    # Unpack the batch
    id_, qns, _, diverse_qns = batch
    cache_id = id_[0]
    
    if STOP:
        return
    
    if is_response_cached_p(cache_id):
        pbar.update(1)
        return

    # Get different prompts and responses
    # Format the question

    # Get response
    try:
        response = CLIENT.chat.completions.create(model="gpt-3.5-turbo", messages=diverse_qns)

    except Exception as e:
        STOP = True
        print(e)
        print(traceback.format_exc())
        raise e

    # Get the response
    response = response.model_dump()["choices"][0]["message"]["content"]

    # Cache response
    cache_response_p(cache_id, response)

    # Update tqdm
    pbar.update(1)

def get_results(dataloader):
    multithread_with_tqdm(get_response, len(dataloader.dataset), dataloader, 20)

if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description = "Getting testing results of trained model")

    # Data settings 
    parser.add_argument("--dataset", type = str, default = "triviaqa", help = "The dataset")
    parser.add_argument("--model", type = str, default = "gpt-3.5-turbo", help = "LLM to use")

    # Set defaults 
    args = parser.parse_args()

    # Settings
    _, dataset_folder, output_folder = get_folders(args.dataset)
    args.dataset_folder = dataset_folder
    args.output_folder = output_folder
    args.temp_folder = os.path.join(output_folder, f"{args.model}_temp_API")
    
    KEY = ""
    CLIENT = OpenAI(api_key = KEY)
    
    # Get results
    STOP = False
    dataloader = get_dataloader(args.dataset, os.path.join(args.dataset_folder, FILENAME), 1)
    is_response_cached_p = partial(is_response_cached, folder = args.temp_folder)
    cache_response_p = partial(cache_response, folder = args.temp_folder)
    get_results(dataloader)