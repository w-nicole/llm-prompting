import os 
import re
from tqdm import tqdm

from config import * 
from utils import read_json, write_json

def lookup_rec(master_data, id_):

    for r in master_data:
        if r["id_"] == id_:
            return r
    
    raise Exception("Record not in master data")

def merge(master_data, folder, clean_record = False):

    all_results = [] 
    
    for f in tqdm(os.listdir(folder)):

        id_ = f.replace(".json", "")
        path = os.path.join(folder, f)
        data = read_json(path)
        if clean_record:
            if type(data) == str:
                data = data.split("\n")
            data = [re.sub(r'\d+.', '', r).strip() for r in data]

        rec = lookup_rec(master_data, id_)

        # Add in the diverse question
        rec["diverse_questions"] = data
        all_results.append(rec)

    return all_results

def merge_truthfulqa(master_data, diverse_qns):

    all_results = [] 
    
    for q in tqdm(diverse_qns):

        id_ = q["id_"]
        q = q["GPT4_diverse_question"]
        rec = lookup_rec(master_data, id_)

        # Add in the diverse question
        rec["diverse_questions"] = q
        all_results.append(rec)

    return all_results

if __name__ == "__main__":

    MERGE_35 = False 

    if MERGE_35: # Note this is VERY specific to triviaqa
        CACHE_FOLDER = "./datasets/cache/cache_triviaqa_train_GPT35-turbo"
        OUTPUT_FOLDER = TRIVIA_QA_PROCESSED_FOLDER
        MASTER_DATA = read_json("./datasets/processed/triviaqa/train.json")
        OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "data_w_GPT35-turbo_output.json")
        OUTPUT_SMALL_PATH = os.path.join(OUTPUT_FOLDER, f"data_w_GPT35-turbo_output_{SMALL_DATASET_SIZE}.json")
        merged_data = merge(MASTER_DATA, CACHE_FOLDER, clean_record = True)
        merged_data_small = merged_data[:SMALL_DATASET_SIZE]
        write_json(merged_data, OUTPUT_PATH)
        write_json(merged_data_small, OUTPUT_SMALL_PATH)
    
    else:
        DATASET = "sciq"
        OUTPUT_FOLDER = SCIQ_PROCESSED_FOLDER
        OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "data_w_GPT4_output.json")
        OUTPUT_SMALL_PATH = os.path.join(OUTPUT_FOLDER, f"data_w_GPT4_output_{SMALL_DATASET_SIZE}.json")
    
        if DATASET == "truthfulqa":
            CACHE_TRAIN = read_json(os.path.join(OUTPUT_FOLDER, "train_w_diverse_qns_GPT4.json"))
            CACHE_VAL = read_json(os.path.join(OUTPUT_FOLDER, "val_w_diverse_qns_GPT4.json"))
            CACHE_TEST = read_json(os.path.join(OUTPUT_FOLDER, "test_w_diverse_qns_GPT4.json"))
            CACHE = CACHE_TRAIN + CACHE_VAL + CACHE_TEST

            TRAIN = read_json(os.path.join(OUTPUT_FOLDER, "train.json"))
            DEV = read_json(os.path.join(OUTPUT_FOLDER, "val.json"))
            TEST = read_json(os.path.join(OUTPUT_FOLDER, "test.json"))
            DATA = TRAIN + DEV + TEST

            merged_data = merge_truthfulqa(DATA, CACHE)

        else:        
            CACHE_FOLDER = f"./datasets/cache/cache_{DATASET}_train"
            MASTER_DATA = read_json(os.path.join(OUTPUT_FOLDER, "train.json"))
            merged_data = merge(MASTER_DATA, CACHE_FOLDER)

        merged_data_small = merged_data[:SMALL_DATASET_SIZE]
        write_json(merged_data, OUTPUT_PATH)
        write_json(merged_data_small, OUTPUT_SMALL_PATH)
    