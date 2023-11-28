import os 
from tqdm import tqdm

from config import * 
from utils import read_json, write_json

def lookup_rec(master_data, id_):

    for r in master_data:
        if r["id_"] == id_:
            return r
    
    raise Exception("Record not in master data")

def merge(master_data, folder):

    all_results = [] 
    
    for f in tqdm(os.listdir(folder)):
        id_ = f.replace(".json", "")
        path = os.path.join(folder, f)
        data = read_json(path)
        rec = lookup_rec(master_data, id_)

        # Format the record
        qns, ans = rec["Question"], rec['Answer']['Value']
        all_results.append({"id_": id_,
                            "original_question": qns, 
                            "answer": ans, 
                            "GPT4_diverse_question" : data})
        
    return all_results

if __name__ == "__main__":

    DATASET = "triviaqa"
    CACHE_FOLDER = f"./datasets/cache/cache_{DATASET}_train"
    OUTPUT_FOLDER = TRIVIA_QA_PROCESSED_FOLDER
    MASTER_DATA = read_json(os.path.join(OUTPUT_FOLDER, "train.json"))
    OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, "subset_data _w_GPT_output.json")
    merged_data = merge(MASTER_DATA, CACHE_FOLDER)
    write_json(merged_data, OUTPUT_PATH)