import os
import random
import collections
import pandas as pd

from utils import * 
from config import *

if __name__ == '__main__':

    # Process truthfulQA
    truthfulqa = pd.read_csv(TRUTHFUL_QA_RAW_PATH)
    truthfulqa_counter = collections.Counter(truthfulqa.loc[:, "Category"])
    val_categories = ["Politics", "Finance", "Law"]
    test_categories = ["Confusion: People", "Superstitions", "Myths and Fairytales", "Religion"]

    truthfulqa_train = truthfulqa.loc[truthfulqa.loc[:, "Category"].apply(lambda x: x not in val_categories + test_categories), :]
    truthfulqa_val = truthfulqa.loc[truthfulqa.loc[:, "Category"].apply(lambda x: x in val_categories), :]
    truthfulqa_test = truthfulqa.loc[truthfulqa.loc[:, "Category"].apply(lambda x: x in test_categories), :]

    truthfulqa_train = json.loads(truthfulqa_train.to_json(orient = "records"))
    truthfulqa_val = json.loads(truthfulqa_val.to_json(orient = "records"))
    truthfulqa_test = json.loads(truthfulqa_test.to_json(orient = "records"))

    # Process sciQ
    sciq_train = read_json(os.path.join(SCIQ_RAW_FOLDER, "train.json"))
    sciq_val = read_json(os.path.join(SCIQ_RAW_FOLDER, "valid.json"))
    sciq_test = read_json(os.path.join(SCIQ_RAW_FOLDER, "test.json"))

    sciq_train_small = sciq_train[:50]
    sciq_val_small = sciq_val[:50]
    sciq_test_small = sciq_test[:50]

    # Process triviaQA 
    # Note: we are reporting results for val and getting a subset of the train dataset as val 
    # Note: TO MAKE SURE THAT'S WHAT THE TIAN 2023 ET AL. PAPER IS DOING
    triviaqa_train = read_json(os.path.join(TRIVIA_QA_RAW_FOLDER, "unfiltered-web-train.json"))["Data"]
    triviaqa_test = read_json(os.path.join(TRIVIA_QA_RAW_FOLDER, "unfiltered-web-dev.json"))["Data"]
    triviaqa_val = triviaqa_train[: len(triviaqa_test)]
    triviaqa_train = triviaqa_train[len(triviaqa_test):]

    triviaqa_train_small = triviaqa_train[:50]
    triviaqa_val_small = triviaqa_val[:50]
    triviaqa_test_small = triviaqa_test[:50]
    
    # Write the datasets to the processed folder
    if not os.path.exists(DATASET_PROCESSED_FOLDER): os.makedirs(DATASET_PROCESSED_FOLDER)
    if not os.path.exists(TRUTHFUL_QA_PROCESSED_PATH): os.makedirs(TRUTHFUL_QA_PROCESSED_PATH)
    if not os.path.exists(SCIQ_PROCESSED_FOLDER): os.makedirs(SCIQ_PROCESSED_FOLDER)
    if not os.path.exists(TRIVIA_QA_PROCESSED_FOLDER): os.makedirs(TRIVIA_QA_PROCESSED_FOLDER)

    # truthfulQA
    write_json(truthfulqa_train, os.path.join(TRUTHFUL_QA_PROCESSED_PATH, "train.json"))
    write_json(truthfulqa_val, os.path.join(TRUTHFUL_QA_PROCESSED_PATH, "val.json"))
    write_json(truthfulqa_test, os.path.join(TRUTHFUL_QA_PROCESSED_PATH, "test.json"))

    # sciQ
    write_json(sciq_train, os.path.join(SCIQ_PROCESSED_FOLDER, "train.json"))
    write_json(sciq_val, os.path.join(SCIQ_PROCESSED_FOLDER, "val.json"))
    write_json(sciq_test, os.path.join(SCIQ_PROCESSED_FOLDER, "test.json"))

    write_json(sciq_train_small, os.path.join(SCIQ_PROCESSED_FOLDER, "train_small.json"))
    write_json(sciq_val_small, os.path.join(SCIQ_PROCESSED_FOLDER, "val_small.json"))
    write_json(sciq_test_small, os.path.join(SCIQ_PROCESSED_FOLDER, "test_small.json"))

    # triviaQA
    write_json(triviaqa_train, os.path.join(TRIVIA_QA_PROCESSED_FOLDER, "train.json"))
    write_json(triviaqa_val, os.path.join(TRIVIA_QA_PROCESSED_FOLDER, "val.json"))
    write_json(triviaqa_test, os.path.join(TRIVIA_QA_PROCESSED_FOLDER, "test.json"))

    write_json(triviaqa_train_small, os.path.join(TRIVIA_QA_PROCESSED_FOLDER, "train_small.json"))
    write_json(triviaqa_val_small, os.path.join(TRIVIA_QA_PROCESSED_FOLDER, "val_small.json"))
    write_json(triviaqa_test_small, os.path.join(TRIVIA_QA_PROCESSED_FOLDER, "test_small.json"))
