import os
import random
import numpy as np
from string import ascii_lowercase

import torch

SEED = 0 

BATCH_SIZE = 4
MAX_OUTPUT_LENGTH = 20

DATASET_RAW_FOLDER = 'datasets/raw'
TRUTHFUL_QA_RAW_PATH = os.path.join(DATASET_RAW_FOLDER, 'truthfulQA.csv')
TRIVIA_QA_RAW_FOLDER = os.path.join(DATASET_RAW_FOLDER, 'triviaqa', 'triviaqa-unfiltered')
SCIQ_RAW_FOLDER = os.path.join(DATASET_RAW_FOLDER, 'sciq', 'SciQ dataset-2 3')

DATASET_PROCESSED_FOLDER = 'datasets/processed'
TRUTHFUL_QA_PROCESSED_PATH = os.path.join(DATASET_PROCESSED_FOLDER, 'truthfulqa')
TRIVIA_QA_PROCESSED_FOLDER = os.path.join(DATASET_PROCESSED_FOLDER, 'triviaqa')
SCIQ_PROCESSED_FOLDER = os.path.join(DATASET_PROCESSED_FOLDER, 'sciq')

MODEL_CHECKPOINTS = {'flan-t5-small' : 'model_weights/flan-t5/small',
                     'vicuna-7b': 'model_weights/vicuna/7b',
                     'mistral':'',
                     'shearedllama-1.3b' :'model_weights/sheared_llama/1.3b',
                     'shearedllama-2.7b' :'model_weights/sheared_llama/2.7b',
                     'alpaca': ''}

INTERVAL = 10
OPTIONS = np.arange(0, 101, INTERVAL)
ALPHABETS = [str.upper(c) for c in ascii_lowercase][:len(OPTIONS)]
CONFIDENCE_OPTIONS = {f"({ALPHABETS[i]})" : OPTIONS[i] for i in range(len(ALPHABETS))}

N_DIVERSE_QUES = 10 

BERT_SCORER_MODEL = "microsoft/deberta-xlarge-mnli"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)