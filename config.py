import os
import random
import numpy as np
from string import ascii_lowercase

import torch

# Hyperparameters
SEED = 0 

BATCH_SIZE = 32
MAX_OUTPUT_LENGTH = 50

CONFIDENCE_INTERVAL = 10

N_DIVERSE_QUES = 10 

# Data paths
DATASET_RAW_FOLDER = 'datasets/raw'
TRUTHFUL_QA_RAW_FOLDER = os.path.join(DATASET_RAW_FOLDER, 'truthfulQA.csv')
TRIVIA_QA_RAW_FOLDER = os.path.join(DATASET_RAW_FOLDER, 'triviaqa', 'triviaqa-unfiltered')
SCIQ_RAW_FOLDER = os.path.join(DATASET_RAW_FOLDER, 'sciq', 'SciQ dataset-2 3')

DATASET_PROCESSED_FOLDER = 'datasets/processed'
TRUTHFUL_QA_PROCESSED_FOLDER = os.path.join(DATASET_PROCESSED_FOLDER, 'truthfulqa')
TRIVIA_QA_PROCESSED_FOLDER = os.path.join(DATASET_PROCESSED_FOLDER, 'triviaqa')
SCIQ_PROCESSED_FOLDER = os.path.join(DATASET_PROCESSED_FOLDER, 'sciq')

OUTPUT_FOLDER = 'outputs'

# Model weights
BERT_SCORER_MODEL = "microsoft/deberta-xlarge-mnli"
MODEL_CHECKPOINTS = {'flan-t5-small' : 'model_weights/flan-t5/small',
                     'flan-t5-base' : 'model_weights/flan-t5/base',
                     'flan-t5-large' : 'model_weights/flan-t5/large',
                     'vicuna-7b': 'model_weights/vicuna/7b',
                     'mistral-7b':'model_weights/mistral/7b', 
                     'shearedllama-1.3b' :'model_weights/sheared_llama/1.3b',
                     'shearedllama-2.7b' :'model_weights/sheared_llama/2.7b',
                     'lamma2-7b': ''}

# Device 
DEVICE_IDX = 1
DEVICE = torch.device(f"cuda:{DEVICE_IDX}" if torch.cuda.is_available() else "cpu")

# For confidence prompting
OPTIONS = np.arange(0, 101, CONFIDENCE_INTERVAL)
ALPHABETS = [str.upper(c) for c in ascii_lowercase][:len(OPTIONS)]
CONFIDENCE_OPTIONS = {f"({ALPHABETS[i]})" : OPTIONS[i] for i in range(len(ALPHABETS))}

# Setting of parameters and creating of folders for saving of results
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
TRUTHFUL_QA_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, 'truthfulqa')
TRIVIA_QA_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, 'triviaqa')
SCIQ_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, 'sciq')

if not os.path.exists(TRUTHFUL_QA_OUTPUT_FOLDER): os.makedirs(TRUTHFUL_QA_OUTPUT_FOLDER)
if not os.path.exists(TRIVIA_QA_OUTPUT_FOLDER): os.makedirs(TRIVIA_QA_OUTPUT_FOLDER)
if not os.path.exists(SCIQ_OUTPUT_FOLDER): os.makedirs(SCIQ_OUTPUT_FOLDER)