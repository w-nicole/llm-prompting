import os
import random
import numpy as np
from string import ascii_lowercase

import torch

# Hyperparameters
SEED = 0 

BATCH_SIZE = 8
MAX_OUTPUT_LENGTH = 80

N_DIVERSE_QUES = 10 

# For bert scorer 
SCORING_THRESHOLD = 0.8

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
MODEL_CHECKPOINTS = {'flan-t5-small' : 'model_weights/flan/flan-t5-small',
                     'flan-t5-base' : 'model_weights/flan/flan-t5-base',
                     'flan-t5-large' : 'model_weights/flan/flan-t5-large',
                     'flan-t5-xl' : 'model_weights/flan/flan-t5-xl',
                     'shearedllama-1.3b' :'model_weights/sheared_llama/v13',
                     'shearedllama-2.7b' :'model_weights/sheared_llama/v27',
                     'shearedllama-bling-1.3b' :'model_weights/sheared_llama_bling/v13',
                     'shearedllama-bling-2.7b' :'model_weights/sheared_llama_bling/v27',
                     'vicuna-7b': 'model_weights/vicuna/7b',
                     'mistral-7b':'model_weights/mistral/7b', 
                     'llama2-7b': 'model_weights/llama2/7b'}
# Device 
DEVICE_IDX = 0
DEVICE = torch.device(f"cuda:{DEVICE_IDX}" if torch.cuda.is_available() else "cpu")

# For confidence prompting
# Percentage as confidence
CONFIDENCE_INTERVAL = 10
OPTIONS = np.arange(0, 101, CONFIDENCE_INTERVAL)
ALPHABETS = [str.upper(c) for c in ascii_lowercase][:len(OPTIONS)]
CONFIDENCE_OPTIONS = {f"{ALPHABETS[i]})" : f"{OPTIONS[i]}" + "% to "  + f"{OPTIONS[i+1]}" + "%" for i in range(len(ALPHABETS) - 1)}

# Natural language as confidence 
CONFIDENCE_OPTIONS_NL = {"A)" : "Absolutely uncertain",
                         "B)" : "Extremely uncertain",
                         "C)" : "Highly uncertain",
                         "D)" : "Uncertain",
                         "E)" : "Unsure, leaning towards uncertain",
                         "F)" : "Unsure, leaning towards certain",
                         "G)" : "Certain",
                         "H)" : "Highly certain", 
                         "I)" : "Extremely certain",
                         "J)" : "Absolutely certain"}

assert len(CONFIDENCE_OPTIONS) == len(CONFIDENCE_OPTIONS_NL)
CONFIDENCE_SCORE_NL_MAPPING = {v : CONFIDENCE_OPTIONS[k] for k, v in CONFIDENCE_OPTIONS_NL.items()}

# The index of true and false 
TRUE_FALSE_IDX = {'flan-t5-small' : {"true": 4273, "false": 150},
                  'flan-t5-base' : {"true": 4273, "false": 150},
                  'flan-t5-large' : {"true": 4273, "false": 150},
                  'shearedllama-bling-1.3b' : {"true": 319, "false": 350},
                  'shearedllama-bling-2.7b' : {"true": 319, "false": 350},
                  'mistral-7b': {"true": 365, "false": 330},
                  'llama2-7b': {"true": 319, "false": 350}}

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