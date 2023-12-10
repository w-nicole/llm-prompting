import os
import random
import numpy as np
from string import ascii_lowercase

import torch

# Hyperparameters
SEED = 0 

BATCH_SIZE = 8
MAX_OUTPUT_LENGTH = 200

N_DIVERSE_QUES = 10 
SMALL_DATASET_SIZE = 200

# For bert scorer 
SCORING_THRESHOLD = 0.7

# For hybrid model 
ALPHA = 0.2

# Data paths
DATASET_RAW_FOLDER = 'datasets/raw'
TRUTHFUL_QA_RAW_FOLDER = os.path.join(DATASET_RAW_FOLDER, 'truthfulQA.csv')
TRIVIA_QA_RAW_FOLDER = os.path.join(DATASET_RAW_FOLDER, 'triviaqa', 'triviaqa-unfiltered')
SCIQ_RAW_FOLDER = os.path.join(DATASET_RAW_FOLDER, 'sciq', 'SciQ dataset-2 3')

DATASET_PROCESSED_FOLDER = 'datasets/processed'
TRUTHFUL_QA_PROCESSED_FOLDER = os.path.join(DATASET_PROCESSED_FOLDER, 'truthfulqa')
TRIVIA_QA_PROCESSED_FOLDER = os.path.join(DATASET_PROCESSED_FOLDER, 'triviaqa')
SCIQ_PROCESSED_FOLDER = os.path.join(DATASET_PROCESSED_FOLDER, 'sciq')

FILENAME = f"data_w_GPT4_output_{SMALL_DATASET_SIZE}.json"

OUTPUT_FOLDER = 'outputs'
RESULTS_FOLDER = 'results'

# Model weights
BERT_SCORER_MODEL = "microsoft/deberta-xlarge-mnli"
MODEL_CHECKPOINTS = {'flan-t5-large' : 'model_weights/flan-t5/large',
                     'flan-t5-xl' : 'model_weights/flan-t5/xl',
                     'mistral-7b-instruct':'model_weights/mistral/7b-instruct', 
                     'llama2-7b-chat': 'model_weights/llama2/7b-chat',
                     'llama2-13b-chat': 'model_weights/llama2/13b-chat',
                     'llama2-70b-chat': 'model_weights/llama2/70b-chat'}

# Device 
DEVICE_IDX = "1,2,3,4,5,6,7" # separate with comma if using multiple GPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEMORY_ALLOCATION = {int(i) : "38GB" for i in DEVICE_IDX.split(",")}

# For confidence prompting
# Percentage as confidence
CONFIDENCE_INTERVAL = 10
OPTIONS = np.arange(0, 101, CONFIDENCE_INTERVAL)
ALPHABETS = [str.upper(c) for c in ascii_lowercase][:len(OPTIONS)]
CONFIDENCE_OPTIONS = {f"({ALPHABETS[i]})" : f"{OPTIONS[i]}" + "% to "  + f"{OPTIONS[i+1]}" + "%" for i in range(len(ALPHABETS) - 1)}

# Natural language as confidence 
CONFIDENCE_OPTIONS_NL = {"(A)" : "Absolutely uncertain",
                         "(B)" : "Extremely uncertain",
                         "(C)" : "Highly uncertain",
                         "(D)" : "Uncertain",
                         "(E)" : "Unsure, leaning towards uncertain",
                         "(F)" : "Unsure, leaning towards certain",
                         "(G)" : "Certain",
                         "(H)" : "Highly certain", 
                         "(I)" : "Extremely certain",
                         "(J)" : "Absolutely certain"}

assert len(CONFIDENCE_OPTIONS) == len(CONFIDENCE_OPTIONS_NL)
CONFIDENCE_SCORE_NL_MAPPING = {v : CONFIDENCE_OPTIONS[k] for k, v in CONFIDENCE_OPTIONS_NL.items()}

# The index of true and false 
TRUE_FALSE_IDX = {'flan-t5-large' : {"true": 4273, "false": 465},
                  'flan-t5-xl' : {"true": 4273, "false": 465},
                  'mistral-7b-instruct': {"true": 6110, "false": 8250},
                  'llama2-7b-chat': {"true": 5852, "false": 7700},
                  'llama2-13b-chat': {"true": 5852, "false": 7700},
                  'llama2-70b-chat': {"true": 5852, "false": 7700}}

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

if not os.path.exists(RESULTS_FOLDER): os.makedirs(RESULTS_FOLDER)