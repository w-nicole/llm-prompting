
import os

SEED = 0 

VAL_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.1

BATCH_SIZE = 2

DATASET_FOLDER = 'datasets'
TRUTHFUL_QA_PATH = os.path.join(DATASET_FOLDER, 'split_truthfulQA.csv')
TRIVIA_QA = os.path.join(DATASET_FOLDER, 'triviaqa', 'triviaqa-unfiltered')

import random
import numpy as np
import torch

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)