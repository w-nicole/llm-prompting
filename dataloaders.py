import os 
import json 
import pandas as pd

import torch
from torch.utils.data import DataLoader

from config import *
from utils import read_json

class TruthfulQADataset(torch.utils.data.Dataset):
    
    def __init__(self, path):
        self.data = read_json(path)
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        print("to be changed")
        a = z 

        entry = self.phase_df.iloc[idx]
        return entry['Question'], entry['Best Answer']
        
class SciQDataset(torch.utils.data.Dataset):
    
    def __init__(self, phase):
        phase_dict = {
            'train' : 'train',
            'val' : 'valid',
            'test' : 'test'
        }
        with open(os.path.join(config.SCIQ_FOLDER, f'{phase_dict[phase]}.json')) as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        entry = self.data[idx]
        return entry['question'], entry['correct_answer']

class TriviaQADataset(torch.utils.data.Dataset):
    
    def __init__(self, phase):
        phase_dict = {
            'train' : 'train',
            'val' : 'dev',
            'test' : 'test-without-answers'
        }
        data_path = os.path.join(config.TRIVIA_QA_FOLDER, f'unfiltered-web-{phase_dict[phase]}.json')
        with open(data_path) as f:
            self.data = json.load(f)['Data']
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        entry = self.data[idx]
        return entry['Question'], entry['Answer']['Value']
        
def get_dataloader(name, path):

    if name not in dataset.keys():
        raise Exception(f"{name} not supported. Please check implementation.")
    
    dataset = {"truthfulqa": TruthfulQADataset, 
               "sciq": SciQDataset, 
               "triviaqa": TriviaQADataset}[name](path)
    
    return DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)
