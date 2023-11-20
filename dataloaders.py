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
        return idx, self.data[idx]['Question'], self.data[idx]['Best Answer']
        
class SciQDataset(torch.utils.data.Dataset):
    
    def __init__(self, path):
        self.data = read_json(path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return idx, self.data[idx]['question'], self.data[idx]['correct_answer']

class TriviaQADataset(torch.utils.data.Dataset):
    
    def __init__(self, path):
        self.data = read_json(path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return idx, self.data[idx]['Question'], self.data[idx]['Answer']['Value']
        
def get_dataloader(name, path, batch_size = BATCH_SIZE):

    datasets = {"truthfulqa": TruthfulQADataset, 
               "sciq": SciQDataset, 
               "triviaqa": TriviaQADataset}
    
    if name not in datasets.keys():
        raise Exception(f"{name} not supported. Please check implementation.")
    
    return DataLoader(datasets[name](path), batch_size = batch_size, shuffle = False)
