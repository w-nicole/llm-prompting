import os 
import json 
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import *
from utils import read_json

class TruthfulQADataset(Dataset):
    
    def __init__(self, path):
        self.data = read_json(path)
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return str(self.data[idx]['id_']), self.data[idx]['Question'], self.data[idx]['Best Answer'], self.data[idx]["diverse_questions"]
        
class SciQDataset(Dataset):
    
    def __init__(self, path):
        self.data = read_json(path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]['id_'], self.data[idx]['question'], self.data[idx]['correct_answer'], self.data[idx]["diverse_questions"]

class TriviaQADataset(Dataset):
    
    def __init__(self, path):
        self.data = read_json(path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]["id_"], self.data[idx]['Question'], self.data[idx]['Answer']['Value'], self.data[idx]["diverse_questions"]
    

class PopQADataset(Dataset):
    def __init__(self, path):
        self.data = read_json(path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]["id_"], self.data[idx]['question'], self.data[idx]['possible_answers'], self.data[idx]["diverse_questions"]

        
def get_dataset(name):

    datasets = {
        "truthfulqa": TruthfulQADataset, 
        "sciq": SciQDataset, 
        "triviaqa": TriviaQADataset,
        "popqa": PopQADataset
        }
    
    if name not in datasets.keys():
        raise Exception(f"{name} not supported. Please check implementation.")

    return datasets[name]

def get_dataloader(name, path, batch_size):

    return DataLoader(get_dataset(name)(path), batch_size = batch_size, shuffle = False, collate_fn = collate_fn)

def collate_fn(data):

    id_, qns, ans, diverse_qns = [], [], [], []

    for rec in data:
        
        id_.append(rec[0])
        qns.append(rec[1]) 
        ans.append(rec[2])
        diverse_qns.append(rec[3])

    return id_, qns, ans, diverse_qns
