
import config
import torch
import pandas as pd
import json

from torch.utils.data import DataLoader

class TruthfulQADataset(torch.utils.data.Dataset):
    
    def __init__(self, phase):
        df = pd.read_csv(config.TRUTHFUL_QA_PATH)
        self.phase_df = df[df.phase == phase]
        
    def __len__(self):
        return self.phase_df.shape[0]
        
    def __getitem__(self, idx):
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
    
    def __init__(self):
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
        return self.data['Question'], self.data['Answer']
        
def get_dataloader(dataset):
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
