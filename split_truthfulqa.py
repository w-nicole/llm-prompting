    
import os
import config
import random
import pandas as pd

if __name__ == '__main__':
    
    df = pd.read_csv('datasets/truthfulQA.csv')
    df_size = df.shape[0]
    
    val_size = int(df_size * config.VAL_PERCENTAGE)
    test_size = int(df_size * config.TEST_PERCENTAGE)
    train_size = df_size - val_size - test_size
    
    phases = ['train'] * train_size + ['val'] * val_size + ['test'] * test_size 
    random.shuffle(phases)
    
    df['phase'] = phases

    df.to_csv(config.TRUTHFUL_QA_PATH)
    print('Dataset written to', config.TRUTHFUL_QA_PATH)