import os
import re
import torch
import json
from torch.utils.data import IterableDataset, DataLoader
from itertools import cycle, islice
import random
from datasets import Dataset
from datasets import load_dataset, load_from_disk
from comm.comm_utils import *



class StreamDataset(IterableDataset):
    def __init__(self, cot_data_path, tokenizer, seq_length=1024):
        
        self.cot_data_path = cot_data_path
        
        with open(cot_data_path) as f:
            self.cot_data = json.load(f)
        
        self.buffer_tokens = []
        
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.it = None
        
    def state_dict(self):
        return {}
    
    def load_state_dict(self, state_dict):
        pass
    
    def get_sequence_from_cot(self):
        
        while True:
            
            keys = list(self.cot_data.keys())
            random.shuffle(keys)
            
            input_ids = []
            
            for k in keys:
                
                v = self.cot_data[k]
                
                input_ids += self.tokenizer(str(v) + '\n\n')['input_ids']
                if len(input_ids) < self.seq_length:
                    continue
                #     input_ids += [self.tokenizer.eos_token_id]*(self.seq_length - len(input_ids))
                
                input_ids = input_ids[:self.seq_length]
                input_ids = torch.tensor(input_ids).long()
                
                yield input_ids
                
                input_ids = []
        
    def get_sequence(self):
        
        it_cot = cycle(self.get_sequence_from_cot())
        
        while True:
            
            input_ids = next(it_cot)
                

            yield {
                'input_ids': input_ids,
            }
            
                
    def get_stream(self):
        return cycle(self.get_sequence())
    
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it
    