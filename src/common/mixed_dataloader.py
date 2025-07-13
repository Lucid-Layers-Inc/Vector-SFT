import random

import torch
from torch.utils.data import DataLoader, IterableDataset


class MixtureIterableLoader(IterableDataset):
    
    def __init__(self, loader_main: DataLoader, loader_calib: DataLoader, calib_prob: float):
        
        self.loaders = [loader_main, loader_calib]
        self.calib_probs = [1 - calib_prob, calib_prob]
        
    def __iter__(self):
        
        iterators = [iter(loader) for loader in self.loaders]
        
        for _ in range(len(self)):
            
            i = random.choices([0, 1], weights=self.calib_probs, k=1)[0]
            iterator = iterators[i]
            try:
                batch = next(iterator)
            except StopIteration:
                # we need to set iterator again
                iterators[i] = iter(self.loaders[i])
                batch = next(iterators[i])
            
            yield batch

 
    def __len__(self):
        return int(len(self.loaders[0]) / self.calib_probs[0])