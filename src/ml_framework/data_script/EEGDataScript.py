from typing import Tuple, List, Union
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
import os
from tqdm import tqdm
from pathlib import Path

from .BaseDataScript import BaseDataScript


class EEGDataset(Dataset):
    def __init__(self, raw_data_path):
        X, y = torch.load(raw_data_path, weights_only=True)
        self.x = X
        self.y = y
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class EEGDataScript(BaseDataScript):    
    def __init__(self, config: dict):
        super().__init__(config)
        self.data_path = Path(self.config['absolute_path'])
           
    def _load_datasets(self) -> List[Dataset]:
        all_datasets = []
        i = 0 
        for file_path in tqdm(sorted(self.data_path.glob('*.pt')), desc='Loading datasets'):
            if i < 1:
                try:
                    dataset = EEGDataset(str(file_path))
                    all_datasets.append(dataset)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    continue
            i += 1

        return ConcatDataset(all_datasets)

    
    def get_datasets(self) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        full_dataset = self._load_datasets()
        if not self.config.get('use_full', True):
            percent_to_use = self.config.get('use_percent', 0.1)
            if not (0.0 < percent_to_use <= 1.0):
                raise ValueError("The 'use_percent' must be a float between 0 and 1.")
            subset_length = int(len(full_dataset) * percent_to_use)
            indices = list(range(subset_length))
            to_use_dataset = Subset(full_dataset, indices)
        else:
            to_use_dataset = full_dataset
        return super()._split_dataset(to_use_dataset)
        
    
    def get_data_loaders(self):
        datasets = self.get_datasets()
        loader_config = {
            'batch_size': self.config.get('batch_size', 128),
            'num_workers': self.config.get('num_workers', 0),
            'pin_memory': self.config.get('pin_memory', True)
        }
        
        if self.config['split_type'] == "train,dev":
            train_dataset, dev_dataset = datasets
            return (
                DataLoader(train_dataset, shuffle=False, **loader_config),
                DataLoader(dev_dataset, shuffle=False, **loader_config)
            )
        else:
            train_dataset, dev_dataset, test_dataset = datasets
            return (
                DataLoader(train_dataset, shuffle=False, **loader_config),
                DataLoader(dev_dataset, shuffle=False, **loader_config),
                DataLoader(test_dataset, shuffle=False, **loader_config)
            )
