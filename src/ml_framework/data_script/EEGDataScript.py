from typing import Tuple, List, Union
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
import os
from tqdm import tqdm
from pathlib import Path

from ml_framework.data_script.BaseDataScript import BaseDataScript

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
        self.data_path = Path(self.config['data_absolute_path'])
           
    def _load_datasets(self) -> List[Dataset]:
        all_datasets = []
        for file_path in tqdm(sorted(self.data_path.glob('*.pt')), desc='Loading datasets'):
            try:
                dataset = EEGDataset(str(file_path))
                all_datasets.append(dataset)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

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
        if self.config['split_type'] == "train,dev":
            train_dataset, dev_dataset = datasets
            return (
                super().create_loader(train_dataset, 'train_batch_size'),
                super().create_loader(dev_dataset, 'dev_batch_size')
            )
        else:  # train,dev,test
            train_dataset, dev_dataset, test_dataset = datasets
            return (
                super().create_loader(train_dataset, 'train_batch_size'),
                super().create_loader(dev_dataset, 'dev_batch_size'),
                super().create_loader(test_dataset, 'test_batch_size')
            )
