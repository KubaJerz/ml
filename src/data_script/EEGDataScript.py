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
        for file_path in tqdm(sorted(self.data_path.glob('*.pt')), desc='Loading datasets'):
            try:
                dataset = EEGDataset(str(file_path))
                all_datasets.append(dataset)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue

        return ConcatDataset(all_datasets)

    
    def get_datasets(self) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        combined_dataset = self._load_datasets()

        if self.config.get('shuffle', True):
            generator = torch.Generator().manual_seed(self.config['seed'])
            indices = torch.randperm(len(combined_dataset), generator=generator).tolist()
            combined_dataset = [combined_dataset[i] for i in indices]


        if self.config['split_type'] == "train,dev":
            train_ratio = self.config['split_ratios'][0]
            train_idx = int(len(combined_dataset) * train_ratio)

            train_dataset = Subset(combined_dataset, range(train_idx))
            dev_dataset = Subset(combined_dataset, range(train_idx, len(combined_dataset)))

            return train_dataset, dev_dataset

        elif self.config['split_type'] == "train,dev,test":
            train_ratio = self.config['split_ratios'][0]
            dev_ratio = self.config['split_ratios'][1]

            train_idx = int(len(combined_dataset) * train_ratio)
            dev_idx = int(len(combined_dataset) * (train_ratio + dev_ratio))
            print(f'trian idx {train_idx} dev idx {dev_idx}')

            train_dataset = Subset(combined_dataset, range(train_idx))
            dev_dataset = Subset(combined_dataset, range(train_idx, dev_idx))
            test_dataset = Subset(combined_dataset, range(dev_idx, len(combined_dataset)))
            print(f' full size is {len(combined_dataset)} \n trian size is: {len(train_dataset)} \n dev siz is: {len(dev_dataset)} \n test size:{len(test_dataset)}')

            return train_dataset, dev_dataset, test_dataset
    
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
