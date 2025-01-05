from typing import Tuple, List, Union
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
import os
from tqdm import tqdm
from pathlib import Path

from .base_data_script import BaseDataScript


class EEGDataset(Dataset):
    
    def __init__(self, data_path: str):
        features, labels = torch.load(data_path, weights_only=True)
        self.features = features.unsqueeze(1)
        self.labels = labels
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]
    
    def __len__(self) -> int:
        return len(self.features)


class EEGDataScript(BaseDataScript):    
    def __init__(self, config: dict):
        super().__init__(config)
        self.data_path = Path(self.config['data_path'])
           
    def _load_datasets(self) -> List[Dataset]:
        datasets = []
        for file_path in tqdm(sorted(self.data_path.glob('*.pt')), desc='Loading EEG datasets'):
            try:
                dataset = EEGDataset(str(file_path))
                datasets.append(dataset)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
        return datasets
    
    def get_datasets(self) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        all_datasets = self._load_datasets()
        combined_dataset = ConcatDataset(all_datasets)

        if self.config.get('shuffle', True):
            generator = torch.Generator().manual_seed(self.config['seed'])
            indices = torch.randperm(len(combined_dataset), generator=generator).tolist()
            combined_dataset = ConcatDataset([combined_dataset[i] for i in indices])

        if self.config['split_type'] == "train,test":
            train_ratio = self.config['train_split'][0]
            train_idx = int(len(combined_dataset) * train_ratio)

            train_dataset = Subset(combined_dataset, range(train_idx))
            test_dataset = Subset(combined_dataset, range(train_idx, len(combined_dataset)))

            return train_dataset, test_dataset

        elif self.config['split_type'] == "train,test,val":
            train_ratio = self.config['train_split'][0]
            val_ratio = self.config['train_split'][1]

            train_idx = int(len(combined_dataset) * train_ratio)
            val_idx = int(len(combined_dataset) * (train_ratio + val_ratio))

            train_dataset = Subset(combined_dataset, range(train_idx))
            val_dataset = Subset(combined_dataset, range(train_idx, val_idx))
            test_dataset = Subset(combined_dataset, range(val_idx, len(combined_dataset)))

            return train_dataset, val_dataset, test_dataset
    
    def get_data_loaders(self):
        datasets = self.get_datasets()
        loader_config = {
            'batch_size': self.config.get('batch_size', 128),
            'num_workers': self.config.get('num_workers', 0),
            'pin_memory': self.config.get('pin_memory', True)
        }
        
        if self.config['split_type'] == "train,test":
            train_dataset, test_dataset = datasets
            return (
                DataLoader(train_dataset, shuffle=False, **loader_config),
                DataLoader(test_dataset, shuffle=False, **loader_config)
            )
        else:
            train_dataset, val_dataset, test_dataset = datasets
            return (
                DataLoader(train_dataset, shuffle=False, **loader_config),
                DataLoader(val_dataset, shuffle=False, **loader_config),
                DataLoader(test_dataset, shuffle=False, **loader_config)
            )
