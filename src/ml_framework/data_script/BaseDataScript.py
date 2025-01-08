import torch
from ..utils.validation_utils import validate_data_config
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset


class BaseDataScript(ABC):
    def __init__(self, config: dict):
        self._validate_config(config)
        self.config = config

    @abstractmethod
    def get_datasets(self):
        """
        Get datasets based on configuration.
        
        Returns:
            If split_type is "train,dev":
                tuple: (train_dataset, dev_dataset)
            If split_type is "train,dev,test":
                tuple: (train_dataset, dev_dataset, test_dataset)
        """
    
    @abstractmethod
    def get_data_loaders(self):
        """
        Get loaders based on configuration.
        
        Returns:
            If split_type is "train,dev":
                tuple: (train_dataset, dev_dataset)
            If split_type is "train,dev,test":
                tuple: (train_dataset, dev_dataset, test_dataset)
        """


    def _validate_config(self, config):
        validate_data_config(config)

    def _split_dataset(self, combined_dataset):
        if self.config.get('shuffle', True):
            generator = torch.Generator().manual_seed(self.config['seed'])
            indices = torch.randperm(len(combined_dataset), generator=generator).tolist()
            combined_dataset = [combined_dataset[i] for i in indices]


        if self.config['split_type'] == "train,dev":
            train_ratio = self.config['split_ratios'][0]
            train_end_idx = int(len(combined_dataset) * train_ratio)

            train_dataset = Subset(combined_dataset, range(train_end_idx))
            dev_dataset = Subset(combined_dataset, range(train_end_idx, len(combined_dataset)))

            return train_dataset, dev_dataset

        elif self.config['split_type'] == "train,dev,test":
            train_ratio = self.config['split_ratios'][0]
            dev_ratio = self.config['split_ratios'][1]

            train_end_idx = int(len(combined_dataset) * train_ratio)
            dev_end_idx = int(len(combined_dataset) * (train_ratio + dev_ratio))

            train_dataset = Subset(combined_dataset, range(train_end_idx))
            dev_dataset = Subset(combined_dataset, range(train_end_idx, dev_end_idx))
            test_dataset = Subset(combined_dataset, range(dev_end_idx, len(combined_dataset)))

            return train_dataset, dev_dataset, test_dataset
        else:
            raise NameError(f"{self.config['split_type']} is not valid for self.config['split_type']")
        
    def create_loader(self, dataset, batch_size_key):
        base_config = {
        'num_workers': self.config.get('num_workers', 0),
        'pin_memory': self.config.get('pin_memory', True),
    }

        batch_size = self.config.get(batch_size_key, 32)
        
        if batch_size == -1:
            batch_size = len(dataset)
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            **base_config
        )
