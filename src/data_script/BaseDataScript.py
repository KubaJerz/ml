import torch
from utils.validation_utils import validate_data_config
from abc import ABC, abstractmethod


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

    def _split_dataset(self, dataset):
        total_size = len(dataset)
        splits = self.config['split_ratios']
        
        if self.config['split_type'] == "train,dev":
            train_size = int(splits[0] * total_size)
            test_size = total_size - train_size
            
            return torch.utils.data.random_split(
                dataset, 
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.config['seed']) 
            )
        elif self.config['split_type'] == "train,dev,test":
            train_size = int(splits[0] * total_size)
            val_size = int(splits[1] * total_size)
            test_size = total_size - train_size - val_size
            
            return torch.utils.data.random_split(
                dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.config['seed'])
            )
        else:
            raise NameError(f"{self.config['split_type']} is not valid for self.config['split_type']")
