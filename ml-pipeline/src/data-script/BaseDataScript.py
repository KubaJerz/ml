import torch
from ..utils.validation_utils import validate_data_config

class BaseDataScript:
    def __init__(self, config: dict):
        self._validate_config(config)
        self.config = config

    def get_data(self):
        """
        Get datasets based on configuration.
        
        Returns:
            If split_type is "train,test":
                tuple: (train_dataset, test_dataset)
            If split_type is "train,test,val":
                tuple: (train_dataset, test_dataset, val_dataset)
        """
        raise NotImplementedError("Subclasses must implement get_data()")

    def _validate_config(self, config):
        validate_data_config(config)

    def _split_dataset(self, dataset):
        total_size = len(dataset)
        splits = self.config['train_split']
        
        if self.config['split_type'] == "train,test":
            train_size = int(splits[0] * total_size)
            test_size = total_size - train_size
            
            return torch.utils.data.random_split(
                dataset, 
                [train_size, test_size],
                generator=torch.Generator().manual_seed(self.config['seed']) 
            )
        elif self.config['split_type'] == "train,test,val":
            train_size = int(splits[0] * total_size)
            val_size = int(splits[1] * total_size)
            test_size = total_size - train_size - val_size
            
            return torch.utils.data.random_split(
                dataset, 
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.config['seed'])
            )
        else:
            raise NameError(f'{self.config['split_type']} is not valid for self.config["split_type"]')
