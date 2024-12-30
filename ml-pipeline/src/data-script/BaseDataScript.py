import torch

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
        required_fields = {
            'absolute_path': str,
            'script_name': str,
            'split_type': str,
            'split_ratios': list,
            'shuffle': bool,
            'seed': int,
            'input_size': int,
            'input_channels': int,
            'output_size': int,
            'num_classes': int
        }
        
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(config[field], field_type):
                raise ValueError(f"Field {field} must be of type {field_type.__name__}")
        
        if config['split_type'] not in ["train,test", "train,test,val"]:
            raise ValueError("split_type must be either 'train,test' or 'train,test,val'")
        
        splits = config['split_ratios']
        if not all(isinstance(split, float) for split in splits):
            raise ValueError("All split values must be floats")
            
        if not all(0 < split < 1 for split in splits):
            raise ValueError("All split values must be between 0 and 1")
        
        expected_splits = 3 if config['split_type'] == "train,test,val" else 2
        if len(splits) != expected_splits:
            raise ValueError(f"Expected {expected_splits} split values for {config['split_type']}")
        
        if not abs(sum(splits) - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(splits)}")
        
        positive_fields = ['input_size', 'input_channels', 'output_size', 'num_classes']
        for field in positive_fields:
            if config[field] <= 0:
                raise ValueError(f"{field} must be positive")
            

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
