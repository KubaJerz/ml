from typing import Tuple, List, Union
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
from ..utils.validation_utils import validate_data_config
from .BaseDataScript import BaseDataScript

class EEGDataset(Dataset):
    """Dataset class for EEG data."""
    def __init__(self, raw_data_path: str):
        X, y = torch.load(raw_data_path, weights_only=True)
        self.x = X
        self.y = y
        self.n_samples = len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return self.n_samples

class EEGDataScript(BaseDataScript):
    """Implementation of data script for EEG data processing."""
    
    def __init__(self, config: dict):
        self.config = config
        validate_data_config(config)
        self.data_path = Path(self.config['data_absolute_path'])
           
    def get_datasets(self) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        """Get train/dev or train/dev/test dataset splits."""
        full_dataset = self._load_datasets()
        processed_dataset = self._process_dataset(full_dataset)
        return self._split_dataset(processed_dataset)

    def get_data_loaders(self) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        """Get DataLoader objects for the datasets."""
        datasets = self.get_datasets()
        
        if self.config['split_type'] == "train,dev":
            train_dataset, dev_dataset = datasets
            return (
                self._create_loader(train_dataset, 'train_batch_size'),
                self._create_loader(dev_dataset, 'dev_batch_size')
            )
        else:  # train,dev,test
            train_dataset, dev_dataset, test_dataset = datasets
            return (
                self._create_loader(train_dataset, 'train_batch_size'),
                self._create_loader(dev_dataset, 'dev_batch_size'),
                self._create_loader(test_dataset, 'test_batch_size')
            )

    def _load_datasets(self) -> ConcatDataset:
        """Load and combine all EEG datasets from the data path."""
        all_datasets = []
        for file_path in tqdm(sorted(self.data_path.glob('*.pt')), desc='Loading datasets'):
            try:
                dataset = EEGDataset(str(file_path))
                all_datasets.append(dataset)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
        return ConcatDataset(all_datasets)

    def _process_dataset(self, dataset: ConcatDataset) -> ConcatDataset:
        """Process the dataset according to configuration settings."""
        if not self.config.get('use_full', True):
            percent_to_use = self.config.get('use_percent', 0.1)
            if not (0.0 < percent_to_use <= 1.0):
                raise ValueError("The 'use_percent' must be a float between 0 and 1.")
            subset_length = int(len(dataset) * percent_to_use)
            indices = list(range(subset_length))
            return Subset(dataset, indices)
        return dataset

    def _split_dataset(self, dataset: ConcatDataset) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        """Split the dataset into train/dev or train/dev/test."""
        if self.config.get('shuffle', True):
            generator = torch.Generator().manual_seed(self.config['seed'])
            indices = torch.randperm(len(dataset), generator=generator).tolist()
            dataset = Subset(dataset, indices)

        if self.config['split_type'] == "train,dev":
            return self._split_train_dev(dataset)
        elif self.config['split_type'] == "train,dev,test":
            return self._split_train_dev_test(dataset)
        else:
            raise ValueError(f"Invalid split_type: {self.config['split_type']}")

    def _split_train_dev(self, dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Split dataset into train and dev sets."""
        train_ratio = self.config['split_ratios'][0]
        train_size = int(len(dataset) * train_ratio)
        
        train_dataset = Subset(dataset, range(train_size))
        dev_dataset = Subset(dataset, range(train_size, len(dataset)))
        
        return train_dataset, dev_dataset

    def _split_train_dev_test(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """Split dataset into train, dev, and test sets."""
        train_ratio = self.config['split_ratios'][0]
        dev_ratio = self.config['split_ratios'][1]
        
        train_size = int(len(dataset) * train_ratio)
        dev_size = int(len(dataset) * (train_ratio + dev_ratio))
        
        train_dataset = Subset(dataset, range(train_size))
        dev_dataset = Subset(dataset, range(train_size, dev_size))
        test_dataset = Subset(dataset, range(dev_size, len(dataset)))
        
        return train_dataset, dev_dataset, test_dataset

    def _create_loader(self, dataset: Dataset, batch_size_key: str) -> DataLoader:
        """Create a DataLoader with the specified configuration."""
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
            shuffle=(batch_size_key == 'train_batch_size'),  # Only shuffle training data
            **base_config
        )