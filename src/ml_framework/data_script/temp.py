from typing import Tuple, List, Union, Set, Dict
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
import re
import warnings
from ..utils.validation_utils import validate_data_config
from .BaseDataScript import BaseDataScript

"""
Chapter 1: The Setup
This script handles EEG data loading with two possible paths:
- Path A: Regular data splitting (mixed rat data)
- Path B: Data leakage prevention (keeping each rat's data together)

The path is chosen based on the 'data_leakage' configuration parameter.
If data_leakage is True (default), we take Path A.
If data_leakage is False, we take Path B to prevent data leakage between splits.
"""

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
    """
    Chapter 2: The Main Character
    This class orchestrates the entire data loading process.
    It decides which path to take and manages the flow of data preparation.
    """
    
    def __init__(self, config: dict):
        """
        Setup phase: Initialize our script with configuration.
        Here we decide which path we'll take: regular splitting or leakage prevention.
        """
        self.config = config
        validate_data_config(config)
        self.data_path = Path(self.config['data_absolute_path'])
        self.prevent_data_leakage = not self.config.get('data_leakage', True)
        
        if not self.config.get('use_full', True) and not self.prevent_data_leakage:
            warnings.warn(
                "Using partial dataset without data leakage prevention may still cause data leakage. "
                "Consider enabling data_leakage prevention for more robust validation."
            )
           
    def get_datasets(self) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        """
        Chapter 3: The Fork in the Road
        This is where we choose our path:
        - Path A: Regular splitting (data_leakage: True)
        - Path B: Leakage prevention (data_leakage: False)
        """
        if self.prevent_data_leakage:
            return self._get_leakage_prevented_datasets()  # Path B
        else:
            full_dataset = self._load_datasets()  # Path A
            processed_dataset = self._process_dataset(full_dataset)
            return self._split_dataset(processed_dataset)

    def _get_rat_ids(self) -> List[str]:
        """
        Chapter 4: Path B - Part 1
        First step in data leakage prevention: identify all unique rat IDs.
        We look for files like '000_00.pt', '000_01.pt' and extract the '000' part.
        """
        pattern = r"(\d{3})_\d{2}\.pt"
        rat_ids = set()
        
        for file_path in self.data_path.glob('*.pt'):
            match = re.match(pattern, file_path.name)
            if match:
                rat_ids.add(match.group(1))
                
        return sorted(list(rat_ids))

    def _split_rat_ids(self, rat_ids: List[str]) -> Union[Tuple[List[str], List[str]], 
                                                         Tuple[List[str], List[str], List[str]]]:
        """
        Chapter 4: Path B - Part 2
        Split the rat IDs into groups before loading any data.
        This ensures each rat's data stays together in its assigned split.
        Maintains original rat order - shuffling happens after all data is loaded.
        """
        if self.config['split_type'] == "train,dev":
            train_size = int(len(rat_ids) * self.config['split_ratios'][0])
            return (
                rat_ids[:train_size],
                rat_ids[train_size:]
            )
        else:  # train,dev,test
            train_ratio, dev_ratio = self.config['split_ratios'][:2]
            train_size = int(len(rat_ids) * train_ratio)
            dev_size = int(len(rat_ids) * (train_ratio + dev_ratio))
            return (
                rat_ids[:train_size],
                rat_ids[train_size:dev_size],
                rat_ids[dev_size:]
            )

    def _load_rat_data(self, rat_ids: List[str]) -> ConcatDataset:
        """
        Chapter 4: Path B - Part 3
        Load all data files for a specific group of rats.
        This keeps each rat's data together in one split.
        Then shuffles the data within each split while maintaining data leakage prevention.
        """
        datasets = []
        for rat_id in rat_ids:
            rat_data_for_id = []
            rat_files = sorted(self.data_path.glob(f'{rat_id}_*.pt'))
            for file_path in rat_files:
                try:
                    dataset = EEGDataset(str(file_path))
                    rat_data_for_id.append(dataset)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    continue
                    
            # Combine all data for this rat
            if rat_data_for_id:
                combined_rat_data = ConcatDataset(rat_data_for_id)
                
                # Shuffle this rat's data if configured
                if self.config.get('shuffle', True):
                    generator = torch.Generator().manual_seed(self.config['seed'])
                    n_samples = len(combined_rat_data)
                    indices = torch.randperm(n_samples, generator=generator).tolist()
                    combined_rat_data = Subset(combined_rat_data, indices)
                    
                datasets.append(combined_rat_data)
                
        return ConcatDataset(datasets) if datasets else None

    def _get_leakage_prevented_datasets(self) -> Union[Tuple[Dataset, Dataset], 
                                                      Tuple[Dataset, Dataset, Dataset]]:
        """
        Chapter 4: Path B - The Complete Story
        Orchestrates the entire data leakage prevention process:
        1. Get all unique rat IDs
        2. Split the rat IDs into groups
        3. Load data for each group separately
        """
        rat_ids = self._get_rat_ids()
        split_ids = self._split_rat_ids(rat_ids)
        
        if self.config['split_type'] == "train,dev":
            train_ids, dev_ids = split_ids
            return (
                self._load_rat_data(train_ids),
                self._load_rat_data(dev_ids)
            )
        else:  # train,dev,test
            train_ids, dev_ids, test_ids = split_ids
            return (
                self._load_rat_data(train_ids),
                self._load_rat_data(dev_ids),
                self._load_rat_data(test_ids)
            )

    def _load_datasets(self) -> ConcatDataset:
        """
        Chapter 3: Path A - Part 1
        Load all datasets without concern for which rat they came from.
        This is the simple approach when we don't need to prevent data leakage.
        """
        all_datasets = []
        for file_path in tqdm(sorted(self.data_path.glob('*.pt')), desc='Loading datasets'):
            try:
                dataset = EEGDataset(str(file_path))
                all_datasets.append(dataset)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                continue
        return ConcatDataset(all_datasets)

    def get_data_loaders(self) -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        """
        Chapter 5: The Final Act
        Transform our datasets into DataLoaders for training.
        This works the same way regardless of which path we took.
        """
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

    def _process_dataset(self, dataset: ConcatDataset) -> ConcatDataset:
        """
        Chapter 3: Path A - Part 2
        Process the full dataset, potentially taking only a portion if configured.
        """
        if not self.config.get('use_full', True):
            percent_to_use = self.config.get('use_percent', 0.1)
            if not (0.0 < percent_to_use <= 1.0):
                raise ValueError("The 'use_percent' must be a float between 0 and 1.")
            subset_length = int(len(dataset) * percent_to_use)
            indices = list(range(subset_length))
            return Subset(dataset, indices)
        return dataset

    def _split_dataset(self, dataset: ConcatDataset) -> Union[Tuple[Dataset, Dataset], 
                                                             Tuple[Dataset, Dataset, Dataset]]:
        """
        Chapter 3: Path A - Part 3
        Split the mixed dataset into train/dev or train/dev/test portions.
        This is the simple splitting approach where rat data might be mixed between splits.
        """
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

    def _create_loader(self, dataset: Dataset, batch_size_key: str) -> DataLoader:
        """
        Epilogue: Creating DataLoaders
        Final preparation of data for training, configuring batch sizes and shuffling.
        """
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
