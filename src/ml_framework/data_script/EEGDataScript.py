from typing import Tuple, List, Union, Set, Dict
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from pathlib import Path
import re
import warnings
from ml_framework.utils.validation_utils import validate_data_config
from ml_framework.data_script.BaseDataScript import BaseDataScript

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
    """Implementation of data script for EEG data processing with data leakage prevention."""
    
    def __init__(self, config: dict):
        self.config = config
        validate_data_config(config)
        self.data_path = Path(self.config['data_absolute_path'])
        self.prevent_data_leakage = self.config.get('prevent_data_leakage', True)
        
        if not self.config.get('use_full', True) and self.prevent_data_leakage:
            warnings.warn(
                "Using partial dataset without data leakage prevention may still cause data leakage. "
                "Consider enabling data_leakage prevention for more robust validation.")
            
    '''
    If: we are to prevent data leakadge then we will scan the dir for unique rat id "000_*" or "001_*" then for 
    each one we will split on the rat ID's:

     If we have 10 rats and 70/30 split:
        Train rats: ["000", "001", "002", "003", "004", "005", "006"]
        Dev rats:   ["007", "008", "009"]

    Then we oncat each set of rats intoa datset and shuffel within the datset.

    Else: (with data leakadge) 
        We load whole dataset then shuffel then split.
    '''

    def get_datasets(self) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        """Get train/dev or train/dev/test dataset splits."""
        if self.prevent_data_leakage:
            return self._get_leakage_prevented_datasets()
        else:
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

    def _get_rat_ids(self) -> List[str]:
        """Get sorted list of unique rat IDs from the data directory."""
        pattern = r"(\d{3})_\d{1}\.pt"
        rat_ids = set()
        
        for file_path in self.data_path.glob('*.pt'):
            match = re.match(pattern, file_path.name)
            if match:
                rat_ids.add(match.group(1))
        return sorted(list(rat_ids))

    def _split_rat_ids(self, rat_ids: List[str]) -> Union[Tuple[List[str], List[str]], 
                                                         Tuple[List[str], List[str], List[str]]]:
        #We shuffel so that we dont get the same order of rats each time
        if self.config.get('shuffle', True):
            generator = torch.Generator().manual_seed(self.config['seed'])
            indices = torch.randperm(len(rat_ids), generator=generator).tolist()
            rat_ids = [rat_ids[i] for i in indices]

        if self.config['split_type'] == "train,dev":
            train_size = int(len(rat_ids) * self.config['split_ratios'][0])
            print(f'Train size: {len(rat_ids[:train_size])} Dev size: {len(rat_ids[train_size:])}')
            return (
                rat_ids[:train_size],
                rat_ids[train_size:]
            )
        else:  # train,dev,test
            train_ratio, dev_ratio = self.config['split_ratios'][:2]
            train_size = int(len(rat_ids) * train_ratio)
            dev_size = int(len(rat_ids) * (train_ratio + dev_ratio))
            print(f'Train size: {len(rat_ids[:train_size])} Dev size: {len(rat_ids[train_size:dev_size])} Test size: {len(rat_ids[dev_size:])}')
            return (
                rat_ids[:train_size],
                rat_ids[train_size:dev_size],
                rat_ids[dev_size:]
            )

    def _load_rat_data(self, rat_ids: List[str]) -> ConcatDataset:
        all_datasets = []
        for rat_id in rat_ids:
            rat_files = sorted(self.data_path.glob(f'{rat_id}_*.pt'))
            for file_path in rat_files:
                try:
                    dataset = EEGDataset(str(file_path))
                    all_datasets.append(dataset)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
                    continue

        combined_dataset = ConcatDataset(all_datasets) if all_datasets else None
        
        # we shuffle again so that within each dataset (train. dev .eg) the samples from each rat are not just sequantial
        if combined_dataset and self.config.get('shuffle', True):
            generator = torch.Generator().manual_seed(self.config['seed'])
            indices = torch.randperm(len(combined_dataset), generator=generator).tolist()
            combined_dataset = Subset(combined_dataset, indices)

        print(f'Dataset samples: {len(ConcatDataset(combined_dataset))}')
            
        return combined_dataset

    def _get_leakage_prevented_datasets(self) -> Union[Tuple[Dataset, Dataset], 
                                                      Tuple[Dataset, Dataset, Dataset]]:
        """Create datasets with data leakage prevention."""
        rat_ids = self._get_rat_ids()
        split_ids = self._split_rat_ids(rat_ids)
        
        if self.config['split_type'] == "train,dev":
            train_ids, dev_ids = split_ids
            print(f"Training on rat:{train_ids}\nDev on rats:{dev_ids}")
            return (
                self._load_rat_data(train_ids),
                self._load_rat_data(dev_ids)
            )
        else:  # train,dev,test
            train_ids, dev_ids, test_ids = split_ids
            print(f"Training on rat:{train_ids}\nDev on rats:{dev_ids}\nTesting on rats:{test_ids}")
            return (
                self._load_rat_data(train_ids),
                self._load_rat_data(dev_ids),
                self._load_rat_data(test_ids)
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

    def _split_dataset(self, dataset: ConcatDataset) -> Union[Tuple[Dataset, Dataset], 
                                                             Tuple[Dataset, Dataset, Dataset]]:
        #here we just shuffle it once since we combine all data, shuffle and then split after we can only shuffel once
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
        print(f'Train size: {len(train_dataset)} Dev size: {len(dev_dataset)}')
        
        
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
        print(f'Train size: {len(train_dataset)} Dev size: {len(dev_dataset)} Test size: {len(test_dataset)}')
        
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