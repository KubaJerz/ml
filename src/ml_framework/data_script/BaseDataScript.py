from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import Tuple, Union

class BaseDataScript(ABC):
    """
    Abstract interface for data script implementations.
    Defines the contract that all data scripts must follow.
    """
    
    @abstractmethod
    def get_datasets(self) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        """
        Get datasets based on configuration.
        
        Returns:
            If split_type is "train,dev":
                tuple: (train_dataset, dev_dataset)
            If split_type is "train,dev,test":
                tuple: (train_dataset, dev_dataset, test_dataset)
        """
        pass
    
    @abstractmethod
    def get_data_loaders(self) -> Union[Tuple[Dataset, Dataset], Tuple[Dataset, Dataset, Dataset]]:
        """
        Get data loaders based on configuration.
        
        Returns:
            If split_type is "train,dev":
                tuple: (train_loader, dev_loader)
            If split_type is "train,dev,test":
                tuple: (train_loader, dev_loader, test_loader)
        """
        pass