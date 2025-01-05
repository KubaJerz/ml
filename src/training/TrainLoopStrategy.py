from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict
# from utils.validation_utils import validate_metrics_structure

"""Abstract base class for training loop strategies."""
class TrainLoopStrategy(ABC):
    def __init__(self, model, optimizer, criterion, total_epochs, callbacks, device, save_dir, train_loader, dev_loader, test_loader=None, save_full_model=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.save_full_model = save_full_model

        
    @abstractmethod
    def fit(self, dataloaders) -> Dict[str, float]:
        """Main training loop"""
        pass

    def _call_callbacks(self, function_name: str, **kwargs) -> bool:
        continue_training = True
        for callback in self.callbacks:
            hook = getattr(callback, function_name)
            result = hook(**kwargs)
            if result is False:
                continue_training = False
        return continue_training
    
