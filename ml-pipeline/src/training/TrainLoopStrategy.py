from abc import ABC, abstractmethod
import torch.nn as nn
from ..utils.logging_utils import save_metrics, save_model

"""Abstract base class for training loop strategies."""
class TrainLoopStrategy(ABC):
    def __init__(self, model, optimizer, criterion, logger, callbacks, device, save_full_model=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks
        self.device = device
        self.current_epoch = 0
        self.save_full_model = save_full_model
        
    @abstractmethod
    def fit(self, train_data, val_data) -> Dict[str, float]:
        """Main training loop"""
        pass
        
    def save_checkpoint(self, metrics):
        save_model(self.model, metrics, f'checkpoint_epoch_{self.current_epoch}', self.save_full_model)
        
    def _call_callbacks(self, hook_name: str, *args, **kwargs) -> bool:
        continue_training = True
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            result = hook(self, *args, **kwargs)
            if result is False:
                continue_training = False
        return continue_training