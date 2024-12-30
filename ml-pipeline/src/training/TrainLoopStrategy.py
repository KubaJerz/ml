from abc import ABC, abstractmethod
import torch.nn as nn

"""Abstract base class for training loop strategies."""
class TrainLoopStrategy(ABC):
    def __init__(self, model, optimizer, criterion, logger, callbacks, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.callbacks = callbacks
        self.device = device
        self.current_epoch = 0
        
    @abstractmethod
    def fit(self, train_data, val_data) -> Dict[str, float]:
        """Main training loop"""
        pass
        
    def save_checkpoint(self, metrics):
        """Save a training checkpoint."""
        self.logger.log_model(self.model,metrics,f'checkpoint_epoch_{self.current_epoch}')
        
    def _call_callbacks(self, hook_name: str, *args, **kwargs) -> bool:
        """Call the specified hook on all callbacks."""
        continue_training = True
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            result = hook(self, *args, **kwargs)
            if result is False:  # Early stopping
                continue_training = False
        return continue_training