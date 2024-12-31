from abc import ABC, abstractmethod
from ..TrainLoopStrategy import TrainLoopStrategy

class Callback(ABC):
    """Base callback interface defining the training hooks."""
    @abstractmethod
    def on_training_start(self, training_loop: 'TrainLoopStrategy', model, datamodule) -> None:
        pass
    
    @abstractmethod
    def on_epoch_start(self, training_loop: 'TrainLoopStrategy', epoch) -> None:
        pass
    
    @abstractmethod
    def on_batch_start(self, training_loop: 'TrainLoopStrategy', batch) -> None:
        pass
    
    @abstractmethod
    def on_batch_end(self, training_loop: 'TrainLoopStrategy', metrics) -> None:
        pass
    
    @abstractmethod
    def on_epoch_end(self, training_loop: 'TrainLoopStrategy', metrics) -> bool:
        pass
    
    @abstractmethod
    def on_training_end(self, training_loop: 'TrainLoopStrategy') -> None:
        pass