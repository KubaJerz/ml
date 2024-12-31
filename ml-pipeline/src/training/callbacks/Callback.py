from abc import ABC, abstractmethod
from ..TrainLoopStrategy import TrainLoopStrategy


''' When you impliment a callback you just write code for the part of the training you need (like on_epoch_start()) then the rest of the funtions you just'''
class Callback(ABC):
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