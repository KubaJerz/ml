from abc import ABC, abstractmethod

class Callback(ABC):
    """Base callback interface defining the training hooks."""
    @abstractmethod
    def on_training_start(self, training_loop: 'TrainLoopStrategy', model: nn.Module, datamodule: Any) -> None:
        pass
    
    @abstractmethod
    def on_epoch_start(self, training_loop: 'TrainLoopStrategy', epoch: int) -> None:
        pass
    
    @abstractmethod
    def on_batch_start(self, training_loop: 'TrainLoopStrategy', batch: Any) -> None:
        pass
    
    @abstractmethod
    def on_batch_end(self, training_loop: 'TrainLoopStrategy', metrics: Dict[str, float]) -> None:
        pass
    
    @abstractmethod
    def on_epoch_end(self, training_loop: 'TrainLoopStrategy', metrics: Dict[str, float]) -> bool:
        pass
    
    @abstractmethod
    def on_training_end(self, training_loop: 'TrainLoopStrategy') -> None:
        pass