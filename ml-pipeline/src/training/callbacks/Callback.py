from abc import ABC, abstractmethod

"""
When implementing a callback, you only need to write code for the specific
stages of training you want to monitor/modify (like on_epoch_start()).

All callback methods return a boolean:
    - Return True to continue training
    - Return False to break out of the current loop
        - False in on_batch_start/end breaks the batch loop
        - False in on_epoch_start/end breaks the epoch loop
        - False in on_training_start/end exits training entirely
"""
class Callback(ABC):
    @abstractmethod
    def on_training_start(self, training_loop=None, datamodule=None) -> bool:
        pass
    
    @abstractmethod
    def on_epoch_start(self, training_loop=None) -> bool:
        pass
    
    @abstractmethod
    def on_batch_start(self, training_loop=None, batch=None) -> bool:
        pass
    
    @abstractmethod
    def on_batch_end(self, training_loop=None, batch_metrics=None) -> bool:
        pass
    
    @abstractmethod
    def on_epoch_end(self, training_loop=None) -> bool:
        pass
    
    @abstractmethod
    def on_training_end(self, training_loop=None) -> bool:
        pass