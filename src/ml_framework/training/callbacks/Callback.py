from abc import ABC, abstractmethod

"""
When implementing a callback, you only need to write code for the specific
stages of training you want to monitor/modify (like on_epoch_start()).

All callback methods return a boolean:
True: Continue the current loop or training process.
False: Break or exit based on the callback type:
    on_batch_start/on_batch_end: Stops processing the current epoch's batches and moves to the next epoch.
    on_epoch_start/on_epoch_end: Halts further epochs and moves to training completion.
    on_training_start/on_training_end: Immediately exits training entire
"""
class Callback(ABC):
    @abstractmethod
    def on_training_start(self, training_loop=None) -> bool:
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