from .Callback import Callback
from utils.logging_utils import save_model


class TrainingCompletionCallback(Callback):
    def on_training_start(self, training_loop=None) -> bool:
        pass
    
    def on_epoch_start(self, training_loop=None) -> bool:
        pass
    
    def on_batch_start(self, training_loop=None, batch=None) -> bool:
        pass
    
    def on_batch_end(self, training_loop=None, batch_metrics=None) -> bool:
        pass
    
    def on_epoch_end(self, training_loop=None) -> bool:
        pass
    
    def on_training_end(self, training_loop=None) -> bool:
        save_model(model=training_loop.model, metrics=training_loop.metrics, name=f'full', save_dir=training_loop.save_dir , save_full_model=training_loop.save_full_model)
