from .Callback import Callback
from ...utils.logging_utils import save_model

class EarlyStoppingCallback(Callback):
   def __init__(self, best_val_so_far, monitor='dev_loss', patience=3, min_delta=0.0):
       self.monitor = monitor
       self.patience = patience
       self.min_delta = min_delta
       self.counter = 0
       self.best_val_so_far = best_val_so_far
       
   def on_training_start(self, training_loop=None, datamodule=None) -> bool:
       return True
       
   def on_epoch_start(self, training_loop=None) -> bool:
       return True
       
   def on_batch_start(self, training_loop=None, batch=None) -> bool:
       return True
       
   def on_batch_end(self, training_loop=None, batch_metrics=None) -> bool:
       return True
       
   def on_epoch_end(self, training_loop=None) -> bool:
       #True means continue training
       #False means stop training
       metrics = training_loop.metrics
       current = metrics.get(f"best_{self.monitor}")
       if current is None:
           return True
           
       if 'loss' in self.monitor:
           improved = current < (self.best_val_so_far - self.min_delta)
       else:
           improved = current > (self.best_val_so_far + self.min_delta)
           
       if improved:
           self.best_val_so_far = current
           self.counter = 0
       else:
           self.counter += 1
           if self.counter >= self.patience:
                print(f'Early stopping triggered after {self.counter} epochs without improvement at epoch: {training_loop.current_epoch}')
                save_model(model=training_loop.model, metrics=metrics, name=f'early_stopping_epoch_{training_loop.current_epoch}', save_dir=training_loop.save_dir , save_full_model=training_loop.save_full_model)
                return False
       return True
       
   def on_training_end(self, training_loop=None) -> bool:
       return True