from .Callback import Callback
from utils.logging_utils import save_metrics, save_model

class BestF1Callback(Callback):
   """Saves model when F1 score improves."""
   
   def __init__(self, metric_to_monitor='dev_f1'):
       self.metric_to_monitor = metric_to_monitor

   def on_training_start(self, training_loop=None) -> bool:
       return True
       
   def on_epoch_start(self, training_loop=None) -> bool:
       return True
       
   def on_batch_start(self, training_loop=None, batch=None) -> bool:
       return True
       
   def on_batch_end(self, training_loop=None, batch_metrics=None) -> bool:
       return True
       
   def on_epoch_end(self, training_loop=None) -> bool:
        metrics = training_loop.metrics
        f1 = metrics.get(self.metric_to_monitor, None)
        if f1 is None:
            raise KeyError(f'"{self.metric_to_monitor}" not valid key in metrics')
        if len(f1) < 1:
            raise IndexError(f'f1 is of len less the one')
        current_f1 = f1[-1]

        best_f1 = training_loop.metrics.get(f'best_{self.metric_to_monitor}', None)
        if best_f1 is None:
            raise KeyError(f'"best_{self.metric_to_monitor}" not valid key in metrics')
        
        if current_f1 > best_f1:
            self.best_f1 = current_f1
            save_model(model=training_loop.model, metrics=metrics, name=f'best_{self.metric_to_monitor}', save_dir=training_loop.save_dir , save_full_model=training_loop.save_full_model)
        return True
   
   def on_training_end(self, training_loop=None) -> bool:
       return True