from Callback import Callback
from ....src.utils.logging_utils import save_metrics, save_model


class BestLossCallback(Callback):
   """Saves model when loss improves."""

   def __init__(self, metric_to_monitor='dev_loss'):
       self.best_loss = float('inf')
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
        loss = metrics.get(self.metric_to_monitor, None)
        if loss is None:
            raise KeyError(f'"{self.metric_to_monitor}" not valid key in metrics')
        if len(loss) < 1:
            raise IndexError(f'loss is of len less the one')
        current_loss = loss[-1]

        best_loss = training_loop.metrics.get(f'best_{self.metric_to_monitor}', None)
        if best_loss is None:
            raise KeyError(f'"best_{self.metric_to_monitor}" not valid key in metrics')
        
        if current_loss < best_loss:
            self.best_loss = current_loss
            save_model(model=training_loop.model, metrics=metrics, name=f'best_{self.metric_to_monitor}')
        return True
       
   def on_training_end(self, training_loop=None) -> bool:
       return True
   


