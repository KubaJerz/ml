from .Callback import Callback
from ..utils.logging_utils import save_model

class BestMetricCallback(Callback):
    
    def __init__(self, best_value,  metric_to_monitor='dev_loss'):
        self.best_val = best_value
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
        history = training_loop.metrics.get(self.metric_to_monitor, None)
        self._validate_history(history)
        current_val = history[-1]

        if self._current_better_than_best(current_val):
            self.best_val = current_val
            training_loop.metrics[f'best_{self.metric_to_monitor}'] = current_val
            save_model(model=training_loop.model, metrics=training_loop.metrics, name=f'best_{self.metric_to_monitor}', save_dir=training_loop.save_dir , save_full_model=training_loop.save_full_model)
            print(f"Best {self.metric_to_monitor} at epoch:{training_loop.current_epoch}")
        return True
       
    def on_training_end(self, training_loop=None) -> bool:
       return True
    
    def _current_better_than_best(self, val_to_compare) -> bool:
        try:
            if 'loss' in self.metric_to_monitor:
                return val_to_compare < self.best_val
            elif 'f1' in self.metric_to_monitor:
                return val_to_compare > self.best_val
        except Exception as e:
            raise ValueError(f" Unknow way to compare {self.metric_to_monitor}: {e}")
        
    def _validate_history(self, history):
        if history is None:
            raise KeyError(f'"{self.metric_to_monitor}" not valid metric_to_monitor in BestMetricCallback')
        if len(history) < 1:
            raise IndexError(f'history of {self.metric_to_monitor} is of len < 1')
        

