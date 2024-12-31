from Callback import Callback

class BestLossCallback(Callback):
    """Saves model when loss improves."""

    def __init__(self, metric_to_monitor='dev_loss'):
        self.best_loss = float('inf')
        self.metric_to_monitor = metric_to_monitor
        
    def on_training_start(self, training_loop, model, datamodule) -> None:
        pass
        
    def on_epoch_start(self, training_loop, epoch) -> None:
        pass
        
    def on_batch_start(self, training_loop, batch) -> None:
        pass
        
    def on_batch_end(self, training_loop, metrics) -> None:
        pass
        
    def on_epoch_end(self, training_loop, metrics) -> bool:
        current_loss = metrics.get(self.metric_to_monitor, None)
        if current_loss is None:
            raise KeyError(f'"{self.metric_to_monitor}" not valid key in metrics')
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            training_loop.logger.log_model(model= training_loop.model, metrics=metrics, name=f'best_{self.metric_to_monitor}')
        return True
        
    def on_training_end(self, training_loop) -> None:
        pass