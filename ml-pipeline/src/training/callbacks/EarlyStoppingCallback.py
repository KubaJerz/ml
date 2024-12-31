from Callback import Callback

class EarlyStoppingCallback(Callback):
    def __init__(self, monitor = 'dev_loss', patience = 3, min_delta = 0.0):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = float('inf') if 'loss' in monitor else float('-inf')
        
    def on_training_start(self, training_loop, model, datamodule) -> None:
        pass
        
    def on_epoch_start(self, training_loop, epoch) -> None:
        pass
        
    def on_batch_start(self, training_loop, batch) -> None:
        pass
        
    def on_batch_end(self, training_loop, metrics) -> None:
        pass
        
    def on_epoch_end(self, training_loop, metrics) -> bool:
        #True means continune training
        #False means stop straining

        current = metrics.get(self.monitor)
        if current is None:
            return True
            
        if 'loss' in self.monitor:
            improved = current < (self.best_metric - self.min_delta)
        else:
            improved = current > (self.best_metric + self.min_delta)
            
        if improved:
            self.best_metric = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'Early stopping triggered after {self.counter} epochs without improvement at epoch: {training_loop.current_epoch}')
                return False
        return True
        
    def on_training_end(self, training_loop) -> None:
        pass
