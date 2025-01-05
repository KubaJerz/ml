from .Callback import Callback
from utils.logging_utils import save_metrics, save_model
import matplotlib.pyplot as plt
import os



class PlotCombinedMetrics(Callback):
   def __init__(self):
       pass

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
        save_dir = training_loop.save_dir

        if _is_even(training_loop.current_epoch) and _is_even(training_loop.total_epochs):
            _plot(metrics=metrics, save_dir=save_dir)
            return True
        elif not _is_even(training_loop.current_epoch) and not _is_even(training_loop.total_epochs):
            _plot(metrics=metrics, save_dir=save_dir)
            return True
        else:
            return True
       
   def on_training_end(self, training_loop=None) -> bool:
       return True
   
def _is_even(num):
    return (num % 2) == 0   

def _plot(metrics, save_dir):
        lossi, devlossi = metrics.get('train_loss'), metrics.get('dev_loss') 
        f1i, devf1i = metrics.get('train_f1'), metrics.get('dev_f1')
        best_f1_dev, best_loss_dev = metrics.get('best_f1_dev'), metrics.get('best_loss_dev')

        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(lossi, label='Train Loss')
        plt.plot(devlossi, label='Dev Loss')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.axhline(best_loss_dev, color='g', linestyle='--', label=f'Best Dev Loss: {best_loss_dev:.3f}')
        #plt.text(0, best_loss_dev - 0.02, f'Best Loss: {best_loss_dev:.4f}', color='g', fontsize=10)
        plt.legend()
        
        # Plot F1 scores
        plt.subplot(1, 2, 2)
        plt.plot(f1i, label='Train F1')
        plt.plot(devf1i, label='Dev F1')
        plt.title('F1 Score vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.axhline(best_f1_dev, color='g', linestyle='--', label=f'Best Dev F1: {best_f1_dev:.3f}')
        #plt.text(0, best_f1_dev - 0.02, f'Best F1: {best_f1_dev:.4f}', color='g', fontsize=10)
        plt.legend()

        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,f"metrics.png"))
        plt.close()


