from Callback import Callback
from ....src.utils.logging_utils import save_metrics, save_model


class BestLossCallback(Callback):
   """Saves model when loss improves."""

   def __init__(self, metric_to_monitor='dev_loss'):
       self.best_loss = float('inf')
       self.metric_to_monitor = metric_to_monitor
       
   def on_training_start(self, training_loop=None, datamodule=None) -> bool:
       return True
       
   def on_epoch_start(self, training_loop=None, metrics=None) -> bool:
       return True
       
   def on_batch_start(self, training_loop=None, batch=None) -> bool:
       return True
       
   def on_batch_end(self, training_loop=None, batch_metrics=None) -> bool:
       return True
       
   def on_epoch_end(self, training_loop=None, metrics=None) -> bool:
        lossi, devlossi, f1i, devf1i, best_f1_dev, best_loss_dev, train_id, save_dir):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(lossi, label='Train Loss')
        plt.plot(devlossi, label='Validation Loss')
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.axhline(best_loss_dev, color='g', linestyle='--', label=f'Best Dev Loss: {best_loss_dev:.3f}')
        #plt.text(0, best_loss_dev - 0.02, f'Best Loss: {best_loss_dev:.4f}', color='g', fontsize=10)
        plt.legend()
        
        # Plot F1 scores
        plt.subplot(1, 2, 2)
        plt.plot(f1i, label='Train F1')
        plt.plot(devf1i, label='Validation F1')
        plt.title('F1 Score vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.axhline(best_f1_dev, color='g', linestyle='--', label=f'Best Dev F1: {best_f1_dev:.3f}')
        #plt.text(0, best_f1_dev - 0.02, f'Best F1: {best_f1_dev:.4f}', color='g', fontsize=10)
        plt.legend()

        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,f"{train_id}_metrics.png"))
        plt.close()
       
   def on_training_end(self, training_loop=None, metrics=None) -> bool:
       return True
   


