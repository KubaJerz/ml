from .TrainLoopStrategy import TrainLoopStrategy
from typing import Dict, Any
from torcheval.metrics.functional import multiclass_f1_score
from tqdm import tqdm
import torch

class TrainingLoop(TrainLoopStrategy):
    """Training loop without validation."""
    def fit(self) -> Dict[str, float]:        
        if not self._call_callbacks('on_training_start', training_loop=self):
            return self.metrics
        
        self.model = self.model.to(self.device)

        for epoch in tqdm(range(self.total_epochs), desc='Progress: '):
            self.current_epoch = epoch
            
            if not self._call_callbacks('on_epoch_start', training_loop=self):
                break
            
            self.model.train()
            epoch_loss = 0.0
            epoch_f1 = 0.0
            
            # training loop
            for X_batch, y_batch in self.train_loader:
                if not self._call_callbacks('on_batch_start', training_loop=self, batch=(X_batch, y_batch)):
                    break

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                out = self.model.forward(X_batch)
                if len(y_batch.shape) > 1:  # Check if one-hot encoded
                    y_batch = torch.argmax(y_batch, dim=1)

                loss = self.criterion(out, y_batch)
                f1 = multiclass_f1_score(out, y_batch, num_classes=self.model.num_classes, average="macro").item()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                epoch_f1 += f1

                batch_metrics = {
                    'loss': loss.item(),
                    'f1': f1
                }
                if not self._call_callbacks('on_batch_end', training_loop=self, batch_metrics=batch_metrics):
                    break
            self.metrics['train_loss'].append(epoch_loss / len(self.train_loader))
            self.metrics['train_f1'].append(epoch_f1 / len(self.train_loader))

            # dev loop
            test_epoch_loss = 0.0
            test_epoch_f1 = 0.0
            self.model.eval()
            with torch.no_grad():
                for X_batch, y_batch in self.dev_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    devout = self.model(X_batch)
                    if len(y_batch.shape) > 1:  # Check if one-hot encoded
                        y_batch = torch.argmax(y_batch, dim=1)
                        
                    dev_loss = self.criterion(devout, y_batch).item()
                    dev_f1 = multiclass_f1_score(devout, y_batch, num_classes=self.model.num_classes, average="macro").item()
                    test_epoch_loss += dev_loss
                    test_epoch_f1 += dev_f1
            self.metrics['dev_loss'].append(test_epoch_loss / len(self.dev_loader))
            self.metrics['dev_f1'].append(test_epoch_f1 / len(self.dev_loader))

            
            if not self._call_callbacks('on_epoch_end', training_loop=self):
                break

        # this is done in the eval stage after the hyper parameter search
        # if self.test_loader:
        #     test_loss = 0.0
        #     test_f1 = 0.0
        #     test_predictions = []
        #     test_targets = []
        #     self.model.eval()

        #     for X_batch, y_batch in self.test_loader:
        #             X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
        #             test_logits = self.model(X_batch)
        #             loss = self.criterion(test_logits, y_batch).item()
        #             f1 = multiclass_f1_score(test_logits, torch.argmax(y_batch, dim=1),num_classes=self.model.num_classes, average="macro").item()
    
        #             test_loss += loss
        #             test_f1 += f1
                    
        #             test_predictions.extend(torch.argmax(test_logits, dim=1).cpu().numpy())
        #             test_targets.extend(torch.argmax(y_batch, dim=1).cpu().numpy())
            
        #     self.metrics['test_loss'] = test_loss / len(self.test_loader)
        #     self.metrics['test_f1'] = test_f1 / len(self.test_loader)
        #     self.metrics['test_predictions'] = test_predictions
        #     self.metrics['test_targets'] = test_targets
            

        if not self._call_callbacks('on_training_end', training_loop=self):
            return self.metrics

        return self.metrics