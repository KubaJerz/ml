from TrainLoopStrategy import TrainLoopStrategy

class NoValTrainingLoop(TrainLoopStrategy):
    """Training loop without validation."""
    def fit(self, train_data, val_data=None) -> Dict[str, float]:
        self._call_callbacks('on_training_start', self.model, train_data)
        
        metrics_history = []
        for epoch in range(self.current_epoch, self.current_epoch + self.max_epochs):
            self.current_epoch = epoch
            self._call_callbacks('on_epoch_start', epoch)
            
            self.model.train()
            epoch_loss = 0.0
            epoch_f1 = 0.0
            num_batches = 0
            
            for batch in train_data:
                self._call_callbacks('on_batch_start', batch)
                
                # Move batch to device
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Calculate F1 score
                f1 = multiclass_f1_score(
                    outputs, 
                    targets,
                    num_classes=self.model.num_classes,
                    average="macro"
                ).item()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                epoch_f1 += f1
                num_batches += 1
                
                batch_metrics = {
                    'loss': loss.item(),
                    'f1': f1
                }
                self._call_callbacks('on_batch_end', batch_metrics)
            
            # Calculate epoch metrics
            epoch_metrics = {
                'loss': epoch_loss / num_batches,
                'f1': epoch_f1 / num_batches,
                'epoch': epoch
            }
            
            # Log metrics
            self.logger.log_metrics(epoch_metrics, epoch)
            metrics_history.append(epoch_metrics)
            
            # Call epoch end callbacks
            if not self._call_callbacks('on_epoch_end', epoch_metrics):
                break
                
            # Save checkpoint
            self.save_checkpoint(epoch_metrics)
            
        self._call_callbacks('on_training_end')
        return metrics_history