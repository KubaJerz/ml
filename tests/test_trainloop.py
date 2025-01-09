import pytest
from ml_framework.training.TrainingLoop import TrainingLoop
import torch
from unittest.mock import Mock, patch

class TestTrainingLoop:
    def test_training_progression(self, base_trianloop_setup):
        """Test that training progresses correctly through epochs"""
        mock_callback = Mock()
        mock_callback.on_training_start = Mock(return_value=True)
        mock_callback.on_epoch_start = Mock(return_value=True)
        mock_callback.on_batch_start = Mock(return_value=True)
        mock_callback.on_batch_end = Mock(return_value=True)
        mock_callback.on_epoch_end = Mock(return_value=True)
        mock_callback.on_training_end = Mock(return_value=True)
        
        training_loop = TrainingLoop(
            model=base_trianloop_setup['model'],
            device=base_trianloop_setup['device'],
            optimizer=base_trianloop_setup['optimizer'],
            criterion=base_trianloop_setup['criterion'],
            metrics=base_trianloop_setup['metrics'],
            callbacks=[mock_callback],
            save_dir=base_trianloop_setup['save_dir'],
            train_loader=base_trianloop_setup['train_loader'],
            dev_loader=base_trianloop_setup['dev_loader'],
            total_epochs=2
        )
        
        metrics = training_loop.fit()
        
        assert len(metrics['train_loss']) == 2
        assert len(metrics['train_f1']) == 2
        assert mock_callback.on_training_start.called
        assert mock_callback.on_epoch_end.call_count == 2

    def test_metrics_tracking(self, base_trianloop_setup):
        training_loop = TrainingLoop(
            model=base_trianloop_setup['model'],
            device=base_trianloop_setup['device'],
            optimizer=base_trianloop_setup['optimizer'],
            criterion=base_trianloop_setup['criterion'],
            metrics=base_trianloop_setup['metrics'],
            callbacks=[],
            save_dir=base_trianloop_setup['save_dir'],
            train_loader=base_trianloop_setup['train_loader'],
            dev_loader=base_trianloop_setup['dev_loader'],
            total_epochs=1
        )
        
        metrics = training_loop.fit()
        
        assert 'train_loss' in metrics
        assert 'train_f1' in metrics
        assert 'dev_loss' in metrics
        assert 'dev_f1' in metrics
        assert all(isinstance(x, float) for x in metrics['train_loss'])
        assert all(0 <= x <= 1 for x in metrics['train_f1'])

    def test_device_placement(self, base_trianloop_setup):
        """Test that model and data are placed on correct device"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        training_loop = TrainingLoop(
            model=base_trianloop_setup['model'],
            device=device,
            optimizer=base_trianloop_setup['optimizer'],
            criterion=base_trianloop_setup['criterion'],
            metrics=base_trianloop_setup['metrics'],
            callbacks=[],
            save_dir=base_trianloop_setup['save_dir'],
            train_loader=base_trianloop_setup['train_loader'],
            dev_loader=base_trianloop_setup['dev_loader'],
            total_epochs=1
        )
        
        _ = training_loop.fit()
        
        assert next(training_loop.model.parameters()).device == device

    def test_gradient_calculation(self, base_trianloop_setup):
        """Test that gradients are calculated and applied correctly"""
        # Get initial parameters
        initial_params = [param.clone() for param in base_trianloop_setup['model'].parameters()]
        
        training_loop = TrainingLoop(
            model=base_trianloop_setup['model'],
            device=base_trianloop_setup['device'],
            optimizer=base_trianloop_setup['optimizer'],
            criterion=base_trianloop_setup['criterion'],
            metrics=base_trianloop_setup['metrics'],
            callbacks=[],
            save_dir=base_trianloop_setup['save_dir'],
            train_loader=base_trianloop_setup['train_loader'],
            dev_loader=base_trianloop_setup['dev_loader'],
            total_epochs=1
        )
        
        _ = training_loop.fit()
        
        current_params = [param.clone() for param in base_trianloop_setup['model'].parameters()]
        assert any(not torch.equal(i, c) for i, c in zip(initial_params, current_params)) # check that at least one has to be diffrent

    def test_callback_integration(self, base_trianloop_setup):
        """Test callback integration and early stopping"""
        class StopNowCallback:
            def on_batch_end(self, **kwargs):
                return True
            
            def on_training_start(self, **kwargs):
                return True
            
            def on_epoch_start(self, **kwargs):
                return True
            
            def on_batch_start(self, **kwargs):
                return True
            
            def on_epoch_end(self, **kwargs):
                return False
            
            def on_training_end(self, **kwargs):
                return True
        
        training_loop = TrainingLoop(
            model=base_trianloop_setup['model'],
            device=base_trianloop_setup['device'],
            optimizer=base_trianloop_setup['optimizer'],
            criterion=base_trianloop_setup['criterion'],
            metrics=base_trianloop_setup['metrics'],
            callbacks=[StopNowCallback()],
            save_dir=base_trianloop_setup['save_dir'],
            train_loader=base_trianloop_setup['train_loader'],
            dev_loader=base_trianloop_setup['dev_loader'],
            total_epochs=10
        )
        
        metrics = training_loop.fit()
        
        # make sure training stopped early so only one data point
        assert len(metrics['train_loss']) == 1

    def test_model_saving(self, base_trianloop_setup):
        """Test model saving behavior"""
        save_callback = Mock()
        save_callback.on_epoch_end = Mock(side_effect=lambda **kwargs: 
            torch.save(kwargs['training_loop'].model.state_dict(), 
                      base_trianloop_setup['save_dir'] / 'model.pth'))
        
        training_loop = TrainingLoop(
            model=base_trianloop_setup['model'],
            device=base_trianloop_setup['device'],
            optimizer=base_trianloop_setup['optimizer'],
            criterion=base_trianloop_setup['criterion'],
            metrics=base_trianloop_setup['metrics'],
            callbacks=[save_callback],
            save_dir=base_trianloop_setup['save_dir'],
            train_loader=base_trianloop_setup['train_loader'],
            dev_loader=base_trianloop_setup['dev_loader'],
            total_epochs=1
        )
        
        _ = training_loop.fit()
        
        # Verify model was saved
        assert (base_trianloop_setup['save_dir'] / 'model.pth').exists()