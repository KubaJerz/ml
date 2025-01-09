import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import matplotlib.pyplot as plt
from ml_framework.training.callbacks.BestMetricCallback import BestMetricCallback
from ml_framework.training.callbacks.EarlyStoppingCallback import EarlyStoppingCallback
from ml_framework.training.callbacks.PlotCombinedMetrics import PlotCombinedMetrics
from ml_framework.training.callbacks.TrainingCompletionCallback import TrainingCompletionCallback

class MockTrainingLoop:
    def __init__(self, metrics=None, current_epoch=0, total_epochs=10, save_dir=None, model=None, save_full_model=True):
        self.metrics = metrics or {}
        self.model = model
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        self.save_dir = save_dir
        self.save_full_model = save_full_model

class TestBestMetricCallback:
    def test_flow_control(self, setup_for_fake_callbacks):
        """Test that callback can control training flow"""
        callback = BestMetricCallback(best_value=float('inf'))
        training_loop = MockTrainingLoop(**setup_for_fake_callbacks)
        
        # All hooks should return True to continue training
        assert callback.on_training_start(training_loop) is True
        assert callback.on_epoch_start(training_loop) is True
        assert callback.on_batch_start(training_loop) is True
        assert callback.on_batch_end(training_loop) is True
        assert callback.on_epoch_end(training_loop) is True
        assert callback.on_training_end(training_loop) is True

    def test_state_management(self, setup_for_fake_callbacks):
        """Test callback state management"""
        callback = BestMetricCallback(best_value=0.5, metric_to_monitor='dev_loss')
        training_loop = MockTrainingLoop(**setup_for_fake_callbacks)
        
        callback.on_epoch_end(training_loop)
        assert callback.best_val == 0.3 
        assert training_loop.metrics['best_dev_loss'] == 0.3

    @patch('ml_framework.training.callbacks.BestMetricCallback.save_model')
    def test_save_called_once_management(self, mock_save, setup_for_fake_callbacks):
        callback = BestMetricCallback(best_value=0.5, metric_to_monitor='dev_loss')
        training_loop = MockTrainingLoop(**setup_for_fake_callbacks)
        
        callback.on_epoch_end(training_loop)
        mock_save.assert_called_once()

class TestEarlyStoppingCallback:

    def test_patience_mechanism(self, setup_for_fake_callbacks):
        """Test early stopping patience mechanism"""
        callback = EarlyStoppingCallback(best_val_so_far=0.3, patience=2)
        training_loop = MockTrainingLoop(**setup_for_fake_callbacks)
        
        #no improvement
        assert callback.on_epoch_end(training_loop) is True
        assert callback.counter == 1
        
        #no improvement
        assert callback.on_epoch_end(training_loop) is True
        assert callback.counter == 2
        
        #should stop training
        assert callback.on_epoch_end(training_loop) is False

    @patch('ml_framework.training.callbacks.EarlyStoppingCallback.save_model')
    def test_model_saving_on_stop(self, mock_save, setup_for_fake_callbacks):
        """Test model saving when early stopping triggers"""
        callback = EarlyStoppingCallback(best_val_so_far=0.3, patience=0)
        training_loop = MockTrainingLoop(**setup_for_fake_callbacks)
        
        callback.on_epoch_end(training_loop)
        mock_save.assert_called_once()

class TestPlotCombinedMetrics:

    def test_plot_creation(self, setup_for_fake_PlotCombinedMetrics):
        """Test that plots are created correctly"""
        callback = PlotCombinedMetrics(plot_live=True)
        training_loop = MockTrainingLoop(**setup_for_fake_PlotCombinedMetrics)
        
        callback.on_epoch_end(training_loop)
        assert (Path(setup_for_fake_PlotCombinedMetrics['save_dir']) / 'metrics.png').exists()

    def test_live_plotting_control(self, setup_for_fake_PlotCombinedMetrics):
        """Test live plotting control based on epochs"""
        callback = PlotCombinedMetrics(plot_live=False)
        training_loop = MockTrainingLoop(**setup_for_fake_PlotCombinedMetrics)
        
        with patch('ml_framework.training.callbacks.PlotCombinedMetrics._plot') as mock_plot:
            callback.on_epoch_end(training_loop)
            mock_plot.assert_not_called()

class TestTrainingCompletionCallback:
    @patch('ml_framework.training.callbacks.TrainingCompletionCallback.save_model')
    def test_model_saving(self, mock_save, setup_for_fake_callbacks):
        """Test model saving at training completion"""
        callback = TrainingCompletionCallback()
        training_loop = MockTrainingLoop(**setup_for_fake_callbacks)
        
        callback.on_training_end(training_loop)
        mock_save.assert_called_once()