from ml_framework.modes.SingleMode import SingleMode
from ml_framework.training.callbacks import (
    BestMetricCallback, PlotCombinedMetrics, 
    EarlyStoppingCallback, TrainingCompletionCallback
)
import pytest
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from unittest.mock import Mock, patch

class TestSingleMode:
    @pytest.fixture
    def sample_single_mode(self, samp_good_config):
        return SingleMode(samp_good_config)

    def test_validate_mode_specific_config(self, sample_single_mode):
        assert sample_single_mode.validate_mode_specific_config_structure() is True

        sample_single_mode.config['experiment']['mode'] = 'invalid'
        with pytest.raises(ValueError, match="Mode must be 'single'"):
            sample_single_mode.validate_mode_specific_config_structure()
            
    def test_model_initialization_errors(self, sample_single_mode):
        """Test various model initialization error cases"""
        # Test import error
        with pytest.raises(ValueError, match="Failed to import model module"):
            sample_single_mode._setup_model()


    def test_setup_callbacks(self, sample_single_mode):
        """Test callback setup with different configurations"""
        metrics = {
            'best_dev_loss': 0.5,
            'best_dev_f1': 0.8
        }
        
        callbacks = sample_single_mode._setup_callbacks(sample_single_mode.config['callbacks'], metrics)
        
        # check all expected callbacks are their
        callback_types = [type(cb) for cb in callbacks]
        assert EarlyStoppingCallback.EarlyStoppingCallback not in callback_types
        assert BestMetricCallback.BestMetricCallback in callback_types
        assert PlotCombinedMetrics.PlotCombinedMetrics in callback_types
        assert TrainingCompletionCallback.TrainingCompletionCallback in callback_types
        
        # Test with early stopping
        sample_single_mode.config['callbacks']['early_stopping'] = True
        callbacks = sample_single_mode._setup_callbacks(sample_single_mode.config['callbacks'], metrics)
        assert EarlyStoppingCallback.EarlyStoppingCallback in [type(cb) for cb in callbacks]

    def test_create_optimizer_custom(self, sample_single_mode):
        """Test custom optimizer creation"""
        model = nn.Linear(10, 2)  # Simple model for testing
        optimizer = sample_single_mode._create_optimizer(model, sample_single_mode.config['training'])
        
        assert isinstance(optimizer, getattr(optim, sample_single_mode.config['training']['optimizer']))
        assert optimizer.param_groups[0]['lr'] == sample_single_mode.config['training']['learning_rate']

    def test_create_optimizer_default(self, sample_single_mode):
        """Test default optimizer creation"""
        model = nn.Linear(10, 2)
        training_config = {}
        optimizer = sample_single_mode._create_optimizer(model, training_config)
        
        assert isinstance(optimizer, optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001