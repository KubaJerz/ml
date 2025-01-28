from ml_framework.modes.SingleMode import SingleMode
from ml_framework.callbacks import (
    BestMetricCallback, PlotCombinedMetrics, 
    EarlyStoppingCallback, TrainingCompletionCallback
)
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

class TestSingleMode:
    @pytest.fixture
    def sample_single_mode(self, samp_good_config):
        (Path(samp_good_config['experiment']['project_root']) / "experiments").mkdir()
        yield SingleMode(samp_good_config)
    
    @pytest.fixture
    def sample_dataloaders(self):
        train_loader = Mock()
        dev_loader = Mock()
        test_loader = Mock()
        return [train_loader, dev_loader, test_loader]

    @pytest.fixture
    def sample_model(self):
        return nn.Linear(10, 2)

    def test_validate_mode_specific_config(self, sample_single_mode):
        assert sample_single_mode.validate_mode_specific_config_structure() is True

        sample_single_mode.config['experiment']['mode'] = 'invalid'
        with pytest.raises(ValueError, match="Mode must be 'single'"):
            sample_single_mode.validate_mode_specific_config_structure()
            
    def test_model_initialization_errors(self, sample_single_mode):
        with pytest.raises(ValueError, match=f"Failed to import module from '{sample_single_mode.config['model']['absolute_path']}': No module named 'temp_model'"):
            sample_single_mode._setup_model()

    def test_setup_callbacks(self, sample_single_mode):
        """Test callback setup with different configurations"""
        metrics = {
            'best_dev_loss': 0.5,
            'best_dev_f1': 0.8
        }
        
        callbacks = sample_single_mode._setup_callbacks(sample_single_mode.config['callbacks'], metrics)
        
        # check all expected callbacks are here
        callback_types = [type(cb) for cb in callbacks]
        assert EarlyStoppingCallback.EarlyStoppingCallback not in callback_types
        assert BestMetricCallback.BestMetricCallback in callback_types
        assert PlotCombinedMetrics.PlotCombinedMetrics in callback_types
        assert TrainingCompletionCallback.TrainingCompletionCallback in callback_types
        
        # Test with early stopping
        # sample_single_mode.config['callbacks']['early_stopping'] = True
        # callbacks = sample_single_mode._setup_callbacks(sample_single_mode.config['callbacks'], metrics)
        # assert EarlyStoppingCallback.EarlyStoppingCallback in [type(cb) for cb in callbacks]

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

    def test_get_training_parameters_custom(self, sample_single_mode):
        training_params = sample_single_mode._get_training_parameters()
        
        assert training_params['total_epochs'] == sample_single_mode.config['training']['epochs']
        assert training_params['device'] == sample_single_mode.config['training']['device']
        assert training_params['save_dir'] == sample_single_mode.dir
        assert training_params['save_full_model'] == sample_single_mode.config['training']['save_full_model']

    def test_get_training_parameters_defaults(self, sample_single_mode):
        # Remove custom settings to test defaults
        sample_single_mode.config['training'] = {}
        
        training_params = sample_single_mode._get_training_parameters()
        
        assert training_params['total_epochs'] == 100
        assert training_params['device'] in ['cuda', 'cpu']
        assert training_params['save_full_model'] is True

    def test_initialize_metrics(self, sample_single_mode):
        """Test metrics initialization"""
        metrics = sample_single_mode._initialize_metrics()
        
        assert 'train_loss' in metrics
        assert 'dev_loss' in metrics
        assert 'train_f1' in metrics
        assert 'dev_f1' in metrics
        assert metrics['best_dev_f1'] == float('-inf')
        assert metrics['best_dev_loss'] == float('inf')
        assert all(isinstance(metric, list) for metric in [
            metrics['train_loss'], metrics['dev_loss'],
            metrics['train_f1'], metrics['dev_f1']
        ])

    def test_initialize_training_components(self, sample_single_mode, sample_model):
        """Test initialization of all training components"""
        components = sample_single_mode._initialize_training_components(sample_model)
        
        assert 'optimizer' in components
        assert 'criterion' in components
        assert 'callbacks' in components
        assert 'metrics' in components
        
        assert isinstance(components['optimizer'], optim.Optimizer)
        assert isinstance(components['criterion'], nn.modules.loss._Loss)
        assert isinstance(components['callbacks'], list)
        assert isinstance(components['metrics'], dict)

    def test_setup_training_with_two_loaders(self, sample_single_mode, sample_model):
        """Test training setup with train and dev loaders"""
        dataloaders = [Mock(), Mock()]  # train and dev loaders
        
        with patch('ml_framework.modes.SingleMode.TrainingLoop') as mock_training_loop:
            training_loop = sample_single_mode._setup_training(sample_model, dataloaders)
            
            mock_training_loop.assert_called_once()
            call_args = mock_training_loop.call_args[1]
            assert 'train_loader' in call_args
            assert 'dev_loader' in call_args
            assert call_args['test_loader'] is None

    def test_setup_training_with_three_loaders(self, sample_single_mode, sample_model):
        """Test training setup with 3 loaders"""
        dataloaders = [Mock(), Mock(), Mock()] 
        
        with patch('ml_framework.modes.SingleMode.TrainingLoop') as mock_training_loop:
            training_loop = sample_single_mode._setup_training(sample_model, dataloaders)
            
            mock_training_loop.assert_called_once()
            call_args = mock_training_loop.call_args[1]
            assert 'train_loader' in call_args
            assert 'dev_loader' in call_args
            assert call_args['test_loader'] is not None

    def test_setup_training_invalid_loader_count(self, sample_single_mode, sample_model):
        """Test training setup with invalid number of dataloaders"""
        dataloaders = [Mock()]  # just one loader
        
        with pytest.raises(ValueError, match="Expected 2 or 3 dataloaders, got 1"):
            sample_single_mode._setup_training(sample_model, dataloaders)

    def test_create_training_loop_components(self, sample_single_mode, sample_model, sample_dataloaders):
        """Test that training loop is created with all necessary components"""
        training_params = sample_single_mode._get_training_parameters()
        components = sample_single_mode._initialize_training_components(sample_model)
        
        training_loop = sample_single_mode._create_training_loop(
            model=sample_model,
            dataloaders=sample_dataloaders,
            training_params=training_params,
            **components
        )
        
        assert hasattr(training_loop, 'model')
        assert hasattr(training_loop, 'optimizer')
        assert hasattr(training_loop, 'criterion')
        assert hasattr(training_loop, 'metrics')
        assert hasattr(training_loop, 'callbacks')
        assert hasattr(training_loop, 'train_loader')
        assert hasattr(training_loop, 'dev_loader')
        assert hasattr(training_loop, 'test_loader')
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_setup_training_cuda_device(self, mock_cuda, sample_single_mode, sample_model):
        sample_single_mode.config['training']['device'] = 'cuda'
        dataloaders = [Mock(), Mock()]

        training_loop = sample_single_mode._setup_training(sample_model, dataloaders)
        assert training_loop.device == 'cuda'

    @patch('torch.cuda.is_available', return_value=False)
    def test_setup_training_cpu_fallback(self, mock_cuda, sample_single_mode, sample_model):
        # Ensure the config device is set to 'cuda', but we expect the fallback to 'cpu'
        sample_single_mode.config['training']['device'] = 'cuda'
        dataloaders = [Mock(), Mock()]
        
        with pytest.raises(ValueError, match="CUDA device requested but CUDA is not available"):
            training_loop = sample_single_mode._setup_training(sample_model, dataloaders)
 