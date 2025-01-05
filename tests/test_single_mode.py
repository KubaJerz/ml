import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modes.SingleMode import SingleMode
import pytest
import torch.nn as nn
import torch.optim as optim

class TestSingleMode:
    
    @pytest.fixture
    def sample_single_mode(self):
        config = {
            'experiment': {
                'name': 'eeg_classification',
                'mode': 'single',
                'output_dir': 'experiments'
            },
            'data': {
                'absolute_path': '/kuba/Docs/data.py',
                'script_name': 'EEGDataScript',
                'split_type': 'train, test',
                'split_ratios': [0.7, 0.3],
                'shuffle': True,
                'seed': 69,
                'input_size': 5000,
                'input_channels': 1,
                'output_size': 3,
                'num_classes': 3
            },
            'model': {
                'architecture': 'SampleModel',
                'absolute_path': '/Users/kuba/projects/ml/tests/sample_model.py'
            },
            'parameters': {
                'hidden_blocks': 4,
                'layer_depth': [4, 8, 16, 32],
                'dropout_rate': 0.3,
                'activation': 'ReLU',
                'normalization': 'batch'
            },
            'training': {
                'epochs': 100,
                'train_batch_size': 64,
                'test_batch_size': -1,
                'optimizer': 'SGD',
                'learning_rate': 0.001,
                'criterion': 'CrossEntropyLoss',
                'device': 'cuda',
                'save_full_model': True,

                'best_f1': True,
                'best_loss': True,
                'early_stopping': False,
                'early_stopping_patience': 10,
                'early_stopping_monitor': 'dev_loss'
            }
        }  
        mode = SingleMode(config)
        return mode
    
    def test_setup_model(self, sample_single_mode):
        model = sample_single_mode._setup_model()
        assert isinstance(model, nn.Module)

    def test_model_arguments(self, sample_single_mode):
        model = sample_single_mode._setup_model()
        assert model.input_size == sample_single_mode.config['data']['input_size']
        assert model.input_channels == sample_single_mode.config['data']['input_channels']
        assert model.output_size == sample_single_mode.config['data']['output_size']
        assert model.num_classes == sample_single_mode.config['data']['num_classes']
        assert model.hyperparams == sample_single_mode.config['parameters']

    def test_fail_path_setup_model(self, sample_single_mode):
        fake_path = 'fak/path/fakemodel.py'
        sample_single_mode.config['model']['absolute_path'] = fake_path
        with pytest.raises(ValueError, match=f"Failed to import model module: No module named '{fake_path.split('/')[-1].split('.')[0]}'"):
            model = sample_single_mode._setup_model()

    def test_setup_callbacks(self, sample_single_mode):
        training_config = sample_single_mode.config['training']
        callbacks = sample_single_mode._setup_callbacks(training_config)

        from src.training.callbacks import BestF1Callback, BestLossCallback, PlotCombinedMetrics
        manual_list = [BestF1Callback.BestF1Callback(), BestLossCallback.BestLossCallback(), PlotCombinedMetrics.PlotCombinedMetrics()]
        assert all([type(x) is type(y) for x,y in zip(callbacks, manual_list)])


    def test_create_optimizer_custom(self, sample_single_mode):
        training_config = sample_single_mode.config['training']
        model = sample_single_mode._setup_model()
        optimizer = sample_single_mode._create_optimizer(model=model, training_config=training_config)
        assert optimizer.param_groups[0]['lr'] == training_config['learning_rate']

        expected_optim = getattr(optim, training_config['optimizer'])
        assert isinstance(optimizer, expected_optim)


    def test_create_optimizer_default(self, sample_single_mode):
        training_config = sample_single_mode.config['training']
        training_config.pop('optimizer')
        training_config.pop('learning_rate')
        model = sample_single_mode._setup_model()
        optimizer = sample_single_mode._create_optimizer(model=model, training_config=training_config)
        assert optimizer.param_groups[0]['lr'] == 0.001

        expected_optim = getattr(optim, 'Adam')
        assert isinstance(optimizer, expected_optim)
        

    def test_create_criterion_custom(self, sample_single_mode):
        training_config = sample_single_mode.config['training']
        criterion = sample_single_mode._create_criterion(training_config=training_config)

        expected_criterion = getattr(nn, training_config['criterion'])
        assert isinstance(criterion, expected_criterion)

    def test_create_criterion_default(self, sample_single_mode):
        training_config = sample_single_mode.config['training']
        training_config.pop('criterion')
        criterion = sample_single_mode._create_criterion(training_config=training_config)

        expected_criterion = getattr(nn, 'CrossEntropyLoss')
        assert isinstance(criterion, expected_criterion)