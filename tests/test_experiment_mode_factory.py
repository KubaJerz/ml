import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ExperimentRunner import ExperimentModeFactory
from src.modes.SingleMode import SingleMode
import pytest
import re

class TestExperimentModeFactory:
    @pytest.fixture
    def sample_config(self):
        config = {
            'experiment': {
                'name': 'eeg_classification',
                'mode': 'single',
                'output_dir': 'experiments'
            },
            'data': {
                'absolute_path': '/kuba/Docs/model.py',
                'script_name': 'EEGDataScript',
                'split_type': 'train, test',
                'split_ratios': [0.7, 0.3],
                'shuffle': 'True',
                'seed': 69,
                'input_size': 5000,
                'input_channels': 1,
                'output_size': 3,
                'num_classes': 3
            },
            'model': {
                'architecture': 'ConvNet',
                'absolute_path': '/kuba/Docs/model.py'
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
                'optimizer': 'Adam',
                'learning_rate': 0.001,
                'criterion': 'CrossEntropyLoss',
                'device': 'cuda',
                'save_full_model': 'True',
                'best_f1': 'True',
                'best_loss': 'True',
                'early_stopping': 'False',
                'early_stopping_patience': 10,
                'early_stopping_monitor': 'dev_loss'
            }
        }   
        return config
    
    def test_verify_valid_mode(self, sample_config):
        factory = ExperimentModeFactory(sample_config)
        try: 
            factory._verify_valid_mode('single')
        except Exception as e:
            pytest.fail(f'Raised exception {e}')

    def test_fail_verify_valid_mode(self, sample_config):
        invaild_mode = 'no vaild'
        factory = ExperimentModeFactory(sample_config)
        with pytest.raises(TypeError, match=re.escape(f"Invalid mode type '{invaild_mode}'. Must be one of: {list(factory._valid_modes.keys())}")):
            factory._verify_valid_mode(invaild_mode)

    def test_create_mode(self, sample_config):
        factory = ExperimentModeFactory(sample_config)
        mode = factory.create_mode()
        assert isinstance(mode, SingleMode) == True

    def test_capital_create_mode(self, sample_config):
        sample_config['experiment']['mode'] = 'Single'
        factory = ExperimentModeFactory(sample_config)
        mode = factory.create_mode()
        assert isinstance(mode, SingleMode) == True

    def test_fail_create_mode(self, sample_config):
        invaild_mode = 'tt wrong'
        sample_config['experiment']['mode'] = invaild_mode
        factory = ExperimentModeFactory(sample_config)
        with pytest.raises(TypeError, match=re.escape(f"Invalid mode type '{invaild_mode}'. Must be one of: {list(factory._valid_modes.keys())}")):
            mode = factory.create_mode()
            assert isinstance(mode, SingleMode) == False
