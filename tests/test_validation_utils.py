import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import validation_utils
import pytest
import yaml

class TestExperimentModeFactory:
    global expected_mode
    expected_mode = 'single' 

    @pytest.fixture
    def sample_good_config(self):
        config = {
            'experiment': {
                'name': 'eeg_classification',
                'mode': expected_mode,
                'output_dir': 'experiments'
            },
            'data': {
                'absolute_path': '/Users/kuba/Documents/data/Raw/pt_ekyn_500hz',
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
                'save_full_model': True,
                'best_f1': True,
                'best_loss': True,
                'early_stopping': False,
                'early_stopping_patience': 10,
                'early_stopping_monitor': 'dev_loss'
            }
        }   
        return config
    
    @pytest.fixture
    def sample_bad_configs(self):
        return {
            'missing_section': {
                'data': {},  # Missing required sections
            },
            'invalid_mode': {
                'experiment': {'mode': 'invalid_mode'},
                'data': {},
                'model': {},
                'parameters': {},
                'training': {}
            },
            'data': {
                'split_ratios': [0.7, 0.4],  # Invalid ratios sum > 1
                'input_size': -1, # Invalid negative input size
                'model': {},
                'parameters': {},
                'training': {}
            },
            'invalid_split':{
                'absolute_path': '/Users/kuba/Documents/data/Raw/pt_ekyn_500hz',
                'script_name': 'EEGDataScript',
                'split_type': 'train, test',
                'split_ratios': [0.9, 0.3],
                'shuffle': True,
                'seed': 69,
                'input_size': 5000,
                'input_channels': 1,
                'output_size': 3,
                'num_classes': 3
            },
            'invalid_training': {
                'experiment': {'mode': expected_mode},
                'data': {},
                'model': {},
                'parameters': {},
                'training': {
                    'epochs': -1,  # Invalid negative epochs
                    'early_stopping_patience': -5  # Invalid negative patience
                }
            }
        }

        
    def test_validate_core_config_structure(self, sample_good_config):
        try:
            validation_utils.validate_core_config_structure(sample_good_config)
        except Exception as e:
            pytest.fail(f"Raised exception {e}")

    def test_validate_data_config(self, sample_good_config):
        try:
            validation_utils.validate_data_config(sample_good_config['data'])
        except Exception as e:
            pytest.fail(f"Raised exception {e}")

    def test_validate_training_config(self, sample_good_config):
        try:
            validation_utils.validate_training_config(sample_good_config['training'])
        except Exception as e:
            pytest.fail(f"Raised exception {e}")

    def test_validate_mode_config(self, sample_good_config):
        try:
            validation_utils.validate_mode_config(sample_good_config, expected_mode)
        except Exception as e:
            pytest.fail(f"Raised exception {e}")

    def test_validate_metrics_structure(self):
        good_metrics = {
        'train_loss': [49, 54, 32, 24],
        'dev_loss': [49, 54, 32, 24],
        'train_f1': [49, 54, 32, 24],
        'dev_f1': [49, 54, 32, 24],
        'best_f1_dev': 89.0,
        'best_loss_dev': 2.5,
        }

        val = validation_utils.validate_metrics_structure(good_metrics)
        assert val == True

    def test_validate_core_config_structure_fails(self, sample_bad_configs):
        with pytest.raises(ValueError, match="Missing experiment section"):
            validation_utils.validate_core_config_structure(sample_bad_configs)

    def test_validate_data_config_fails(self, sample_bad_configs):
        with pytest.raises(FileNotFoundError):
            validation_utils.validate_data_config(sample_bad_configs['data'])
        
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0, got 1.2"):
            validation_utils.validate_data_config(sample_bad_configs['invalid_split'])

    def test_validate_training_config_fails(self, sample_bad_configs):
        with pytest.raises(ValueError, match="Missing train_batch_size"):
            validation_utils.validate_training_config(sample_bad_configs['invalid_training']['training'])

    def test_validate_mode_config_fails(self, sample_bad_configs):
        with pytest.raises(ValueError, match="Mode must be 'single'"):
            validation_utils.validate_mode_config(sample_bad_configs['invalid_mode'], expected_mode)
