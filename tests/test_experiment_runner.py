import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ExperimentRunner import ExperimentRunner
import pytest
import yaml


class TestExperimentRunner:
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
        config_file = './temp_config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        return config_file


    def test_load_config_success(self, sample_config) -> None:
        experiment_runner = ExperimentRunner(sample_config)
        assert isinstance(experiment_runner.config, dict)
        assert all(key in experiment_runner.config for key in ['experiment', 'data', 'model', 'parameters', 'training'])
        
        # Test specific values
        assert experiment_runner.config['experiment']['name'] == 'eeg_classification'
        assert experiment_runner.config['data']['input_size'] == 5000
        assert experiment_runner.config['parameters']['layer_depth'] == [4, 8, 16, 32]
        assert experiment_runner.config['training']['epochs'] == 100
        assert experiment_runner.config['model']['architecture'] == 'ConvNet'
        
    def test_nonexist_yaml(self):
        with pytest.raises(FileNotFoundError):
            experiment_runner = ExperimentRunner('./fake_config.yaml')

    def test_init(self, sample_config) -> None:
        experiment_runner = ExperimentRunner(sample_config)
        assert experiment_runner.config_path == './temp_config.yaml'


# import pytest
# from unittest.mock import Mock, patch, mock_open
# from pathlib import Path
# import yaml



# @pytest.fixture
# def valid_config():
#     return {
#         'experiment': {
#             'name': 'test_experiment',
#             'mode': 'single',
#             'version': '1.0'
#         },
#         'data': {
#             'type': 'EEGDataset',
#             'input_channels': 1,
#             'output_size': 3,
#             'num_classes': 3
#         },
#         'model': {
#             'architecture': 'ConvNet',
#             'parameters': {
#                 'batch_size': 64,
#                 'learning_rate': 0.001
#             }
#         }
#     }

# @pytest.fixture
# def yaml_config_str(valid_config):
#     return yaml.dump(valid_config)

# class TestExperimentRunner:
#     def test_initialization(self, yaml_config_str):
#         """Test that ExperimentRunner initializes correctly with a valid config."""
#         with patch('builtins.open', mock_open(read_data=yaml_config_str)):
#             runner = ExperimentRunner('dummy_path.yaml')
#             assert isinstance(runner, ExperimentRunner)
#             assert runner.config_path == 'dummy_path.yaml'
#             assert isinstance(runner.config, dict)

#     def test_load_config_invalid_yaml(self):
#         """Test that loading invalid YAML raises appropriate exception."""
#         invalid_yaml = "invalid: yaml: content: - ["
#         with patch('builtins.open', mock_open(read_data=invalid_yaml)):
#             with pytest.raises(yaml.YAMLError):
#                 ExperimentRunner('dummy_path.yaml')

#     def test_load_config_file_not_found(self):
#         """Test that attempting to load non-existent file raises exception."""
#         with pytest.raises(FileNotFoundError):
#             ExperimentRunner('nonexistent_config.yaml')

#     @patch('ExperimentRunner.validate_core_config_structure')
#     @patch('ExperimentRunner.ExperimentModeFactory')
#     def test_run_successful_execution(self, mock_factory, mock_validate, valid_config):
#         """Test successful execution of the run method."""
#         # Setup mock mode
#         mock_mode = Mock()
#         mock_factory.create_mode.return_value = mock_mode

#         # Create runner with mocked config
#         with patch('builtins.open', mock_open(read_data=yaml.dump(valid_config))):
#             runner = ExperimentRunner('dummy_path.yaml')
#             runner.run()

#         # Verify all expected methods were called
#         mock_validate.assert_called_once_with(valid_config)
#         mock_factory.create_mode.assert_called_once_with(valid_config)
#         mock_mode.validate_mode_specific_config_structure.assert_called_once()
#         mock_mode.setup_experimant_dir.assert_called_once()
#         mock_mode.execute.assert_called_once()

#     @patch('ExperimentRunner.validate_core_config_structure')
#     def test_run_validation_error(self, mock_validate, valid_config):
#         """Test that validation error is handled appropriately."""
#         mock_validate.side_effect = ValueError("Invalid config structure")

#         with patch('builtins.open', mock_open(read_data=yaml.dump(valid_config))):
#             runner = ExperimentRunner('dummy_path.yaml')
#             with pytest.raises(ValueError, match="Invalid config structure"):
#                 runner.run()

#     @patch('ExperimentRunner.validate_core_config_structure')
#     @patch('ExperimentRunner.ExperimentModeFactory')
#     def test_run_mode_validation_error(self, mock_factory, mock_validate, valid_config):
#         """Test that mode-specific validation error is handled appropriately."""
#         mock_mode = Mock()
#         mock_mode.validate_mode_specific_config_structure.side_effect = ValueError("Invalid mode config")
#         mock_factory.create_mode.return_value = mock_mode

#         with patch('builtins.open', mock_open(read_data=yaml.dump(valid_config))):
#             runner = ExperimentRunner('dummy_path.yaml')
#             with pytest.raises(ValueError, match="Invalid mode config"):
#                 runner.run()

#     @patch('ExperimentRunner.validate_core_config_structure')
#     @patch('ExperimentRunner.ExperimentModeFactory')
#     def test_run_execution_error(self, mock_factory, mock_validate, valid_config):
#         """Test that execution error is handled appropriately."""
#         mock_mode = Mock()
#         mock_mode.execute.side_effect = RuntimeError("Execution failed")
#         mock_factory.create_mode.return_value = mock_mode

#         with patch('builtins.open', mock_open(read_data=yaml.dump(valid_config))):
#             runner = ExperimentRunner('dummy_path.yaml')
#             with pytest.raises(RuntimeError, match="Execution failed"):
#                 runner.run()

# if __name__ == '__main__':
#     pytest.main([__file__])