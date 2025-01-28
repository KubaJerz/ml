import pytest
from pathlib import Path
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, call
import tempfile
from ml_framework.modes.RandomSearchMode import RandomSearchMode
import random

class TestRandomSearchMode:
    @pytest.fixture
    def sample_search_config(self, samp_good_config):
        (Path(samp_good_config['experiment']['project_root']) / "experiments").mkdir()
        samp_good_config['experiment']['mode'] = 'random_search'
        samp_good_config['search_space'] = { 
            'learning_rate': {'min': 0.0001, 'max': 0.1},
            'batch_size': [32, 64, 128],
            'dropout': {'min': 0.1, 'max': 0.5},
            'hidden_layers': [2, 3, 4],
            'fixed_param': 42
        }
        samp_good_config['sampling_control'] = { 
            'seed': 69,
            'num_trials': 5
        }

        return samp_good_config

    # @pytest.fixture
    # def random_search_mode(self, sample_search_config):
    #     """Fixture for RandomSearchMode instance"""
    #     with patch('RandomSearchMode.validate_mode_specific_config_structure', return_value=True):
    #         return RandomSearchMode(sample_search_config)

    def test_validate_config_valid(self, sample_search_config):
        sample_search_config['experiment']['name'] = 'test_search_exp'
        mode = RandomSearchMode(sample_search_config)
        assert mode.validate_mode_specific_config_structure() is True

    @pytest.mark.parametrize("invalid_config", [
        {'experiment': {'name': 'test_no_search'}},  # Missing 'search' in name
        {'experiment': {'name': 'test_search', 'mode': 'invalid'}},  # Invalid mode
        {'experiment': {'name': 'test_search'}, 'training': {}},  # Missing num_trials
    ])
    def test_validate_config_invalid(self, sample_search_config, invalid_config):
        """Test validation fails with invalid configurations"""
        config = sample_search_config.copy()
        config.update(invalid_config)
        
        with pytest.raises(ValueError):
            RandomSearchMode(config).validate_mode_specific_config_structure()

    def test_sample_hyperparameters_types(self, sample_search_config):
        """Test hyperparameter sampling for different types"""
        sample_search_config['experiment']['name'] = 'test_search_exp'
        random_search_mode = RandomSearchMode(sample_search_config)
        hyperparams = random_search_mode._sample_hyperparameters()
        
        # Check types and ranges
        assert isinstance(hyperparams['learning_rate'], float)
        assert 0.0001 <= hyperparams['learning_rate'] <= 0.1
        
        assert hyperparams['batch_size'] in [32, 64, 128]
        assert 0.1 <= hyperparams['dropout'] <= 0.5
        assert hyperparams['hidden_layers'] in [2, 3, 4]
        assert hyperparams['fixed_param'] == 42

    def test_sample_hyperparameters_distribution(self, sample_search_config):
        """Test hyperparameter sampling distribution"""
        sample_search_config['experiment']['name'] = 'test_search_exp'
        random_search_mode = RandomSearchMode(sample_search_config)
        samples = [random_search_mode._sample_hyperparameters() for _ in range(100)]
        
        batch_sizes = [s['batch_size'] for s in samples]
        assert set(batch_sizes).issubset({32, 64, 128})
        
        learning_rates = [s['learning_rate'] for s in samples]
        assert all(0.0001 <= lr <= 0.1 for lr in learning_rates)

    def test_create_trial_config(self, sample_search_config):
        """Test trial configuration creation"""
        sample_search_config['experiment']['name'] = 'test_search_exp'
        random_search_mode = RandomSearchMode(sample_search_config)
        hyperparams = {
            'learning_rate': 0.01,
            'test_batch_size': 64,
            'dropout': 0.3
        }
        
        trial_config = random_search_mode._create_trial_config(0, hyperparams)
        
        assert trial_config['experiment']['name'] == 'trial_0'
        assert trial_config['experiment']['mode'] == 'single'
        assert trial_config['parameters']['dropout'] == 0.3
        assert trial_config['training']['learning_rate'] == 0.01
        assert trial_config['data']['test_batch_size'] == 64
        assert 'search_space' not in trial_config

    def test_get_trial_dir_constructor(self, sample_search_config, tmp_path):
        """Test trial directory constructor"""
        sample_search_config['experiment']['name'] = 'test_search_exp'
        random_search_mode = RandomSearchMode(sample_search_config)
        constructor = random_search_mode._get_trial_dir_constructor(0)
        trial_dir = constructor()
        
        assert trial_dir.exists()
        assert trial_dir.name == 'trial_0'
        assert trial_dir.parent == random_search_mode.dir

    def test_execute_multiple_trials(self, sample_search_config):
        """Test execution of multiple trials"""
        sample_search_config['experiment']['name'] = 'test_search_exp'
        random_search_mode = RandomSearchMode(sample_search_config)
        with patch('ml_framework.modes.RandomSearchMode.SingleMode') as mock_single_mode:
            mock_instance = Mock()
            mock_single_mode.return_value = mock_instance
            
            random_search_mode.execute()
            
            # Check number of trials
            assert mock_single_mode.call_count == random_search_mode.config['sampling_control']['num_trials']
            # Check each trial was executed
            assert mock_instance.execute.call_count == random_search_mode.config['sampling_control']['num_trials']

    def test_execute_error_handling(self, sample_search_config):
        """Test error handling during trial execution"""
        sample_search_config['experiment']['name'] = 'test_search_exp'
        random_search_mode = RandomSearchMode(sample_search_config)
        with patch('ml_framework.modes.RandomSearchMode.SingleMode') as mock_single_mode:
            mock_instance = Mock()
            mock_instance.execute.side_effect = [Exception("Trial error"), None, None]
            mock_single_mode.return_value = mock_instance
            
            # Should continue despite error in first trial
            random_search_mode.execute()
            
            # Should still complete all trials
            assert mock_single_mode.call_count == random_search_mode.config['sampling_control']['num_trials']

    def test_execute_trial_directory_structure(self, sample_search_config, tmp_path):
        """Test directory structure created during execution"""
        sample_search_config['experiment']['name'] = 'test_search_exp'
        random_search_mode = RandomSearchMode(sample_search_config)
        with patch('ml_framework.modes.SingleMode') as mock_single_mode:
            random_search_mode.execute()
            
            # Check trial directories were created
            for i in range(random_search_mode.config['sampling_control']['num_trials']):
                trial_dir = random_search_mode.dir / f"trial_{i}"
                assert trial_dir.exists()

    def test_experiment_reproducibility(self, sample_search_config):
        sample_search_config['experiment']['name'] = 'test_search_exp_00'
        sample_search_config['sampling_control']['seed'] = 42
        mode1 = RandomSearchMode(sample_search_config)
        hyperparams1 = mode1._sample_hyperparameters()
        
        sample_search_config['experiment']['name'] = 'test_search_exp_01'
        sample_search_config['sampling_control']['seed'] = 42
        mode2 = RandomSearchMode(sample_search_config)
        hyperparams2 = mode2._sample_hyperparameters()
        
        assert hyperparams1 == hyperparams2