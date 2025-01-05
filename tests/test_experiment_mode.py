import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modes.SingleMode import SingleMode
import pytest
import shutil
from pathlib import Path



class TestExperimentModes:

    @pytest.fixture
    def sample_single_mode(self):
        config = {
            'experiment': {
                'name': 'eeg_classification',
                'mode': 'single',
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
        mode = SingleMode(config)
        return mode
    
    @pytest.fixture
    def path_to_experiments(self):
        path_to_experiments = Path('experiments') 
        yield path_to_experiments
        shutil.rmtree(path_to_experiments)

    
    def test_validate_mode_specific_config_structure(self, sample_single_mode):
        mode = sample_single_mode
        res = mode.validate_mode_specific_config_structure()
        assert res == True

    def test_setup_experimant_dir(self, sample_single_mode, path_to_experiments):
        sample_single_mode.setup_experimant_dir()
        experiment_name = sample_single_mode.config['experiment']['name']
        assert path_to_experiments.is_dir()
        assert (path_to_experiments / experiment_name / 'models').is_dir()
        assert (path_to_experiments / experiment_name / 'metrics').is_dir()
        assert (path_to_experiments / experiment_name / 'config.yaml').is_file()

    def test_self_dir(self, sample_single_mode, path_to_experiments):
        sample_single_mode.setup_experimant_dir()
        experiment_name = sample_single_mode.config['experiment']['name']
        assert sample_single_mode.dir ==  Path(f'/Users/kuba/projects/ml/{str(path_to_experiments)}/{str(experiment_name)}')





        