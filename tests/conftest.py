import pytest
import yaml
from pathlib import Path
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import os 

@pytest.fixture
def sample_empty_config_path():
    config_file = Path('./temp_config.yaml')
    config = {}
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    yield str(config_file) 
    if config_file.exists():
        config_file.unlink()

@pytest.fixture
def sample_good_config_path():
    config_file = Path('./temp_config.yaml')
    experiment_dir = Path('/Users/kuba/projects/ml-test/experiments/tester00')

    config = {
        "experiment": {
            "name": "tester00",
            "mode": "single",
            "project_root": "/Users/kuba/projects/ml-test/"
        },
        
        "data": {
            # Data source configuration
            'data_absolute_path': '/Users/kuba/Documents/data/Raw/pt_ekyn_500hz',
            'script_absolute_path': '/Users/kuba/projects/ml/src/ml_framework/data_script/EEGDataScript.py',
            
            # Data sampling configuration
            "use_full": True,
            "use_percent": 0.15,
            
            # Split configuration
            "split_type": "train,dev,test",
            "split_ratios": [0.8, 0.1, 0.1],
            "shuffle": True,
            "seed": 69,
            "train_batch_size": 64,
            "dev_batch_size": -1,
            "test_batch_size": -1,

            
            # Data dimensions
            "input_size": 5000,
            "input_channels": 1,
            "output_size": 3,
            "num_classes": 3
        },
        
        "model": {
            "absolute_path": "/Users/kuba/projects/ml-test/tests/sample_model.py"
        },

        "parameters": {
            "depth": 5
        },
        
        "training": {
            # Training parameters
            "epochs": 15,
            
            # Optimization parameters
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "criterion": "CrossEntropyLoss",
            
            # Environment configuration
            "device": "cpu",
            "save_full_model": True
        },
        
        "callbacks": {
            # Visualization callbacks
            "plot_combined_metrics": True,
            "plot_metrics_live": True,
            
            # Model saving callbacks
            "best_f1": True,
            "best_loss": True,
            
            # Early stopping configuration
            "early_stopping": False,
            "early_stopping_patience": 3,
            "early_stopping_monitor": "dev_loss"
        }
    }
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    yield str(config_file) 
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
    if config_file.exists():
        config_file.unlink()

@pytest.fixture
def samp_good_config():
    temp_dir = tempfile.mkdtemp()
    pt_file = Path(temp_dir+'/temp_for_test.pt')
    pt_file.touch()

    temp_model = Path(temp_dir+'/temp_model.pt')
    temp_model.touch()
    config = {
        "experiment": {
            "name": "tester00",
            "mode": "single",
            "project_root": str(temp_dir) #"/Users/kuba/projects/ml-test/"
        },
        
        "data": {
            # Data source configuration
            "data_absolute_path": str(temp_dir), # "/Users/kuba/Documents/data/Raw/pt_ekyn_500hz",
            'script_absolute_path': '/Users/kuba/projects/ml/src/ml_framework/data_script/EEGDataScript.py',
            
            # Data sampling configuration
            "use_full": True,
            "use_percent": 0.15,
            
            # Split configuration
            "split_type": "train,dev,test",
            "split_ratios": [0.8, 0.1, 0.1],
            "shuffle": True,
            "seed": 69,
            "train_batch_size": 64,
            "dev_batch_size": -1,
            "test_batch_size": -1,
            #"num_workers": 2, #optional
            #"pin_memory": true, #optional
            
            # Data dimensions
            "input_size": 5000,
            "input_channels": 1,
            "output_size": 3,
            "num_classes": 3
        },
        
        "model": {
            "absolute_path": str(temp_model)
        },

        "parameters": {
                "depth": 5
        },
        
        "training": {
            # Training parameters
            "epochs": 15,
            
            # Optimization parameters
            "optimizer": "Adam",
            "learning_rate": 0.041,
            "criterion": "CrossEntropyLoss",
            
            # Environment configuration
            "device": "cpu",
            "save_full_model": True
        },
        
        "callbacks": {
            # Visualization callbacks
            "plot_combined_metrics": True,
            "plot_metrics_live": True,
            
            # Model saving callbacks
            "best_f1": True,
            "best_loss": True,
            
            # Early stopping configuration
            "early_stopping": False,
            "early_stopping_patience": 3,
            "early_stopping_monitor": "dev_loss"
        }
    }   
    yield config
    if pt_file.exists():
        pt_file.unlink()
    # if temp_model.exists():
    #     temp_model.unlink()

@pytest.fixture
def valid_data_config_without_test():
    return {
        "data_absolute_path": "/Users/kuba/Documents/data/Raw/pt_ekyn_500hz",
        "script_absolute_path": "/Users/kuba/projects/ml/src/ml_framework/data_script/EEGDataScript.py",
        
        # Data sampling configuration
        "use_full": False,
        "use_percent": 0.15,
        
        # Split configuration
        "split_type": "train,dev",
        "split_ratios": [0.8, 0.2],
        "shuffle": True,
        "seed": 69,
        "train_batch_size": 16,
        "dev_batch_size": -1,
        "num_workers": 2, #optional
        "pin_memory": True, #optional
        
        # Data dimensions
        "input_size": 5,
        "input_channels": 1,
        "output_size": 1,
        "num_classes": 2
    }

@pytest.fixture
def valid_data_config_with_test():
    return {
        "data_absolute_path": "/Users/kuba/Documents/data/Raw/pt_ekyn_500hz",
        "script_absolute_path": "/Users/kuba/projects/ml/src/ml_framework/data_script/EEGDataScript.py",
        
        # Data sampling configuration
        "use_full": True,
        "use_percent": 0.15,
        
        # Split configuration
        "split_type": "train,dev,test",
        "split_ratios": [0.7, 0.15, 0.15],
        "shuffle": True,
        "seed": 69,
        "train_batch_size": 64,
        "dev_batch_size": -1,
        "test_batch_size": -1,
        "num_workers": 2, #optional
        "pin_memory": True, #optional
        
        # Data dimensions
        "input_size": 5000,
        "input_channels": 1,
        "output_size": 3,
        "num_classes": 3
    }

@pytest.fixture
def invalid_datasplit_config_without_test():
    return {
        "data_absolute_path": "/Users/kuba/Documents/data/Raw/pt_ekyn_500hz",
        "script_absolute_path": "/Users/kuba/projects/ml/src/ml_framework/data_script/EEGDataScript.py",
        
        # Data sampling configuration
        "use_full": False,
        "use_percent": 0.15,
        
        # Split configuration
        "split_type": "train,dev",
        "split_ratios": [0.8, 0.4],
        "shuffle": True,
        "seed": 69,
        "train_batch_size": 64,
        "dev_batch_size": -1,
        "num_workers": 2, #optional
        "pin_memory": True, #optional
        
        # Data dimensions
        "input_size": 5000,
        "input_channels": 1,
        "output_size": 3,
        "num_classes": 3
    }

@pytest.fixture
def invalid_datasplit_config_with_test():
    return {
        "data_absolute_path": "/Users/kuba/Documents/data/Raw/pt_ekyn_500hz",
        "script_absolute_path": "/Users/kuba/projects/ml/src/ml_framework/data_script/EEGDataScript.py",
        
        # Data sampling configuration
        "use_full": False,
        "use_percent": 0.15,
        
        # Split configuration
        "split_type": "train,dev,test",
        "split_ratios": [0.8, 0.1, 0.5],
        "shuffle": True,
        "seed": 69,
        "train_batch_size": 64,
        "dev_batch_size": -1,
        "test_batch_size": -1,
        "num_workers": 2, #optional
        "pin_memory": True, #optional
        
        # Data dimensions
        "input_size": 5000,
        "input_channels": 1,
        "output_size": 3,
        "num_classes": 3
    }

@pytest.fixture
def invalid_data_config_with_test():
    return {
        "data_absolute_path": "/Users/kuba/Documents/data/Raw/pt_ekyn_500hz",
        "script_absolute_path": "/Users/kuba/projects/ml/src/ml_framework/data_script/EEGDataScript.py",
        
        "use_full": False,
        "use_percent": 0.15,
        
        "split_type": "train,dev,test",
        "split_ratios": [0.8, 0.1, 0.5],
        "shuffle": True,
        "seed": "not a number",
        "train_batch_size": 64,
        "test_batch_size": -1,
        "num_workers": 2, #optional
        "pin_memory": True, #optional
        
        # Data dimensions
        "input_size": 5000,
        "input_channels": 1,
        "output_size": 3,
        "num_classes": 3
    }

class SimpleModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(10, num_classes)
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.linear(x)

@pytest.fixture
def base_trianloop_setup():
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(X, y)
    
    model = SimpleModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(dataset, batch_size=16)
    dev_loader = DataLoader(dataset, batch_size=16)
    
    metrics = {
            'train_loss': [],
            'dev_loss': [],
            'train_f1': [],
            'dev_f1': [],
            'best_dev_f1': float('-inf'),
            'best_dev_loss': float('inf'),
        }
    
    temp_dir = tempfile.mkdtemp()
    save_dir = Path(temp_dir)
    
    return {
        'model': model,
        'device': device,
        'optimizer': optimizer,
        'criterion': criterion,
        'train_loader': train_loader,
        'dev_loader': dev_loader,
        'metrics': metrics,
        'save_dir': save_dir
    }

@pytest.fixture
def setup_for_fake_callbacks():
    path = Path(os.getcwd()+"/test_callbacks_temp")
    path.mkdir()
    
    path_modes = path / "models"
    path_modes.mkdir()
    path_metrics = path / "metrics"
    path_metrics.mkdir()

    yield {
        'save_dir': path,
        'model': SimpleModel(),
        'metrics': {
            'dev_loss': [0.5, 0.4, 0.3],
            'dev_f1': [0.6, 0.7, 0.8],
            'best_dev_loss': 0.3,
            'best_dev_f1': 0.8
        }
    }
    shutil.rmtree(path=path)

@pytest.fixture
def setup_for_fake_PlotCombinedMetrics():
    return {
        'save_dir': Path(tempfile.mkdtemp()),
        'metrics': {
            'train_loss': [0.5, 0.4, 0.3],
            'dev_loss': [0.6, 0.5, 0.4],
            'train_f1': [0.6, 0.7, 0.8],
            'dev_f1': [0.5, 0.6, 0.7],
            'best_dev_loss': 0.4,
            'best_dev_f1': 0.7
        }
    }

@pytest.fixture
def valid_core_config():
    """Fixture for a valid core configuration"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield {
            'experiment': {
                'name': 'test_experiment',
                'mode': 'single',
                'project_root': str(Path(tmp_dir).absolute())
            },
            'data': {},
            'model': {},
            'parameters': {},
            'training': {},
            'callbacks': {}
        }

@pytest.fixture
def valid_data_config():
    """Fixture for a valid data configuration"""
    temp_dir = tempfile.mkdtemp()
    pt_file = Path(temp_dir+'/temp_for_test.pt')
    pt_file.touch()
    yield {
        "data_absolute_path": "/Users/kuba/Documents/data/Raw/pt_ekyn_500hz",
        "script_absolute_path": "/Users/kuba/projects/ml/src/ml_framework/data_script/EEGDataScript.py",
        'split_type': 'train,dev',
        'split_ratios': [0.8, 0.2],
        'shuffle': True,
        'seed': 42,
        "train_batch_size": 16,
        "dev_batch_size": -1,
        'input_size': 100,
        'input_channels': 1,
        'output_size': 10,
        'num_classes': 3
    }
    if pt_file.exists():
        pt_file.unlink()
