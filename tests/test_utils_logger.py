import tempfile
import pytest
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from unittest.mock import patch, Mock
from ml_framework.utils.logging_utils import *

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.linear(x)

class TestLoggingUtils:
    @pytest.fixture
    def sample_metrics(self):
        return {
            'train_loss': [0.5, 0.3, 0.2],
            'dev_loss': [0.6, 0.4, 0.3],
            'train_f1': [0.7, 0.8, 0.9],
            'dev_f1': [0.6, 0.7, 0.8],
            'best_dev_loss': 0.3,
            'best_dev_f1': 0.8
        }

    @pytest.fixture
    def sample_model(self):
        return SimpleModel()

    def test_save_metrics_file_creation(self, sample_metrics):
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_dir =  tmp_dir
            name = "test_metrics"    
            
            save_metrics(sample_metrics, name, Path(save_dir))
            
            metrics_file = Path(save_dir) / "metrics" / f"metrics_{name}.json"
            assert metrics_file.exists()
            print(metrics_file)
            
            with open(metrics_file) as f:
                saved_metrics = json.load(f)
            assert saved_metrics == sample_metrics

    def test_save_metrics_json_structure(self, sample_metrics, tmp_path):
        """Test that saved JSON has correct structure"""
        save_dir = tmp_path
        name = "test_metrics"
        
        save_metrics(sample_metrics, name, save_dir)
        
        metrics_file = save_dir / "metrics" / f"metrics_{name}.json"
        with open(metrics_file) as f:
            saved_metrics = json.load(f)
            
        # Check structure
        assert isinstance(saved_metrics['train_loss'], list)
        assert isinstance(saved_metrics['best_dev_loss'], float)
        assert len(saved_metrics['train_loss']) == len(sample_metrics['train_loss'])

    def test_save_metrics_directory_creation(self, sample_metrics, tmp_path):
        """Test that directories are created if they don't exist"""
        save_dir = tmp_path / "deep" / "nested" / "directory"
        name = "test_metrics"

        
        save_metrics(sample_metrics, name, save_dir)
        
        assert (save_dir / "metrics").exists()
        assert (save_dir / "metrics" / f"metrics_{name}.json").exists()


    def test_save_model_state_saving(self, sample_model, sample_metrics, tmp_path):
        """Test saving model state dict"""
        save_dir = tmp_path
        name = "test_model"
        
        save_model(sample_model, sample_metrics, name, save_dir, save_full_model=False)
        
        model_path = save_dir / "models" / f"{name}.pth"
        loaded_state_dict = torch.load(model_path)
        
        # Verify it's a state dict
        assert isinstance(loaded_state_dict, dict)
        assert 'linear.weight' in loaded_state_dict
        
        # Verify it can be loaded into a model
        new_model = SimpleModel()
        new_model.load_state_dict(loaded_state_dict)

    def test_save_model_full_saving(self, sample_model, sample_metrics, tmp_path):
        """Test saving full model"""
        save_dir = tmp_path
        name = "test_model"
        
        save_model(sample_model, sample_metrics, name, save_dir, save_full_model=True)
        
        model_path = save_dir / "models" / f"{name}.pth"
        loaded_model = torch.load(model_path)
        
        # Verify it's a full model
        assert isinstance(loaded_model, nn.Module)
        assert isinstance(loaded_model, SimpleModel)



    def test_save_model_with_cuda_tensors(self, sample_metrics, tmp_path):
        """Test saving model with CUDA tensors if available"""
        if torch.cuda.is_available():
            save_dir = tmp_path
            name = "test_model"
            model = SimpleModel().cuda()
            
            save_model(model, sample_metrics, name, save_dir, save_full_model=True)
            
            # Load and verify
            loaded_model = torch.load(save_dir / "models" / f"{name}.pth")
            assert next(loaded_model.parameters()).is_cuda