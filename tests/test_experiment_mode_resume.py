import pytest
import torch
import torch.nn as nn
from pathlib import Path
import json
import torch.nn as nn
from unittest.mock import Mock, patch
from ml_framework.modes.ResumeMode import ResumeMode
import re
import os

class TestResumeMode:
    @pytest.fixture
    def valid_metrics(self): 
        return {
            'train_loss': [0.5, 0.4, 0.3],
            'dev_loss': [0.6, 0.5, 0.4],
            'train_f1': [0.7, 0.8, 0.9],
            'dev_f1': [0.6, 0.7, 0.8],
            'best_dev_f1': 0.8,
            'best_dev_loss': 0.4
        }

    @pytest.fixture
    def sample_model(self):
        class SAMPLE_MODEL(nn.Module):
            def __init__(self, input_size, input_channels, output_size, num_classes, hyperparams):
                super(SAMPLE_MODEL, self).__init__()
                self.layer1 = nn.Linear(input_size, output_size)

            def forward(self, x):
                return self.layer1(x)
        
        return SAMPLE_MODEL(
            input_size=10,  
            input_channels=1,  
            output_size=2,  
            num_classes=2,  
            hyperparams={}  
        )
            
    @pytest.fixture
    def sample_model_architecture_path(self):
        model_architecture = """
        import torch.nn as nn

        class MYMODEL(nn.Module):
            def __init__(self, input_size, input_channels, output_size, num_classes, hyperparams):
                super(MYMODEL, self).__init__()
                self.layer1 = nn.Linear(input_size, output_size)
                # Add more layers based on hyperparameters...

            def forward(self, x):
                return self.layer1(x)
        """
        path = Path(os.getcwd()) / "model_architecture.py"
        with open(path, "w") as f:
            f.write(model_architecture)
            
        yield str(path)
        if path.exists():
            path.unlink()
        



    @pytest.fixture
    def base_config(self, tmp_path):
        """Base configuration with required fields"""
        metrics_file = Path(os.getcwd()) / "metrics_f1.json"
        model_file = Path(os.getcwd()) / "model_f1.pth"

        metrics_file.touch()
        model_file.touch()

        (Path(tmp_path) / "experiments").mkdir()
        
        yield {
            'experiment': {
                'name': 'test_resume',
                'mode': 'resume',
                'project_root': str(tmp_path)
            },
            'resume': {
                'metrics_path': str(metrics_file),
                'model_path': str(model_file),
                'model_name': str(model_file),
                'resume_type': 'f1'
            },
            'data': {
                'script_name': 'TestDataScript',
                'absolute_path': str(tmp_path),
                'split_type': 'train,dev',
                'split_ratios': [0.8, 0.2]
            },
            'model': {
                'absolute_path': str(tmp_path / 'model.py')
            },
            'training': {}
        }
        if metrics_file.exists():
            os.unlink(metrics_file)
        if model_file.exists():    
            os.unlink(model_file)


    @pytest.mark.parametrize("resume_type", ['f1', 'loss', 'full'])
    def test_validate_mode_specific_config_valid_combinations(self, base_config, valid_metrics, tmp_path, resume_type):
        """Test valid configuration combinations for different resume types"""
        metrics_file = tmp_path / f"metrics_{resume_type}.json"
        model_file = tmp_path / f"model_{resume_type}.pth"
        
        metrics_file.write_text(json.dumps(valid_metrics))
        model_file.touch()
        
        config = base_config.copy()
        config['resume']['metrics_path'] = str(metrics_file)
        config['resume']['model_path'] = str(model_file)
        config['resume']['model_name'] = str(model_file)
        config['resume']['resume_type'] = resume_type
        
        resume_mode = ResumeMode(config)
        assert resume_mode.validate_mode_specific_config_structure() is True

    @pytest.mark.parametrize("invalid_field", [
        {'metrics_path': 'relative/path'},
        {'model_path': 'relative/path'},
        {'resume_type': 'invalid'},
        {'metrics_path': None},
        {'model_path': None}
    ])
    def test_validate_mode_specific_config_invalid_fields(self, base_config, invalid_field):
        """Test validation fails with invalid fields"""
        config = base_config.copy()
        config['resume'].update(invalid_field)
        
        with pytest.raises((ValueError, FileNotFoundError)):
            resume_mode = ResumeMode(config)

    def test_validate_mode_specific_config_missing_metrics_fields(self, base_config, tmp_path):
        invalid_metrics = {'train_loss': [0.5]}  # missing  fields that are needed
        Path(base_config['resume']['metrics_path']).write_text(json.dumps(invalid_metrics))
        
        with pytest.raises(ValueError, match=re.escape("Metrics file missing required metrics: ['dev_loss', 'train_f1', 'dev_f1', 'best_dev_f1', 'best_dev_loss']")):
            resume_mode = ResumeMode(base_config)

    def test_initialize_metrics_valid(self, base_config, valid_metrics, tmp_path):
        Path(base_config['resume']['metrics_path']).write_text(json.dumps(valid_metrics))
        
        resume_mode = ResumeMode(base_config)
        loaded_metrics = resume_mode._initialize_metrics()
        
        assert loaded_metrics == valid_metrics
        assert all(key in loaded_metrics for key in ResumeMode.REQUIRED_METRICS)

    def test_initialize_metrics_invalid_file(self, base_config):
        with pytest.raises(ValueError):
            resume_mode = ResumeMode(base_config)

    # def test_setup_model_state_dict(self, base_config, sample_model, sample_model_architecture_path, valid_metrics, tmp_path):
    #     """Test loading model from state dict"""
    #     model_file = tmp_path / "model_f1.pth"
    #     torch.save(sample_model.state_dict(), model_file)

    #     Path(base_config['resume']['metrics_path']).write_text(json.dumps(valid_metrics))        
    #     base_config['resume']['model_path'] = str(model_file)
    #     base_config['model']['absolute_path'] = sample_model_architecture_path
        
    #     resume_mode = ResumeMode(base_config)
    #     # Patch both the model setup and dataloader validation
    #     with patch('ml_framework.modes.SingleMode.SingleMode._setup_model', return_value=sample_model), \
    #         patch('ml_framework.utils.validation_utils.validate_dataloader_count'):  # Skip dataloader validation
    #             loaded_model = resume_mode._setup_training(model=None, dataloaders=None)
                
    #     assert isinstance(loaded_model, nn.Module)
    #     all([key_0 == key_1 for key_0, key_1 in zip(loaded_model.state_dict().keys(), sample_model.state_dict().keys())])

        
    # def test_setup_model_full(self, base_config, sample_model, tmp_path):
    #     """Test loading full model"""
    #     # Save full model
    #     model_file = tmp_path / "model_f1.pth"
    #     torch.save(sample_model, model_file)
        
    #     # Update config
    #     base_config['resume']['model_path'] = str(model_file)
        
    #     # Create instance and test
    #     resume_mode = ResumeMode(base_config)
    #     loaded_model = resume_mode._setup_model()
        
    #     # Verify
    #     self.assertEqual(
    #         loaded_model.state_dict().keys(),
    #         sample_model.state_dict().keys()
    #     )


    def test_setup_model_invalid_format(self, base_config, tmp_path, valid_metrics):
        """Test loading invalid model format"""
        model_file = tmp_path / "model_f1.pth"
        torch.save({'invalid': 'format'}, model_file)
        Path(base_config['resume']['metrics_path']).write_text(json.dumps(valid_metrics))
        
        resume_mode = ResumeMode(base_config)
        with pytest.raises(ValueError, match="Unknown model format"):
            resume_mode._setup_model()

    def test_execute_super_called(self, base_config, valid_metrics, tmp_path):
        """Test that execute calls parent's execute method"""        
        Path(base_config['resume']['metrics_path']).write_text(json.dumps(valid_metrics))

        
        with patch('ml_framework.modes.SingleMode.SingleMode.execute') as mock_super_execute:
            resume_mode = ResumeMode(base_config)
            resume_mode.execute()
            
            mock_super_execute.assert_called_once()