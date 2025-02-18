from pathlib import Path
import json
from .SingleMode import SingleMode
import torch
from ..utils.validation_utils import check_section_exists, validate_mode_config, check_field, validate_metrics_file_format, validate_path_is_absolute, validate_path_exists
import sys
import warnings

class ResumeMode(SingleMode):
    """Mode for resuming training from a previous checkpoint."""

    REQUIRED_METRICS = [
    'train_loss',
    'dev_loss',
    'train_f1',
    'dev_f1',
    'best_dev_f1',
    'best_dev_loss'
    ]
    
    VALID_RESUME_TYPES = ["f1", "loss", "full"]
    
    def execute(self):
        super().execute()

    def validate_mode_specific_config_structure(self):
        validate_mode_config(self.config, 'resume')
        check_section_exists(self.config, 'resume')
        
        resume_config = self.config['resume']
        check_field(resume_config, 'metrics_path', str)
        validate_path_is_absolute(resume_config['metrics_path'])
        validate_path_exists(resume_config['metrics_path'])
        validate_metrics_file_format(Path(resume_config['metrics_path']), self.REQUIRED_METRICS)

        check_field(resume_config, 'model_path', str)
        validate_path_is_absolute(resume_config['model_path'])
        validate_path_exists(resume_config['model_path'])
        
        check_field(resume_config, 'resume_type', str)
        if resume_config['resume_type'] not in self.VALID_RESUME_TYPES:
            raise ValueError(f"'resume_type': {resume_config['resume_type']} is invalid. Must be one of: {self.VALID_RESUME_TYPES}")

        metrics_name = resume_config['metrics_path'].split(sep='/')[-1]
        model_name = resume_config['model_path'].split(sep='/')[-1]

        if resume_config['resume_type'].lower() not in metrics_name:
            print(f"Resume type ({resume_config['resume_type']}) is never in metrics file name FFF")

        if resume_config['resume_type'].lower() not in model_name:
            print(f"Resume type ({resume_config['resume_type']}) is never in model file name GG")
               
        return True

    def _initialize_metrics(self):
        try:
            metrics_path = Path(self.config['resume']['metrics_path'])
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics 
        except Exception as e:
            raise RuntimeError(f"Failed to load metrics from {metrics_path}: {e}")

    def _setup_model(self):
        model_path = Path(self.config['resume']['model_path'])
        try:

            obj = torch.load(model_path, map_location='cpu')

            if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
                print("Loading model from state dict.")
                model = super()._setup_model()
                model.load_state_dict(obj)
                return model
            elif hasattr(obj, "state_dict"):
                print("Loading model instance (not state dict format)")
                return obj
        except Exception as e:
            raise ValueError(f"Unknown model format: {e}")
            