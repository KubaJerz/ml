import yaml
from typing import Dict, Any
from .ExperimentModeFactory import ExperimentModeFactory
from .utils.validation_utils import validate_core_config_structure, validate_data_config, validate_training_config, validate_model_config
from pathlib import Path
import os


class ExperimentRunner:
    def __init__(self, path_to_config: str):
        self.config = self._load_config(path_to_config)
        self.path_to_config = path_to_config
        self._validate_core_config_structure()
        self._initialize_experiment_directory()

    def run(self):
        #prepare
        factory = ExperimentModeFactory(self.config)
        mode = factory.create_mode()
        
        #execute
        mode.execute()

    def _load_config(self, path_to_config):
        with open(path_to_config, 'r') as f:
            return yaml.safe_load(f)
        
    def _initialize_experiment_directory(self):
        experiment_name = self.config['experiment']['name']
        base_path = self._get_project_root() / "experiments"
        experiment_path = base_path / experiment_name
    
        if experiment_path.exists():
            raise FileExistsError(f"Cannot create experiment '{experiment_name}': Directory already exists. Please use a unique name.")
    
        base_path.mkdir(exist_ok=True)
        experiment_path.mkdir(exist_ok=False)
        self._create_subdirectories(experiment_path)
        self._save_config(experiment_path)

    def _get_project_root(self):
        return Path(self.config['experiment']['project_root'])
    
    def _create_subdirectories(self, exp_dir):
        subdirs = ['models', 'metrics']
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(exist_ok=False)
    
    def _save_config(self, exp_dir):
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)

    def _validate_core_config_structure(self):
        validate_core_config_structure(self.config)
        validate_data_config(self.config['data'])
        validate_training_config(self.config['training'])
        validate_model_config(self.config['model'])
