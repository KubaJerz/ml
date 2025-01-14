import yaml
from typing import Dict, Any
from .ExperimentModeFactory import ExperimentModeFactory
from .utils.validation_utils import validate_path_is_absolute, validate_path_exists, validate_core_config_structure
from pathlib import Path


class ExperimentRunner:
    def __init__(self, path_to_config: str):
        self.config = self._load_config(path_to_config)
        self.path_to_config = path_to_config
        validate_core_config_structure(self.config)
        self._initialize_experiments_directory()

    def run(self):
        #prepare
        factory = ExperimentModeFactory(self.config)
        mode = factory.create_mode()
        
        #execute
        mode.execute()

    def _load_config(self, path_to_config):
        with open(path_to_config, 'r') as f:
            return yaml.safe_load(f)
        
    def _initialize_experiments_directory(self):
        project_root = self.config['experiment']['project_root']
        validate_path_is_absolute(project_root)
        validate_path_exists(project_root)
        base_path = Path(project_root) / "experiments"
    
        base_path.mkdir(exist_ok=True)