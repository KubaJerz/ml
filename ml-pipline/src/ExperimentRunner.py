import yaml
from pathlib import Path
from typing import Dict, Any
from .ExperimentModeFactory import ExperimentModeFactory
from ..src.utils import check_section_exists, check_field

class ExperimentRunner:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.config_path = config_path

    def run(self):
        self._validate_core_config_structure()
        mode = ExperimentModeFactory.create_mode(self.config)
        mode.validate_mode_specific_config_structure()
        mode.setup_experimant_dir()
        mode.execute()

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_core_config_structure(self):
        check_section_exists(self.config, 'experiment')
        check_section_exists(self.config, 'data')
        check_section_exists(self.config, 'model')
        check_section_exists(self.config, 'parameters')
        check_section_exists(self.config, 'training')
        
        experiment = self.config['experiment']
        check_field(experiment, 'name', str)
        check_field(experiment, 'mode', str)
        
        data = self.config['data']
        check_field(data, 'shuffle', str)
        check_field(data, 'seed', int)