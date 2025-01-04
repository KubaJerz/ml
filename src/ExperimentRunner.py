import yaml
from pathlib import Path
from typing import Dict, Any
from .ExperimentModeFactory import ExperimentModeFactory
from .utils.validation_utils import validate_core_config_structure

class ExperimentRunner:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.config_path = config_path

    def run(self):
        #prepare
        self._validate_core_config_structure()
        factory = ExperimentModeFactory(self.config)
        mode = factory.create_mode()
        mode.validate_mode_specific_config_structure()
        mode.setup_experimant_dir()

        #execute
        mode.execute()

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _validate_core_config_structure(self):
        validate_core_config_structure(self.config)