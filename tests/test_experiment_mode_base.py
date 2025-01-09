import pytest
import os
from pathlib import Path
from ml_framework.modes.ExperimentMode import ExperimentMode
import shutil

class ConcreteExperimentMode(ExperimentMode):
    def __init__(self, config):
        self.config = config
        
    def validate_mode_specific_config_structure(self):
        pass
        
    def execute(self):
        pass

class TestExperimentMode:
    def test_construct_experiment_path(self):
        path = '/Users/kuba/projects/ml-test/tests'
        project_root = Path(path)

        config = {
            "experiment": {
                "name": "test",
                "mode": "single",
                "project_root": path
            }
        }
        
        mode = ConcreteExperimentMode(config)
        result = mode._construct_experiment_path()
        
        expected_path = project_root / 'experiments' / config['experiment']['name']
        assert result == expected_path
        assert isinstance(result, Path)

    def test_construct_experiment_path_with_non_exist_path(self):
        path = '/Users/bob/not/real/path'
        project_root_with_experiments = Path(path)

        config = {
            "experiment": {
                "name": "test",
                "mode": "single",
                "project_root": path
            }
        }
        
        mode = ConcreteExperimentMode(config)
        with pytest.raises(ValueError, match=f"Project root does not exist: {path}"):
            result = mode._construct_experiment_path()