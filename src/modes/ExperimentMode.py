from abc import ABC, abstractmethod
from pathlib import Path
import yaml


class ExperimentMode(ABC):
    """Abstract base class defining the interface for different experiment modes."""

    def setup_experimant_dir(self):
        base_experiments_dir = self._create_directory(self._get_project_root(), "experiments")
        single_experiment_dir = self._create_directory(base_experiments_dir, self.config['experiment']['name'])
        self._create_subdirectories(single_experiment_dir)

        self._save_config(single_experiment_dir)
        self.dir = single_experiment_dir

    @abstractmethod
    def validate_mode_specific_config_structure(self):
        pass

    @abstractmethod
    def execute(self) -> None:
        pass

    def _get_project_root(self):
        return Path(__file__).resolve().parent.parent.parent
    
    def _create_directory(self, base, extention):
        exp_dir = base / extention
        exp_dir.mkdir(exist_ok=True)
        return exp_dir
    
    def _create_subdirectories(self, exp_dir):
        subdirs = ['models', 'metrics']
        for subdir in subdirs:
            (exp_dir / subdir).mkdir(exist_ok=False)
    
    def _save_config(self, exp_dir):
        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)
    