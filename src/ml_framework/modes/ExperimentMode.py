from abc import ABC, abstractmethod
from pathlib import Path
from ..utils.validation_utils import validate_path_exists
import yaml

class ExperimentMode(ABC):
    """Abstract base class defining the interface for different experiment modes."""

    @abstractmethod
    def validate_mode_specific_config_structure(self):
        pass

    @abstractmethod
    def execute(self) -> None:
        pass

    def _construct_experiment_path(self) -> Path:
        project_root = Path(self.config['experiment']['project_root'])
        experiment_name = self.config['experiment']['name']
        
        validate_path_exists(project_root)
            
        experiment_path = project_root / 'experiments' / experiment_name
        if experiment_path.exists():
            raise FileExistsError(f"Cannot create experiment '{experiment_name}': Directory already exists. Please use a unique name.")
        
        experiment_path.mkdir()
        return experiment_path
    
    def _save_config(self):
        config_path = Path(self.dir) / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.safe_dump(self.config, f, default_flow_style=False, sort_keys=False)
    