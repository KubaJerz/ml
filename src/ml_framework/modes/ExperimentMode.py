from abc import ABC, abstractmethod
from pathlib import Path


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
        
        if not project_root.exists():
            raise ValueError(f"Project root does not exist: {project_root}")
            
        experiment_path = project_root / 'experiments' / experiment_name
        return experiment_path
    