from abc import ABC, abstractmethod

class ExperimentMode(ABC):
    """Abstract base class defining the interface for different experiment modes."""
    
    @abstractmethod
    def execute(self, config: dict) -> None:
        pass
    
    @abstractmethod
    def validate_mode_specific_config_structure(self) -> bool:
        pass