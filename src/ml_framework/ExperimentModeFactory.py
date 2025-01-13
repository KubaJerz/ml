from .modes.ExperimentMode import ExperimentMode
from .modes.SingleMode import SingleMode
# from .modes.RandSearchMode import RandSearchMode
from .modes.ResumeMode import ResumeMode

class ExperimentModeFactory:    
    def __init__(self, config: dict):
        self.config = config
        self._valid_modes = {
            'single': SingleMode,
            # 'random_search': RandSearchMode,
            'resume': ResumeMode
        }

    def create_mode(self) -> ExperimentMode:
        mode_type = self.config['experiment'].get('mode', 'no_mode_provided').lower()
        self._verify_valid_mode(mode_type)
        self.config['experiment']['mode'] = mode_type

        mode_class = self._valid_modes[mode_type]
        return mode_class(self.config)
        
    def _verify_valid_mode(self, mode_type):
        if mode_type not in self._valid_modes:
            valid_modes = list(self._valid_modes.keys())
            raise TypeError(f"Invalid mode type '{mode_type}'. Must be one of: {valid_modes}")