from modes.ExperimentMode import ExperimentMode
from modes.SingleMode import SingleMode
# from .modes.RandSearchMode import RandSearchMode
# from .modes.ResumeF1 import ResumeF1
# from .modes.ResumeLoss import ResumeLoss
# from .modes.ResumeFull import ResumeFull

"""
TODO

we can make this so that it just searches each file in the ./modes/ and trys to insatiate it since it wont be hard coded them.
"""

class ExperimentModeFactory:    
    def __init__(self, config: dict):
        self.config = config
        self._valid_modes = {
            'single': SingleMode,
            # 'random_search': RandSearchMode,
            # 'resume_f1': ResumeF1,
            # 'resume_loss': ResumeLoss,
            # 'resume_full': ResumeFull
        }


    def create_mode(self, config: dict) -> ExperimentMode:
        mode_type = config.get('mode', 'no_mode_provided').lower()
        self._verify_valid_mode(mode_type)

        mode_class = self._valid_modes[mode_type]
        return mode_class(config)
        
    def _verify_valid_mode(self, mode_type):
        if mode_type not in self._valid_modes:
            valid_modes = list(self._valid_modes.keys())
            raise TypeError(f"Invalid mode type '{mode_type}'. Must be one of: {valid_modes}")


