from pathlib import Path
import random
import copy
from .ExperimentMode import ExperimentMode
from .SingleMode import SingleMode
from ..utils.validation_utils import check_section_exists, validate_mode_config

class RandomSearchMode(ExperimentMode):    
    def __init__(self, config):
        self.config = config
        self.validate_mode_specific_config_structure()
        self.dir = self._construct_experiment_path()
        
    def validate_mode_specific_config_structure(self):
        if 'search' not in self.config['experiment']['name']:
            raise ValueError(f"Experiment name must contain 'search' in in for random search mode. This does not: {self.config['experiment']['name']}")
        
        validate_mode_config(self.config, "search")
        check_section_exists(self.config, 'search_space')
        if 'num_trials' not in self.config['training']:
            raise ValueError("Training config must specify 'num_trials' for random search")
            
        return True
        
    def _sample_hyperparameters(self):
        search_space = self.config['search_space']
        hyperparams = {}
        
        for param, space in search_space.items():
            if isinstance(space, list):
                hyperparams[param] = random.choice(space)
            elif isinstance(space, dict) and 'min' in space and 'max' in space:
                hyperparams[param] = random.uniform(space['min'], space['max'])
            elif isinstance(space, (int, float, str)):
                hyperparams[param] = space
            else:
                raise ValueError(f"Invalid search space format for parameter: {param}")
                
        return hyperparams
        
    def _create_trial_config(self, trial_num: int, hyperparams: dict):
        trial_config = copy.deepcopy(self.config)
        trial_config['experiment']['name'] = f"trial_{trial_num}"
        trial_config['experiment']['mode'] = "single"
        trial_config['parameters'] = hyperparams
        del trial_config['search_space']
        return trial_config
        
    def _get_trial_dir_constructor(self, trial_num: int):
        trial_dir = self.dir / f"trial_{trial_num}"
        
        def construct_trial_dir():
            trial_dir.mkdir(parents=True, exist_ok=True)
            return trial_dir
            
        return construct_trial_dir
        
    def execute(self):
        num_trials = self.config['training']['num_trials']
        
        for trial in range(num_trials):
            hyperparams = self._sample_hyperparameters()
            trial_config = self._create_trial_config(trial, hyperparams)
            trial_dir_constructor = self._get_trial_dir_constructor(trial)
            
            print(f"\nStarting Trial {trial}/{num_trials}")
            print("Hyperparameters:", hyperparams)
            
            try:
                single_mode = SingleMode(config=trial_config, experiment_dir_constructor=trial_dir_constructor)
                single_mode.execute()
                
            except Exception as e:
                print(f"Error in trial {trial}: {e}")
                continue