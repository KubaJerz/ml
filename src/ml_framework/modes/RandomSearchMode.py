from pathlib import Path
import random
import copy
from .ExperimentMode import ExperimentMode
from .SingleMode import SingleMode
from ..utils.validation_utils import check_section_exists, validate_mode_config, check_field

class RandomSearchMode(ExperimentMode): 

    PARAMETER_LOCATIONS = {
        'epochs': 'training',
        'learning_rate': 'training',
        'train_batch_size': 'data',
        'dev_batch_size': 'data',
        'test_batch_size': 'data',
        'hidden_blocks': 'parameters',
        'layer_depth': 'parameters',
        'dropout_rate': 'parameters',
        'activation': 'parameters',
        'normalization': 'parameters'
    }

    def __init__(self, config):
        self.config = config
        self.validate_mode_specific_config_structure()
        self.dir = super()._construct_experiment_path()
        super()._save_config()
        
    def validate_mode_specific_config_structure(self):
        if 'search' not in self.config['experiment']['name']:
            raise ValueError(f"Experiment name must contain 'search' in in for random search mode. This does not: {self.config['experiment']['name']}")
        
        validate_mode_config(self.config, "random_search")
        check_section_exists(self.config, 'sampling_control')
        check_field(self.config['sampling_control'], 'seed', int)
        check_field(self.config['sampling_control'], 'num_trials', int)

        check_section_exists(self.config, 'search_space')
            
        return True
        
    def _sample_hyperparameters(self):
        random.seed(self.config['sampling_control']['seed'])
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
        """Create config for single mode"""
        trial_config = copy.deepcopy(self.config)
        trial_config['experiment']['name'] = f"trial_{trial_num}"
        trial_config['experiment']['mode'] = "single"

        for param_name, value in hyperparams.items():
            if param_name not in self.PARAMETER_LOCATIONS:
                print(f"Unknown parameter location for: {param_name}, will place it in 'parameters' which are passed to the model __init__")
                section = 'parameters'
            else:
                section = self.PARAMETER_LOCATIONS[param_name]
            trial_config[section][param_name] = value

        del trial_config['search_space']
        return trial_config
    
    def _get_trial_dir_constructor(self, trial_num: int):
        trial_dir = self.dir / f"trial_{trial_num}"
        
        def construct_trial_dir():
            trial_dir.mkdir(parents=True, exist_ok=True)
            return trial_dir
            
        return construct_trial_dir
        
    def execute(self):
        num_trials = self.config['sampling_control']['num_trials']
        
        for trial in range(num_trials):
            hyperparams = self._sample_hyperparameters()
            trial_config = self._create_trial_config(trial, hyperparams)
            trial_dir_constructor = self._get_trial_dir_constructor(trial)
            
            print(f"\nStarting Trial {trial}/{num_trials}")
            print("Hyperparameters:")
            for name, value in hyperparams.items():
                section = self.PARAMETER_LOCATIONS.get(name, 'parameters')
                print(f"  {name} ({section}): {value}") 
            
            try:
                single_mode = SingleMode(config=trial_config, experiment_dir_constructor=trial_dir_constructor)
                single_mode.execute()
                
            except Exception as e:
                print(f"Error in trial {trial}: {e}")
                continue