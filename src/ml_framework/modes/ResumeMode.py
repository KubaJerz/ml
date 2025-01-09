# from .ExperimentMode import ExperimentMode
# from typing import Dict, Any

# import os, sys
# import importlib
# from ..utils.validation_utils import validate_mode_config, validate_data_config, validate_training_config, check_section_exists, validate_model_config, validate_dataloader_count
# import torch



# class ResumeMode(ExperimentMode):
#     def __init__(self, config):
#         self.config = config
#         self.dir = super()._construct_experiment_path()

#     def execute(self):
#         model = self._setup_model() 
#         dataloaders = self._setup_data()
#         training_loop = self._setup_training(model=model, dataloaders=dataloaders)
#         training_loop.fit()

#     def validate_mode_specific_config_structure(self):
#         validate_mode_config(self.config, "single")
    
#         for section in ['data', 'training', 'model']:
#             check_section_exists(self.config, section)
        
#         validate_data_config(self.config['data'])
#         validate_training_config(self.config['training'])
        
#         return True      