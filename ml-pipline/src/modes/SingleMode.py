from .ExperimentMode import ExperimentMode
from typing import Dict, Any
from ..utils import check_field, check_section_exists
import os, sys
import importlib


class SingleMode(ExperimentMode):
    def __init__(self, config):
        self.config = config
        self.dir = None

        self.model = None
        self.training_loop = None
    
    def execute(self):
        self._setup_model() 
        self._setup_data() #when we implinet this go back to validate_mode_specific_config_structure and fic this
        self._setup_training()
        self.training_loop.fit()

    def validate_mode_specific_config_structure(self):
        
        if self.config.get('experiment', {}).get('mode') != 'single':
            raise ValueError("Mode must be 'single' to use SingleMode")
 
        check_section_exists(self.config, 'data')
        check_section_exists(self.config, 'training')
        check_section_exists(self.config, 'model')
        
        data = self.config['data']
        check_field(data, 'num_classes', int)
        check_field(data, 'input_size', int)
        check_field(data, 'input_channels', int)
        check_field(data, 'output_size', int)
        
        training = self.config['training']
        check_field(training, 'epochs', int)
        check_field(training, 'train_batch_size', int)
        check_field(training, 'test_batch_size', int)
        check_field(training, 'learning_rate', float)
        
        return True      
            

    def setup_experimant_dir(self):
        super().setup_experimant_dir()

    def _setup_model(self):
        model_config = self.config['model']

        model_path = model_config.get('absolute_path')

        try:
            dir_path, file_name = os.path.split(model_path)
            module_name = os.path.splitext(file_name)[0]
            model_class_name = module_name.upper()
            
            sys.path.append(dir_path)
            model_module = importlib.import_module(module_name)
            model_class = getattr(model_module, model_class_name)
            
            model_params = {
                'input_size': self.config['data']['input_size'],
                'input_channels': self.config['data']['input_channels'],
                'output_size': self.config['data']['output_size'],
                'num_classes': self.config['data']['num_classes'],
                'hyperparams': model_config['parameters']
            }
            self.model = model_class(**model_params)
        
        except ImportError as e:
            raise ValueError(f"Failed to import model module: {e}")
        except AttributeError as e:
            raise ValueError(f"Failed to find model class in module: {e}")
        except TypeError as e:
            raise ValueError(f"Model initialization failed - incorrect parameters: {e}")
        except Exception as e:
            raise ValueError(f"Error setting up model: {e}")

    def _setup_data(self):
        pass

    def _setup_training(self):
        pass




