from .ExperimentMode import ExperimentMode
from typing import Dict, Any
import os, sys
import importlib
from utils.validation_utils import validate_mode_config, validate_data_config, validate_training_config, check_section_exists




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
        validate_mode_config(self.config, "single")
    
        for section in ['data', 'training', 'model']:
            check_section_exists(self.config, section)
        
        validate_data_config(self.config['data'])
        validate_training_config(self.config['training'])
        
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
        try:
            script_name = self.config['data']['script_name']
            module_name = os.path.splitext(script_name)[0]
            data_module = importlib.import_module(module_name)
            data_script_class = getattr(data_module, module_name)
            
            self.data_script = data_script_class(self.config['data'])
            self.datasets = self.data_script.get_data()
            
            if not self.datasets or any(dataset is None for dataset in self.datasets):
                raise RuntimeError("Data script returned None or empty datasets")
            
        except ImportError as e:
            raise RuntimeError(f"Failed to import data script {script_name}: {e}")
        except AttributeError as e:
            raise RuntimeError(f"Failed to find data script class {module_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def _setup_training(self):
        
        pass

    def _set_metrics(self, metrics):
        if metrics == None:
            self.metrics = {
                'train_loss': [], 'dev_loss': [], 
                'train_f1': [], 'dev_f1': [],
                'best_f1_dev': 0, 'best_loss_dev': float('inf'), 
                'confusion_matrix': None
                }
        elif validate_metrics_structure(metrics):
            self.metrics = metrics




