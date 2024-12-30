from .ExperimentMode import ExperimentMode
from typing import Dict, Any
from utils import check_field, check_section_exists
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
        check_field(data, 'absolute_path', str)
        check_field(data, 'script_name', str)
        check_field(data, 'split_type', str)
        check_field(data, 'split_ratios', float, is_sequence=True)
        check_field(data, 'shuffle', bool)
        check_field(data, 'seed', int)
        
        split_type = data['split_type']
        split_ratios = data['split_ratios']
        
        if split_type == "train,test":
            if len(split_ratios) != 2:
                raise ValueError("train,test split type requires exactly 2 split values")
        elif split_type == "train,test,val":
            if len(split_ratios) != 3:
                raise ValueError("train,test,val split type requires exactly 3 split values")
        else:
            raise ValueError("split_type must be either 'train,test' or 'train,test,val'")
        
        if not abs(sum(split_ratios) - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
        
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




