from .ExperimentMode import ExperimentMode
from typing import Dict, Any

import os, sys
import importlib
from ..utils.validation_utils import validate_mode_config, validate_dataloader_count, validate_data_config, validate_training_config, validate_model_config
import torch
from ..training.TrainingLoop import TrainingLoop
from ..callbacks.CallbackFactory import CallbackFactory



class SingleMode(ExperimentMode):
    def __init__(self, config, experiment_dir_constructor=None):
        self.config = config
        self.validate_mode_specific_config_structure()
        if experiment_dir_constructor == None:
            self.dir = super()._construct_experiment_path()
        else:
            self.dir = experiment_dir_constructor()
        super()._save_config()
        self._create_subdirectories()


    def execute(self):
        model = self._setup_model() 
        dataloaders = self._setup_data()
        training_loop = self._setup_training(model=model, dataloaders=dataloaders)
        training_loop.fit()

    def validate_mode_specific_config_structure(self):
        validate_mode_config(self.config, "single")

        validate_data_config(self.config['data'])
        validate_training_config(self.config['training'])
        validate_model_config(self.config['model'])
        return True 
    
    def _create_subdirectories(self):
        subdirs = ['models', 'metrics']
        for subdir in subdirs:
            (self.dir / subdir).mkdir(exist_ok=False)

    def _setup_model(self):
        model_path = self.config['model'].get('absolute_path')
        if not model_path:
            raise ValueError("Config missing 'absolute_path' in model section")

        if not os.path.isabs(model_path):
            raise ValueError(f"Model path must be absolute: {model_path}")
        if not os.path.exists(model_path):
            raise ValueError(f"Model file does not exist: {model_path}")
            
        try:
            dir_path, file_name = os.path.split(model_path)
            module_name = os.path.splitext(file_name)[0]
            
            sys.path.insert(0, dir_path)
            try:
                model_module = importlib.import_module(module_name)
                model_class = getattr(model_module, module_name.upper())
                
                model_params = {
                    'input_size': self.config['data']['input_size'],
                    'input_channels': self.config['data']['input_channels'],
                    'output_size': self.config['data']['output_size'],
                    'num_classes': self.config['data']['num_classes'],
                    'hyperparams': self.config['parameters']
                }
                return model_class(**model_params)
            finally:
                sys.path.pop(0)
        
        except ImportError as e:
            raise ValueError(f"Failed to import model module from '{model_path}': {e}")
        except AttributeError as e:
            raise ValueError(f"Failed to find model class in module: {e}. Model class name must be all capital version of the file name")
        except TypeError as e:
            raise ValueError(f"Model initialization failed - incorrect parameters: {e}")
        except Exception as e:
            raise ValueError(f"Error setting up model: {e}")

    def _setup_data(self):
        try:
            script_path = self.config['data'].get('script_absolute_path')
            if not script_path:
                raise ValueError("Config missing 'script_absolute_path' in data section")

            if not os.path.isabs(script_path):
                raise ValueError(f"Script path must be absolute: {script_path}")
            if not os.path.exists(script_path):
                raise ValueError(f"Script file does not exist: {script_path}")
            
            dir_path, file_name = os.path.split(script_path)
            module_name = os.path.splitext(file_name)[0]
            
            sys.path.insert(0, dir_path)
            try:
                data_module = importlib.import_module(module_name)
                
                data_script_class = getattr(data_module, module_name)
                
                data_script = data_script_class(self.config['data'])
                return data_script.get_data_loaders()
                
            finally:
                sys.path.pop(0)
                
        except ImportError as e:
            raise ValueError(f"Failed to import data script from '{script_path}': {e}")
        except AttributeError as e:
            raise ValueError(f"Failed to find data script class '{module_name}' in module: {e}")
        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

        # try:
        #     script_name = self.config['data']['script_name']
        #     module_name = os.path.splitext(script_name)[0]
        #     data_module = importlib.import_module('..data_script.'+module_name, package=__package__)
        #     data_script_class = getattr(data_module, module_name)
            
        #     data_script = data_script_class(self.config['data'])
        #     return data_script.get_data_loaders()
            
        # except ImportError as e:
        #     raise RuntimeError(f"Failed to import data script {script_name}: {e}")
        # except AttributeError as e:
        #     raise RuntimeError(f"Failed to find data script class {module_name}: {e}")
        # except Exception as e:
        #     raise RuntimeError(f"Error loading data: {e}")

    def _setup_training(self, model, dataloaders):
        validate_dataloader_count(dataloaders)

        training_params = self._get_training_parameters()
        training_components = self._initialize_training_components(model)
        return self._create_training_loop(model=model, dataloaders=dataloaders, training_params=training_params, **training_components)

    def _get_training_parameters(self) -> Dict[str, Any]:
        training_config = self.config.get('training', {})
        device = self._get_valid_device(training_config.get('device'))
        return {
            'total_epochs': training_config.get('epochs', 100),
            'device': device,
            'save_dir': self.dir,
            'save_full_model': training_config.get('save_full_model', True)
        }
    
    def _get_valid_device(self, requested_device) -> str:
        if not requested_device:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
            
        if requested_device == 'cpu':
            return 'cpu'
            
        if requested_device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise ValueError("CUDA device requested but CUDA is not available")
                
            if ':' in requested_device:
                device_id = int(requested_device.split(':')[1])
                if device_id >= torch.cuda.device_count():
                    raise ValueError(f"Requested GPU {device_id} but only {torch.cuda.device_count()} GPUs available")
            
            return requested_device
                    
    def _initialize_training_components(self, model: torch.nn.Module) -> Dict[str, Any]:
        training_config = self.config.get('training')
        metrics = self._initialize_metrics()
        
        return {
            'optimizer': self._create_optimizer(model, training_config),
            'criterion': self._create_criterion(training_config),
            'callbacks': self._setup_callbacks(self.config.get('callbacks', {}), metrics),
            'metrics': metrics
        }
    
    def _initialize_metrics(self):
        return {
            'train_loss': [],
            'dev_loss': [],
            'train_f1': [],
            'dev_f1': [],
            'best_dev_f1': float('-inf'),
            'best_dev_loss': float('inf')
        }
    
    def _create_optimizer(self, model, training_config):
        optimizer_name = training_config.get('optimizer', 'Adam')
        learning_rate = training_config.get('learning_rate', 0.001)
        
        optimizer_class = getattr(torch.optim, optimizer_name)
        return optimizer_class(model.parameters(), lr=learning_rate)

    def _create_criterion(self, training_config):
        criterion_name = training_config.get('criterion', 'CrossEntropyLoss')
        criterion_class = getattr(torch.nn, criterion_name)
        return criterion_class()

    def _setup_callbacks(self, callback_config, metrics):
        return CallbackFactory.setup_callbacks(callback_config, metrics)
    
    def _create_training_loop(self, model, dataloaders, training_params, **components) -> TrainingLoop:
        train_loader, dev_loader, *test = dataloaders
        
        return TrainingLoop(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test[0] if test else None,
            **training_params,
            **components
        )