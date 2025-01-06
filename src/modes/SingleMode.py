from .ExperimentMode import ExperimentMode
from typing import Dict, Any
import os, sys
import importlib
from utils.validation_utils import validate_mode_config, validate_data_config, validate_training_config, check_section_exists, validate_model_config
import torch
from training.TrainingLoop import TrainingLoop
from training.callbacks import EarlyStoppingCallback, PlotCombinedMetrics, BestMetricCallback



class SingleMode(ExperimentMode):
    def __init__(self, config):
        self.config = config
        self.dir = None

    
    def execute(self):
        model = self._setup_model() 
        dataloaders = self._setup_data() #when we implinet this go back to validate_mode_specific_config_structure and fic this
        training_loop = self._setup_training(model=model, dataloaders=dataloaders)
        training_loop.fit()

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
        validate_model_config(self.config['model'])
        model_path = self.config['model'].get('absolute_path')
        try:
            dir_path, file_name = os.path.split(model_path)
            module_name = os.path.splitext(file_name)[0]
            
            sys.path.append(dir_path)
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
        
        except ImportError as e:
            raise ValueError(f"Failed to import model module: {e}")
        except AttributeError as e:
            raise ValueError(f"Failed to find model class in module: {e}. Model class name msut be all capital verison of the file name")
        except TypeError as e:
            raise ValueError(f"Model initialization failed - incorrect parameters: {e}")
        except Exception as e:
            raise ValueError(f"Error setting up model: {e}")

    def _setup_data(self):
        try:
            script_name = self.config['data']['script_name']
            module_name = os.path.splitext(script_name)[0]
            data_module = importlib.import_module('data_script.'+module_name)
            data_script_class = getattr(data_module, module_name)
            
            data_script = data_script_class(self.config['data'])
            return data_script.get_data_loaders()
            
        except ImportError as e:
            raise RuntimeError(f"Failed to import data script {script_name}: {e}")
        except AttributeError as e:
            raise RuntimeError(f"Failed to find data script class {module_name}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading data: {e}")

    def _setup_training(self, model, dataloaders):
        try:
            training_config = self.config['training']
            metrics = {
                        'train_loss': [],
                        'dev_loss': [],
                        'train_f1': [],
                        'dev_f1': [],
                        'best_dev_f1': float('-inf'),
                        'best_dev_loss': float('inf'),
                    }
            
            optimizer = self._create_optimizer(model, training_config)
            criterion = self._create_criterion(training_config)
            callbacks = self._setup_callbacks(self.config['callbacks'], metrics)

            
            total_epochs = training_config.get('epochs', 100)
            device = training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            save_dir = self.dir
            save_full_model = training_config.get('save_full_model', True)

            if len(dataloaders) == 3:
                train_loader, dev_loader, test_loader = dataloaders
                return TrainingLoop(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    metrics=metrics,
                    total_epochs=total_epochs,
                    callbacks=callbacks,
                    device=device,
                    save_dir=save_dir,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    test_loader=test_loader,
                    save_full_model=save_full_model
                )
            elif len(dataloaders) == 2:
                train_loader, dev_loader = dataloaders
                return TrainingLoop(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    metrics=metrics,
                    total_epochs=total_epochs,
                    callbacks=callbacks,
                    device=device,
                    save_dir=save_dir,
                    train_loader=train_loader,
                    dev_loader=dev_loader,
                    save_full_model=save_full_model
                )
            else:
                raise RuntimeError(f"Unsupported number of dataloaders: {len(dataloaders)}. Must be 2 or 3.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to setup training loop: {str(e)}")

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
        callbacks = []
        
        if callback_config.get('early_stopping', False):
            best_val_so_far = metrics[f"best_{callback_config.get('early_stopping_monitor', 'dev_loss')}"]
            callbacks.append(EarlyStoppingCallback.EarlyStoppingCallback(
                best_val_so_far = best_val_so_far,
                patience=callback_config.get('early_stopping_patience', 10),
                monitor=callback_config.get('early_stopping_monitor', 'dev_loss')
            ))
        
        if callback_config.get('best_f1', True):
            callbacks.append(BestMetricCallback.BestMetricCallback(best_value=metrics['best_dev_f1'], metric_to_monitor='dev_f1'))

        if callback_config.get('best_loss', True):
            callbacks.append(BestMetricCallback.BestMetricCallback(best_value=metrics['best_dev_loss'], metric_to_monitor='dev_loss'))


        if callback_config.get('plot_combined_metric', True):
            callbacks.append(PlotCombinedMetrics.PlotCombinedMetrics())
            
        return callbacks