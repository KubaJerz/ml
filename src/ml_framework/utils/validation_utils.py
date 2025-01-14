from typing import Any
from pathlib import Path
import json

def check_section_exists(config, section_name):
    if section_name not in config:
        raise ValueError(f"Missing {section_name} section")
    return True

def check_field(config, field_name, field_type, is_sequence = False):
    if field_name not in config:
        raise ValueError(f"Missing {field_name}")
        
    value = config[field_name]
    
    if is_sequence:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field_name} must be a list or tuple")
            
        if not all(isinstance(item, field_type) for item in value):
            raise ValueError(f"All items in {field_name} must be of type: {field_type.__name__}")
    else:
        if not isinstance(value, field_type):
            raise ValueError(f"{field_name} must be of type: {field_type.__name__}")
            
    return True

def validate_core_config_structure(config):
    check_section_exists(config, 'experiment')
    experiment = config['experiment']
    check_field(experiment, 'name', str)
    check_field(experiment, 'mode', str)
    check_field(experiment, 'project_root', str)
    validate_path_is_absolute(Path(experiment['project_root']))
    validate_path_exists(Path(experiment['project_root']))

    check_section_exists(config, 'data')
    check_section_exists(config, 'model')
    check_section_exists(config, 'parameters')
    check_section_exists(config, 'training')
    check_section_exists(config, 'callbacks')

    return True
    
def validate_split_configuration(split_type, split_ratios):
    if split_type == "train,dev":
        if len(split_ratios) != 2:
            raise ValueError("train,dev split type requires exactly 2 split values")
    elif split_type == "train,dev,test":
        if len(split_ratios) != 3:
            raise ValueError("train,dev,test split type requires exactly 3 split values")
    else:
        raise ValueError(f"split_type must be either 'train,dev' or 'train,dev,test' NOT {split_type}")
    
    if not abs(sum(split_ratios) - 1.0) < 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
    
    return True

def validate_data_config(data_config):
    required_fields = {
        'data_absolute_path': str,
        'script_absolute_path': str,
        'split_type': str,
        'split_ratios': (list, float),
        'shuffle': bool,
        'seed': int,
        'train_batch_size': int,
        'dev_batch_size': int,
        'input_size': int,
        'input_channels': int,
        'output_size': int,
        'num_classes': int
    }
    for field_name, field_type in required_fields.items():
        is_sequence = isinstance(field_type, tuple)
        check_field(data_config, field_name, field_type[1] if is_sequence else field_type, is_sequence)

    validate_path_is_absolute(data_config.get('script_absolute_path'))
    validate_path_exists(data_config.get('script_absolute_path'))

    _validate_data_path(data_config.get('data_absolute_path', f"No 'data_absolute_path' was provided"))
    
    validate_split_configuration(data_config['split_type'], data_config['split_ratios'])
    return True

def validate_model_config(model_config):
    check_field(model_config, 'absolute_path', str)
    validate_path_is_absolute(model_config.get('absolute_path'))
    validate_path_exists(model_config.get('absolute_path'))
    return True

def validate_training_config(training_config):
    required_fields = {
        'epochs': int,
        'learning_rate': float
    }
    for field_name, field_type in required_fields.items():
        check_field(training_config, field_name, field_type)
    return True

def validate_dataloader_count(dataloaders):
    dataloader_count = len(dataloaders)
    if dataloader_count not in [2, 3]:
        raise ValueError(f"Expected 2 or 3 dataloaders, got {dataloader_count}")

def validate_mode_config(config, expected_mode):
    if config.get('experiment', {}).get('mode') != expected_mode:
        raise ValueError(f"Mode must be '{expected_mode}'")
    return True

def validate_metrics_structure(metrics):
    required_fields = {
        'train_loss': list,
        'dev_loss': list,
        'train_f1': list,
        'dev_f1': list,
        'best_f1_dev': float,
        'best_loss_dev': float,
    }
    
    for field_name, field_type in required_fields.items():
        check_field(metrics, field_name, field_type)
    return True

def validate_path_is_absolute(path):
    path =  Path(path)
    if not path.is_absolute():
        raise ValueError(f"Path must be absolute: {path}")

def validate_path_exists(path):
    path =  Path(path)
    if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

def _validate_data_path(path):
        path =  Path(path)

        validate_path_is_absolute(path)
        validate_path_exists(path)
        
        pt_files = list(path.glob('*.pt'))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {path}")
        
def validate_metrics_file_format(metrics_path: Path, required_metrics):
    validate_path_exists(metrics_path)
        
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in metrics file: {e}")
    except Exception as e:
        raise ValueError(f"Error reading metrics file: {e}")

    missing_metrics = [m for m in required_metrics if m not in metrics]
    if missing_metrics:
        raise ValueError(f"Metrics file missing required metrics: {missing_metrics}")
        
    return metrics

