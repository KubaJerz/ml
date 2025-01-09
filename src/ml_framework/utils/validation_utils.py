from typing import Any
from pathlib import Path

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
    path =  Path(experiment['project_root'])
    if not path.is_absolute():
        raise ValueError(f"Path must be absolute: {path}")
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
        'absolute_path': str,
        'script_name': str,
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
    
    _validate_data_path(data_config.get('absolute_path', f"No 'absolute_path' was provided"))


    for field_name, field_type in required_fields.items():
        is_sequence = isinstance(field_type, tuple)
        check_field(data_config, field_name, field_type[1] if is_sequence else field_type, is_sequence)
    
    validate_split_configuration(data_config['split_type'], data_config['split_ratios'])
    return True

def validate_model_config(model_config):
    check_field(model_config, 'absolute_path', str)
    path =  Path(model_config['absolute_path'])
    if not path.is_absolute():
        raise ValueError(f"Path must be absolute: {path}")
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

def _validate_data_path(path):
        path =  Path(path)
        if not path.is_absolute():
            raise ValueError(f"Path must be absolute: {path}")
        
        if not path.exists():
            raise FileNotFoundError(f"Data path does not exist: {path}")
        
        pt_files = list(path.glob('*.pt'))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {path}")
