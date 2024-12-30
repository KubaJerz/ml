from typing import Dict, Any

def check_section_exists(config, section_name):
    if section_name not in config:
        raise ValueError(f"Missing {section_name} section")
    return True

def check_field(config, field_name, field_type,is_sequence = False):
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