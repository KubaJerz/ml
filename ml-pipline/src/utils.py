from typing import Dict, Any

def check_field(config, field_name, field_type):
    if field_name not in config:
        raise ValueError(f"Missing {field_name}")
    if not isinstance(config[field_name], (field_type)):
        raise ValueError(f"{field_name} must be of type: {field_type.__name__}")
    return True


def check_section_exists(config, section_name):
    if section_name not in config:
        raise ValueError(f"Missing {section_name} section")
    return True