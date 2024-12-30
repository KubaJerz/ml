
    # def _config_path_to_dict(config_path):
    #     try:
    #         with open(config_path, 'r') as file:
    #             data = yaml.safe_load(file)
    #             return data
    #     except FileNotFoundError:
    #         print(f"Error: File '{config_path}' not found")
    #     except yaml.YAMLError as e:
    #         print(f"Error parsing YAML file: {e}")
    #     return None
    
    # def _validate_config(self):
    #     #make sure names match
    #     pass
    
    # def _setup_experimant_dir(self):
        
        
        
    #     # try:
    #     #     parent_dir = Path(self.config_path).parent
    #     #     parent_dir_name = parent_dir.name
        
    #     #     experiment_name_from_yamal = self.config.get('experiment').get('name')
        
    #     #     if not experiment_name_from_yamal:
    #     #         raise ValueError("Config file must contain 'experiment.name' field")
            
    #     #     if parent_dir_name != experiment_name_from_yamal:
    #     #         raise ValueError(f"Directory name '{parent_dir_name}' does not match experiment name '{experiment_name_from_yamal}' in config")
            
    #     #     if not parent_dir.exists():
    #     #         parent_dir.mkdir(parents=True, exist_ok=True)
            
    #     # except FileNotFoundError:
    #     #     raise FileNotFoundError(f"Config file not found at: {self.config_path}")
    #     # except yaml.YAMLError as yaml_error:
    #     #     raise yaml.YAMLError(f"Invalid YAML in config file: {yaml_error}")