from .ExperimentMode import ExperimentMode
from typing import Dict, Any
from ..utils import check_field, check_section_exists

class SingleMode(ExperimentMode):
    def __init__(self, config):
        self.config = config
        self.dir = None
        self.training_loop = None
    
    def execute(self):
        self._setup_model() #when we implinet this go back to validate_mode_specific_config_structure and fic this
        self._setup_data()
        self._setup_training()
        self._train()

    def validate_mode_specific_config_structure(self):
        
        if self.config.get('experiment', {}).get('mode') != 'single':
            raise ValueError("Mode must be 'single' to use SingleMode")
 
        check_section_exists(self.config, 'data')
        check_section_exists(self.config, 'training')
        
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
        pass

    def _setup_data(self):
        pass

    def _setup_training(self):
        pass

    def _train(self):
        pass



