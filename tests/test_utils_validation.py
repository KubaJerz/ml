import pytest
from ml_framework.utils.validation_utils import *
import tempfile

global expected_mode
expected_mode = 'single' 

class TestValidationUtils:

    def test_validate_core_config_all_sections(self, valid_core_config):
        assert validate_core_config_structure(valid_core_config) is True

    @pytest.mark.parametrize("missing_section", ['experiment', 'data', 'model', 'parameters', 'training', 'callbacks'])
    def test_validate_core_config_missing_sections(self, valid_core_config, missing_section):
        """Test validation fails when required sections are missing"""
        del valid_core_config[missing_section]
        with pytest.raises(ValueError, match=f"Missing {missing_section} section"):
            validate_core_config_structure(valid_core_config)

    @pytest.mark.parametrize("field,invalid_value,expected_type", [('name', 123, 'str'), ('mode', ['invalid'], 'str'), ('project_root', 456, 'str')])
    def test_validate_core_config_invalid_types(self, valid_core_config, field, invalid_value, expected_type):
        """Test validation fails with invalid field types"""
        valid_core_config['experiment'][field] = invalid_value
        with pytest.raises(ValueError, match=f"{field} must be of type: {expected_type}"):
            validate_core_config_structure(valid_core_config)

    def test_validate_core_config_relative_path(self, valid_core_config):
        """Test validation fails with relative project root path"""
        valid_core_config['experiment']['project_root'] = 'relative/path'
        with pytest.raises(ValueError, match="Path must be absolute"):
            validate_core_config_structure(valid_core_config)

    @pytest.mark.parametrize("split_type,split_ratios,should_pass", [
        ('train,dev', [0.8, 0.2], True),
        ('train,dev,test', [0.7, 0.2, 0.1], True),
        ('train,dev', [0.7, 0.2, 0.1], False), 
        ('train,dev,test', [0.8, 0.2], False), 
        ('train,dev', [0.8, 0.3], False), 
        ('train,dev,test', [0.7, 0.2, 0.2], False),  
        ('invalid', [0.8, 0.2], False)
    ])
    def test_validate_split_configuration(self, split_type, split_ratios, should_pass):
        """Test split configuration validation with various scenarios"""
        if should_pass:
            assert validate_split_configuration(split_type, split_ratios) is True
        else:
            with pytest.raises(ValueError):
                validate_split_configuration(split_type, split_ratios)

    def test_validate_data_config_valid(self, valid_data_config):
        """Test validation of a complete, valid data configuration"""
        assert validate_data_config(valid_data_config) is True

    @pytest.mark.parametrize("missing_field", ['data_absolute_path', 'script_absolute_path', 'split_type', 'split_ratios','shuffle', 'seed', 'input_size', 'input_channels', 'output_size', 'num_classes'])
    def test_validate_data_config_missing_fields(self, valid_data_config, missing_field):
        """Test validation fails when required fields are missing"""
        del valid_data_config[missing_field]
        with pytest.raises(ValueError, match=f"Missing {missing_field}"):
            validate_data_config(valid_data_config)

    def test_validate_data_config_missing_absolute_path(self, valid_data_config):
        """Test validation fails when required fields are missing"""
        del valid_data_config['data_absolute_path']
        with pytest.raises(ValueError, match=f"Missing data_absolute_path"):
            validate_data_config(valid_data_config)

    @pytest.mark.parametrize("field, invalid_value, expected_type", [('data_absolute_path', 123, 'str'),('shuffle', 'true', 'bool'),('seed', 42.0, 'int'),('split_ratios', 0.8, 'list'), ('split_ratios', [0.8, 'invalid'], 'float')])
    def test_validate_data_config_invalid_types(self, valid_data_config, field, invalid_value, expected_type):
        """Test validation fails with invalid field types"""
        valid_data_config[field] = invalid_value
        with pytest.raises(ValueError):
            validate_data_config(valid_data_config)

    def test_validate_data_config_invalid_path(self, valid_data_config):
        """Test validation fails with various invalid path scenarios"""
        valid_data_config['data_absolute_path'] = 'relative/path'
        with pytest.raises(ValueError, match="Path must be absolute"):
            validate_data_config(valid_data_config)

        valid_data_config['data_absolute_path'] = '/path/does/not/exist'
        with pytest.raises(FileNotFoundError, match=f"Path does not exist: {valid_data_config['data_absolute_path']}"):
            validate_data_config(valid_data_config)


    def test_validate_data_config_invalid_split_config(self, valid_data_config):
        valid_data_config['split_type'] = 'invalid'
        with pytest.raises(ValueError):
            validate_data_config(valid_data_config)

        valid_data_config['split_type'] = 'train,dev'
        valid_data_config['split_ratios'] = [0.8, 0.3] 
        with pytest.raises(ValueError):
            validate_data_config(valid_data_config)

    def test_validate_metrics_structure(self):
        good_metrics = {
        'train_loss': [49, 54, 32, 24],
        'dev_loss': [49, 54, 32, 24],
        'train_f1': [49, 54, 32, 24],
        'dev_f1': [49, 54, 32, 24],
        'best_f1_dev': 89.0,
        'best_loss_dev': 2.5,
        }

        val = validate_metrics_structure(good_metrics)
        assert val == True
    
    def test_validate_mode_config(self, samp_good_config):
        try:
            validate_mode_config(samp_good_config, expected_mode)
        except Exception as e:
            pytest.fail(f"Raised exception {e}")

    def test_validate_training_config(self, samp_good_config):
        try:
            validate_training_config(samp_good_config['training'])
        except Exception as e:
            pytest.fail(f"Raised exception {e}")