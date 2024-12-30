import unittest
from unittest.mock import Mock, patch, MagicMock
import yaml

from ml_pipeline.src.ExperimentRunner import ExperimentRunner
from ml_pipeline.src.ExperimentModeFactory import ExperimentModeFactory, ExperimentModeError
from ml_pipeline.src.modes.SingleMode import SingleMode
from ml_pipeline.src.modes.RandSearchMode import RandSearchMode
from ml_pipeline.src.utils import check_field, check_section_exists

class TestExperimentModeFactory(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            'experiment': {
                'name': 'test_experiment',
                'mode': 'single',
                'version': '1.0'
            }
        }
        self.factory = ExperimentModeFactory(self.valid_config)

    def test_create_mode_with_valid_single_mode(self):
        """Test that factory creates SingleMode instance with valid config"""
        mode = self.factory.create_mode(self.valid_config)
        self.assertIsInstance(mode, SingleMode)
        self.assertEqual(mode.config, self.valid_config)

    # def test_create_mode_with_valid_random_search_mode(self):
    #     """Test that factory creates RandSearchMode instance with valid config"""
    #     config = self.valid_config.copy()
    #     config['experiment']['mode'] = 'random_search'
    #     mode = self.factory.create_mode(config)
    #     self.assertIsInstance(mode, RandSearchMode)

    def test_create_mode_with_invalid_mode(self):
        """Test that factory raises error with invalid mode type"""
        config = self.valid_config.copy()
        config['experiment']['mode'] = 'invalid_mode'
        with self.assertRaises(ExperimentModeError) as context:
            self.factory.create_mode(config)
        self.assertIn("Invalid mode type", str(context.exception))

    def test_create_mode_with_missing_mode(self):
        """Test that factory handles missing mode appropriately"""
        config = {'experiment': {'name': 'test'}}  # No mode specified
        with self.assertRaises(ExperimentModeError):
            self.factory.create_mode(config)

class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            'experiment': {
                'name': 'test_experiment',
                'mode': 'single',
                'version': '1.0'
            },
            'data': {
                'shuffle': 'true',
                'seed': 42
            },
            'model': {
                'absolute_path': '/path/to/model'
            },
            'parameters': {},
            'training': {}
        }
        self.config_path = 'test_config.yaml'

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=yaml.dump({'experiment': {'name': 'test'}}))
    def test_load_config(self, mock_open):
        """Test that config is loaded correctly from file"""
        runner = ExperimentRunner(self.config_path)
        self.assertIsNotNone(runner.config)
        mock_open.assert_called_once_with(self.config_path, 'r')

    def test_validate_core_config_structure_valid(self):
        """Test validation passes with valid config"""
        with patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=yaml.dump(self.valid_config)):
            runner = ExperimentRunner(self.config_path)
            self.assertIsNone(runner._validate_core_config_structure())

    def test_validate_core_config_structure_missing_section(self):
        """Test validation fails when required section is missing"""
        invalid_config = self.valid_config.copy()
        del invalid_config['experiment']
        with patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=yaml.dump(invalid_config)):
            runner = ExperimentRunner(self.config_path)
            with self.assertRaises(ValueError) as context:
                runner._validate_core_config_structure()
            self.assertIn("Missing experiment section", str(context.exception))

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('yaml.safe_load')
    def test_run_executes_mode(self, mock_yaml_load, mock_open):
        """Test that run method executes the mode correctly"""
        mock_yaml_load.return_value = self.valid_config
        mock_mode = Mock()
        
        with patch('ExperimentModeFactory.create_mode', return_value=mock_mode):
            runner = ExperimentRunner(self.config_path)
            runner.run()
            
            mock_mode.validate_mode_specific_config_structure.assert_called_once()
            mock_mode.setup_experimant_dir.assert_called_once()
            mock_mode.execute.assert_called_once()

class TestUtils(unittest.TestCase):
    def test_check_section_exists_valid(self):
        """Test check_section_exists with valid config"""
        config = {'test_section': {}}
        self.assertTrue(check_section_exists(config, 'test_section'))

    def test_check_section_exists_invalid(self):
        """Test check_section_exists with missing section"""
        config = {}
        with self.assertRaises(ValueError) as context:
            check_section_exists(config, 'test_section')
        self.assertIn("Missing test_section section", str(context.exception))

    def test_check_field_valid(self):
        """Test check_field with valid field"""
        config = {'test_field': 'test_value'}
        self.assertTrue(check_field(config, 'test_field', str))

    def test_check_field_invalid_type(self):
        """Test check_field with invalid field type"""
        config = {'test_field': 123}
        with self.assertRaises(ValueError) as context:
            check_field(config, 'test_field', str)
        self.assertIn("must be of type: str", str(context.exception))

    def test_check_field_missing(self):
        """Test check_field with missing field"""
        config = {}
        with self.assertRaises(ValueError) as context:
            check_field(config, 'test_field', str)
        self.assertIn("Missing test_field", str(context.exception))

    def test_check_field_sequence_valid(self):
        """Test check_field with valid sequence"""
        config = {'test_field': [1, 2, 3]}
        self.assertTrue(check_field(config, 'test_field', int, is_sequence=True))

    def test_check_field_sequence_invalid(self):
        """Test check_field with invalid sequence types"""
        config = {'test_field': [1, '2', 3]}
        with self.assertRaises(ValueError) as context:
            check_field(config, 'test_field', int, is_sequence=True)
        self.assertIn("All items in test_field must be of type: int", str(context.exception))

if __name__ == '__main__':
    unittest.main()