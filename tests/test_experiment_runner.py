import pytest
from ml_framework.ExperimentRunner import ExperimentRunner

class TestExperimentRunner:
    def test_load_config_success(self, sample_good_config_path) -> None:
        experiment_runner = ExperimentRunner(sample_good_config_path)
        assert isinstance(experiment_runner.config, dict)
        assert all(key in experiment_runner.config for key in ['experiment', 'data', 'model', 'parameters', 'training', 'callbacks'])
        
        assert experiment_runner.config['experiment']['name'] == 'tester00'
        assert experiment_runner.config['data']['input_size'] == 5000
        assert experiment_runner.config['parameters']['depth'] == 5
        assert experiment_runner.config['training']['epochs'] == 15
        
    def test_nonexist_yaml(self):
        with pytest.raises(FileNotFoundError):
            experiment_runner = ExperimentRunner('./fake_config.yaml')
    
    def test_empty_yaml(self, sample_empty_config_path):
        with pytest.raises(ValueError, match="Missing experiment section"):
            experiment_runner = ExperimentRunner(sample_empty_config_path)

    def test_init(self, sample_good_config_path) -> None:
        experiment_runner = ExperimentRunner(sample_good_config_path)
        assert experiment_runner.path_to_config == sample_good_config_path
