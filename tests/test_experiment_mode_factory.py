from ml_framework.ExperimentModeFactory import ExperimentModeFactory
from ml_framework.modes.SingleMode import SingleMode
import pytest
import re
from pathlib import Path

valid_modes = ['single'] #, 'random_search', 'resume'] 

class TestExperimentModeFactory:
    def test_verify_valid_mode(self, samp_good_config):
        (Path(samp_good_config['experiment']['project_root']) / "experiments").mkdir()
        factory = ExperimentModeFactory(samp_good_config)
        try: 
            factory._verify_valid_mode('single')
        except Exception as e:
            pytest.fail(f'Raised exception {e}')

    def test_fail_verify_valid_mode(self, samp_good_config):
        (Path(samp_good_config['experiment']['project_root']) / "experiments").mkdir()
        invaild_mode = 'no vaild'
        factory = ExperimentModeFactory(samp_good_config)
        with pytest.raises(TypeError, match=re.escape(f"Invalid mode type '{invaild_mode}'. Must be one of: {list(factory._valid_modes.keys())}")):
            factory._verify_valid_mode(invaild_mode)

    def test_create_mode(self, samp_good_config):
        (Path(samp_good_config['experiment']['project_root']) / "experiments").mkdir()
        factory = ExperimentModeFactory(samp_good_config)
        mode = factory.create_mode()
        assert isinstance(mode, SingleMode) == True

    def test_capital_create_mode(self, samp_good_config):
        (Path(samp_good_config['experiment']['project_root']) / "experiments").mkdir()
        samp_good_config['experiment']['mode'] = 'SINGLE'
        factory = ExperimentModeFactory(samp_good_config)
        mode = factory.create_mode()
        assert isinstance(mode, SingleMode) == True

    def test_fail_create_mode(self, samp_good_config):
        (Path(samp_good_config['experiment']['project_root']) / "experiments").mkdir()
        invaild_mode = 'no vaild'
        samp_good_config['experiment']['mode'] = invaild_mode
        factory = ExperimentModeFactory(samp_good_config)
        with pytest.raises(TypeError, match=re.escape(f"Invalid mode type '{invaild_mode}'. Must be one of: {list(factory._valid_modes.keys())}")):
            mode = factory.create_mode()
            assert isinstance(mode, SingleMode) == False

    def test_no_mode_provides(self, samp_good_config):
        (Path(samp_good_config['experiment']['project_root']) / "experiments").mkdir()
        del samp_good_config['experiment']['mode']
        factory = ExperimentModeFactory(samp_good_config)
        with pytest.raises(TypeError, match=re.escape(f"Invalid mode type 'no_mode_provided'. Must be one of: {list(factory._valid_modes.keys())}")):
            mode = factory.create_mode()
            assert isinstance(mode, SingleMode) == False
