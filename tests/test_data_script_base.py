import pytest
import torch
from torch.utils.data import Dataset
from ml_framework.data_script.BaseDataScript import BaseDataScript

class MockDataset(Dataset):
    def __init__(self, size=100):
        self.data = [(torch.tensor([i]), torch.tensor([i])) for i in range(size)]
        self.size = size
    
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.data[idx]

class FakeDataScript(BaseDataScript):
    def get_datasets(self):
        pass
    
    def get_data_loaders(self):
        pass

class TestBaseDataScript:
    @pytest.fixture
    def mock_dataset(self):
        return MockDataset(100)

    def test_valid_with_test_config_initialization(self, valid_data_config_with_test):
        """Test initialization with valid [train,dev] config"""
        test_script = FakeDataScript(valid_data_config_with_test)
        assert test_script.config == valid_data_config_with_test


    def test_valid_without_test_config_initialization(self, valid_data_config_without_test):
        """Test initialization with valid [train,dev,test] config"""
        test_script = FakeDataScript(valid_data_config_without_test)
        assert test_script.config == valid_data_config_without_test

    def test_missing_required_fields(self):
        invalid_config = {'split_type': 'train,dev'}
        with pytest.raises(ValueError, match="Path must be absolute: No 'absolute_path' was provided"):
            FakeDataScript(invalid_config)

    def test_invalid_data_types(self, invalid_data_config_with_test):
        """Test initialization with invalid data types"""
        with pytest.raises(ValueError, match="seed must be of type: int"):
            FakeDataScript(invalid_data_config_with_test)

    def test_invalid_split_ratios00(self, invalid_datasplit_config_without_test):
        with pytest.raises(ValueError, match="Split ratios must sum to 1"):
            FakeDataScript(invalid_datasplit_config_without_test)

    def test_invalid_split_ratios01(self, invalid_datasplit_config_with_test):
        with pytest.raises(ValueError, match="Split ratios must sum to 1"):
            FakeDataScript(invalid_datasplit_config_with_test)

    def test_train_dev_split(self, valid_data_config_without_test, mock_dataset):
        """Test train/dev split"""
        test_script = FakeDataScript(valid_data_config_without_test)
        train_dataset, dev_dataset = test_script._split_dataset(mock_dataset)
        
        assert len(train_dataset) == 80  # 80% of 100
        assert len(dev_dataset) == 20    # 20% of 100
        
        train_indices = set(train_dataset.indices)
        dev_indices = set(dev_dataset.indices)
        assert not train_indices.intersection(dev_indices)
        assert len(train_indices) + len(dev_indices) == len(mock_dataset)

    def test_train_dev_test_split(self, valid_data_config_with_test,  mock_dataset):
        """Test train/dev/test split"""
        test_script = FakeDataScript(valid_data_config_with_test)
        train_dataset, dev_dataset, test_dataset = test_script._split_dataset(mock_dataset)
        
        assert len(train_dataset) == 70  # 70% of 100
        assert len(dev_dataset) == 15    # 15% of 100
        assert len(test_dataset) == 15   # 15% of 100
        
        train_indices = set(train_dataset.indices)
        dev_indices = set(dev_dataset.indices)
        test_indices = set(test_dataset.indices)
        assert not train_indices.intersection(dev_indices)
        assert not train_indices.intersection(test_indices)
        assert not dev_indices.intersection(test_indices)

    def test_seed_reproducibility(self, valid_data_config_without_test, mock_dataset):
        """Test that using the same seed produces the same splits"""
        test_script1 = FakeDataScript(valid_data_config_without_test)
        test_script2 = FakeDataScript(valid_data_config_without_test)
        
        split1_train, split1_dev = test_script1._split_dataset(mock_dataset)
        split2_train, split2_dev = test_script2._split_dataset(mock_dataset)
        
        assert split1_train.indices == split2_train.indices
        assert split1_dev.indices == split2_dev.indices

    def test_no_shuffle(self, valid_data_config_without_test, mock_dataset):
        """Test split behavior when shuffle is False"""
        valid_data_config_without_test["shuffle"] = False
        test_script = FakeDataScript(valid_data_config_without_test)
        train_dataset, dev_dataset = test_script._split_dataset(mock_dataset)
        
        assert train_dataset.indices == range(80)
        assert dev_dataset.indices == range(80, 100)