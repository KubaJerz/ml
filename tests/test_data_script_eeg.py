import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil
from torch.utils.data import ConcatDataset, Subset, DataLoader
from ml_framework.data_script.EEGDataScript import EEGDataScript

class TestEEGDataScript:

    @pytest.fixture
    def mock_data_dir(self):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        for i in range(3):
            data = (torch.randn(10, 5), torch.randint(0, 2, (10,)))
            torch.save(data, Path(temp_dir) / f'data_{i}.pt')
            
        with open(Path(temp_dir) / 'invalid.pt', 'w') as f:
            f.write('invalid data')
            
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_load_datasets_multiple_files(self, valid_data_config_without_test, mock_data_dir):
        """Test loading multiple dataset files correctly"""
        valid_data_config_without_test['data_absolute_path'] = mock_data_dir
        script = EEGDataScript(valid_data_config_without_test)
        
        dataset = script._load_datasets()
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 30  # 3 files * 10 samples each

    def test_load_datasets_invalid_format(self, valid_data_config_without_test, mock_data_dir):
        """Test handling of invalid file formats"""
        # make a invalid file in same dir
        Path(mock_data_dir+'/invalid.txt').touch()
        
        valid_data_config_without_test['data_absolute_path'] = mock_data_dir
        script = EEGDataScript(valid_data_config_without_test)
        
        dataset = script._load_datasets()
        assert isinstance(dataset, ConcatDataset)
        #  only load valid  .pt files
        assert len(dataset) == 30  # 3 valid files * 10 samples

    def test_get_datasets_tuple_structure(self, valid_data_config_without_test, valid_data_config_with_test, mock_data_dir):
        """Test correct tuple structure from get_datasets"""
        valid_data_config_without_test['data_absolute_path'] = mock_data_dir
        script = EEGDataScript(valid_data_config_without_test)
        
        datasets = script.get_datasets()
        assert len(datasets) == 2
        assert isinstance(datasets[0], Subset)
        assert isinstance(datasets[1], Subset)
        
        #train,dev,test split
        script = EEGDataScript(valid_data_config_with_test)
        datasets = script.get_datasets()
        assert len(datasets) == 3
        assert all(isinstance(d, Subset) for d in datasets)

    def test_get_datasets_subset_creation(self, valid_data_config_without_test, mock_data_dir):
        """Test subset creation functionality"""
        valid_data_config_without_test['data_absolute_path'] = mock_data_dir
        valid_data_config_without_test['use_full'] = False
        valid_data_config_without_test['use_percent'] = 0.5
        script = EEGDataScript(valid_data_config_without_test)
        
        datasets = script.get_datasets()
        total_samples = sum(len(d) for d in datasets)
        assert total_samples == 15  # 50% of 30 samples

    def test_get_datasets_data_integrity(self, valid_data_config_without_test, mock_data_dir):
        """Test data integrity through transformations"""
        valid_data_config_without_test['data_absolute_path'] = mock_data_dir
        script = EEGDataScript(valid_data_config_without_test)
        
        datasets = script.get_datasets()
        
        # Check first sample from each dataset
        for dataset in datasets:
            x, y = dataset[0]
            assert isinstance(x, torch.Tensor)
            assert isinstance(y, torch.Tensor)
            assert x.shape[-1] == 5  # Match input dimension

    @patch('torch.cuda.is_available', return_value=True)
    def test_get_data_loaders_configuration(self, mock_cuda, samp_good_config, mock_data_dir):
        """Test DataLoader configuration"""
        samp_good_config['data']['data_absolute_path'] = mock_data_dir
        samp_good_config['data']['train_batch_size'] = 16
        samp_good_config['data']['num_workers'] = 2
        samp_good_config['data']['pin_memory'] = True
        script = EEGDataScript(samp_good_config['data'])
        
        loaders = script.get_data_loaders()
        _, dev_set, test_set = script.get_datasets()
        
        for loader in loaders:
            assert isinstance(loader, DataLoader)
            assert loader.num_workers == 2
            assert loader.pin_memory is True
        
        train_loader, dev_loader, test_loader = loaders
        assert train_loader.batch_size == 16
        assert dev_loader.batch_size == len(dev_set)
        assert test_loader.batch_size == len(test_set)

    def test_get_data_loaders_batch_sizes(self, valid_data_config_without_test, mock_data_dir):
        """Test correct batch size handling  when the batch is bigger than the dataset"""
        valid_data_config_without_test['data_absolute_path'] = mock_data_dir
        valid_data_config_without_test['train_batch_size'] = 100
        valid_data_config_without_test["dev_batch_size"]
        script = EEGDataScript(valid_data_config_without_test)

        loaders = script.get_data_loaders()
        train_loader, dev_loader = loaders

        x, y = next(iter(train_loader))
        assert x.shape[0] <= 100 
        
        #dev size specified so they should be 32 or less
        x, y = next(iter(dev_loader))
        assert x.shape[0] <= 32 


    def test_get_data_loaders_worker_count(self, valid_data_config_without_test, mock_data_dir):
        """Test worker count handling"""
        valid_data_config_without_test['data_absolute_path'] = mock_data_dir
        valid_data_config_without_test['num_workers'] = 0  # Test single-process loading
        script = EEGDataScript(valid_data_config_without_test)
        
        loaders = script.get_data_loaders()
        for loader in loaders:
            assert loader.num_workers == 0
            
        # Test with multiple workers
        valid_data_config_without_test['num_workers'] = 2
        script = EEGDataScript(valid_data_config_without_test)
        loaders = script.get_data_loaders()
        for loader in loaders:
            assert loader.num_workers == 2

    def test_memory_efficiency(self, valid_data_config_without_test, mock_data_dir):
        """Test memory usage with different configurations"""
        valid_data_config_without_test['data_absolute_path'] = mock_data_dir
        
        # Test with minimal memory settings
        valid_data_config_without_test.update({
            'pin_memory': False,
            'num_workers': 0,
            'use_full': False,
            'use_percent': 0.1
        })
        
        script = EEGDataScript(valid_data_config_without_test)
        datasets = script.get_datasets()
        
        # Verify subset size
        total_samples = sum(len(d) for d in datasets)
        assert total_samples == int(30 * 0.1)  # 10% of total samples