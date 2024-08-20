# Dataset Creation Templates for PyTorch Training






## Description for CSV Builder (0_csv_builder.ipynb)

This Jupyter Notebook serves as a template for creating and saving PyTorch dataset objects from raw data CSV. The datasets are saved as pickle/(DILL) files, which can be easily loaded into the training script.


### Dependencies

- torch
- numpy
- pandas
- dill (as a replacement for pickle its better)

### Key Components

#### Configuration

Adjust these variables according to your specific dataset:

```python
labels_col = 13  # Column index for labels
raw_data_path = 'Data/Raw/heart.csv'  # Absolute path to raw data
train_percent = 0.7  # Percentage of data for training
test_percent = 1 - train_percent
path_to_save_dir = 'Data/Datasets/Heart'  # Directory to save processed datasets
```
### Usage Instructions

Copy this notebook to your project directory.
Modify the configuration variables to match your dataset:

- Set the correct labels_col for your data
- Update raw_data_path to the absolute path of your CSV file
- Adjust train_percent if needed
- Set path_to_save_dir to your desired output directory


Run the notebook to create and save your datasets

### Notes

- Ensure your raw data is in CSV format
- The script assumes all data can be converted to float32
- Column names in the raw data are replaced with numeric indices
- Datasets are saved using dill, which offers more flexibility than the standard pickle module



```python
train_set = TrainDataSet(raw_data_path, train_percent, labels_col)
test_set = TestDataSet(raw_data_path, train_percent, labels_col, train_set.rows)
save_dataset(train_set, path_to_save_dir, 'train.pkl')
save_dataset(test_set, path_to_save_dir, 'test.pkl')
```



## Description for Tensor Builder (0_tensor_builder.ipynb)

This Jupyter Notebook serves as a template for creating and saving PyTorch dataset objects from raw data pt files. The datasets are saved as pickle/(DILL) files, which can be easily loaded into the training script.

### Dependencies

- torch
- dill (as a replacement for pickle its better)

### Key Components

#### Configuration

Adjust these variables according to your specific dataset:

```python
raw_data_path = 'Data/Raw/heart.pt'  # Absolute path to raw data
train_percent = 0.7  # Percentage of data for training
test_percent = 1 - train_percent
path_to_save_dir = 'Data/Datasets/Heart'  # Directory to save processed datasets
```
### Usage Instructions

Copy this notebook to your project directory.
Modify the configuration variables to match your dataset:

- Update raw_data_path to the absolute path of your `pt` file
- Adjust train_percent if needed
- Set path_to_save_dir to your desired output directory


Run the notebook to create and save your datasets

### Notes

- Ensure your raw data is in `pt` format
    - so thse means `torch.load(path)` returns the `X,y` for the full dataset
- The script assumes all data can be converted to float32
- Datasets are saved using dill, which offers more flexibility than the standard pickle module



```python
train_set = TrainDataSet(raw_data_path, train_percent)
test_set = TestDataSet(raw_data_path, train_percent, train_set.rows)

save_dataset(train_set, path_to_save_dir, 'train.pkl')
save_dataset(test_set, path_to_save_dir, 'test.pkl')
```

## Description for Multi Tensor Builder (0_multiTensor_builder.ipynb)

This Jupyter Notebook serves as a template for creating and saving PyTorch dataset objects from <ins>MULTIPLE</ins> raw data pt files. The datasets are saved as pickle/(DILL) files, which can be easily loaded into the training script.

### Overview

To use this template you need to jsut pass in the below paths to the data:


```python
raw_data_dir = '/home/kuba/Documents/Data/Raw/RatEEG_R/'     #THIS NEEDS TO BE ABSOLUTE PATH
train_percent = 0.7
test_percent = 1 - train_percent
path_to_save_dir = '/home/kuba/Documents/Data/Datasets/RatEEG_D/DS01' #path to save the data too

```