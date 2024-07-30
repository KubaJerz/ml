# Dataset Creation Templates for PyTorch Training






## Description for 0_csv_builder.ipynb (0_csv_builder.ipynb)

This Jupyter Notebook serves as a template for creating and saving PyTorch dataset objects from raw data CSV. The datasets are saved as pickle/(DILL) files, which can be easily loaded into the training script.

### Overview

This template demonstrates how to:
1. Load raw data from a CSV file
2. Create separate training and testing datasets via random perputation
3. Save these datasets as pickle files for later use in a PyTorch training pipeline

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



```
train_set = TrainDataSet(raw_data_path, train_percent, labels_col)
test_set = TestDataSet(raw_data_path, train_percent, labels_col, train_set.rows)
save_dataset(train_set, path_to_save_dir, 'train.pkl')
save_dataset(test_set, path_to_save_dir, 'test.pkl')
```