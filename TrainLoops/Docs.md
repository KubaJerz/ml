# Neural Network Training Scripts






## Description for TrainLoop02 (tl02.py)

This script provides a basic framework for training neural network models using PyTorch. 

### Dependencies

- torch
- argparse
- tqdm
- matplotlib
- torcheval
- dill (as a replacement for pickle)

### Components

#### Data Loading

- Uses `dill` to load custom datasets from pickle files.
- Creates DataLoadersby passing in the Dataset.

- <span style="color:pink;"> **NOTE**  </span> these dataset are predefined in via the DataBuilder then same using `dill` as a pkl file  
- <span style="color:pink;"> **NEED**  </span> to use `dill` as its non lazy like pickle and will acualy save he class structure other wise there will be and error  

#### Training Function

The `train` function:
- Trains the model for a specified number of epochs.
- Computes loss and F1 score for both training and validation sets.
- Plots metrics every 2 epochs.
- <span style="color:pink;"> **NOTE**  </span> the tl02 version is built for **multi-class** 

#### Metric Visualization

The `plot_combined_metrics` function creates and saves plots for:
- Training and validation loss
- Training and validation F1 scores

#### Main Execution Flow

1. Parse command-line arguments
2. Set up device (CPU/GPU)
3. Load(from resume) or initialize(new from def) the model
6. Train the model
7. Save the trained model (save at location of the training script)

### Usage
```python script_name.py <training_id> <model_path> <data_path> --train_batch_size <size> [optionl args...] ```


#### Arguments

- `training_id`: Unique identifier for the training run
- `--resume` or `-r`: Flag to resume training from a saved model
- `model_path`: Path to the model definition or saved model
- `data_path`: Directory containing train.pkl and test.pkl
- `--train_batch_size` or `-trbs`: Batch size for training (required)
- `--test_batch_size` or `-tebs`: Batch size for testing (default: full batch (-1))
- `--epochs`: Number of training epochs (default: 15)

### <span style="color:red;"> **Notes**  </span>

- The script expects the model's forward pass to return logits.
- Cross-Entropy Loss is used as the default criterion.
- Adam is used as the default optimizer.
- The script supports resuming training from a saved model.