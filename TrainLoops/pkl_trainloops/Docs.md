# Neural Network Training Scripts


## Description for TrainLoop04 (tl04.py)

This script provides a basic framework for training neural network models using PyTorch. 


### Usage
```python3 tl04.py <training_id> --resume(optional) -(if resume then types: full, f1, loss) <model_path> <data_path> --train_batch_size <size> [optional args...]' ```

### Arguments

- `training_id`: Unique identifier for the training run. (**Required**)
-  `model_path`: Path to the model definition or saved model. (**Required**)
-  `data_path`: Directory containing the train.pkl and test.pkl files. (**Required**)
-  `--train_batch_size` or `-trbs`: Batch size for training (required). Use -1 for full batch. (**Required**)
-    `--test_batch_size` or `-tebs`: Batch size for testing (default: full batch (-1)).
-    `--epochs`:  Number of training epochs (default: 15).
-    `--resume` or `-r`: Flag to resume training from a saved model. Requires one of the following:
        - `-full`: Resume from the very end of the last model's training.
        - `-f1`: Resume from the best F1 score of the model.
        - `-loss`: Resume from the best loss of the model.

### Data Loading
- Uses `dill` to load custom datasets from pickle files.
- Creates DataLoadersby passing in the Dataset.

- <span style="color:pink;"> **NOTE**  </span> these dataset are predefined in via the DataBuilder then same using `dill` as a pkl file  
- <span style="color:pink;"> **NEED**  </span> to use `dill` as its non lazy like pickle and will acualy save he class structure other wise there will be and error  

### Model Saving
When we reach a new best f1 model or loss model we revome the old best and save the new best.

At the end you will have three models: `best_f1` `best_loss` and then `full`

*important note*: we use `<` and `>` when saving models meaning if they are the same loss or f1 we prior the older one and **will not** save the new one

### Notes

- The script expects the model's forward pass to return logits.
- Cross-Entropy Loss is used as the default criterion.
- Adam is used as the default optimizer.
- The script supports resuming training from a saved model.
    - <span style="color:red;"> **ENSURE THAT THE MODEL type (full,f1,or loss) MATCHES UP WITH THE FLAG PASSED **</span> this will be fixed in later tl and not allowed
    - if you pass in `-r` or `--resume` then your must specify what type of resume
        - `full` will train from the end of the best previous model
        - `f1` will start train from the last best f1
        - `loss` will start from the last best loss


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
- <span style="color:pink;"> **NOTE**  </span> the tl02 version is built for **multi-class** *it assumes that the model has a `.num_classes` atribute used for the f1 score calc*

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
```python tl02.py <training_id> <model_path> <data_path> --train_batch_size <size> [optionl args...] ```


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
