# Neural Network Training Scripts


## Description for TrainLoop06 (tl06.py)

This script provides a basic framework for training neural network models using PyTorch with a custom dataset loader. We now use `getDataSet()` from <ins> custom module that must be imported and customized before running the script </ins> 

### Usage
```python3 tl06.py <training_id>  <model_path> --train_batch_size [optional args: [--resume] [-full|-f1|-loss] ...] ```

or 

``` run_training( training_id, model_path, sub_dir, train_percent, random_state, train_batch_size, test_batch_size, epochs, resume, full, f1, loss) ```

### Arguments

- `training_id`: Unique identifier for the training run. (**Required**)
-  `model_path`: Path to the model definition or saved model. `.pth` if resume and `.py` if new model (**Required**) <span style="color:pink;"> NOTE: When using the ``run_training()`` this can be a torch model object </span>
-  `--train_batch_size` or `-trbs`: Batch size for training (required). Use -1 for full batch. (**Required**)
-    `--test_batch_size` or `-tebs`: Batch size for testing (default: full batch (-1)).
-  `--train_percent`: Percentage of data to use for training (default: 0.7)
- `--random_state`: Random state for dataset split (default: 69)
-    `--epochs`:  Number of training epochs (default: 15).
-    `--resume` or `-r`: Flag to resume training from a saved model. Requires one of the following:
        - `-full`: Resume from the very end of the last model's training.
        - `-f1`: Resume from the best F1 score of the model.
        - `-loss`: Resume from the best loss of the model.

- `random_state` to use when shuffling the data. (The default is 69).
- `sub_dir` is used when we are instance training models (model path is a python object). In that case you must pass in a `sub_dir` where to save the model etc.
### Data Loading
- Uses a custom getDataSet function to load and split the dataset. **Must import**
- <span style="color:pink;"> NOTE </span> The dataset is now dynamically split into train and test sets using the getDataSet function.
- <span style="color:pink;"> NEED </span> to ensure that the getDataSet function is properly implemented and imported.


### Model Saving
When we reach a new best f1 model or loss model we revome the old best and save the new best.

At the end you will have three models: `best_f1` `best_loss` and then `full`

*important note*: we use `<` and `>` when saving models meaning if they are the same loss or f1 we prior the older one and **will not** save the new one

### Notes

- The script expects the model's forward pass to return logits.
- Macro F1 is used. If a Batch does not have any instances of it in the batch it is ignored in the calc
- The `util_tl.py` file must be in the same dir
- Cross-Entropy Loss is used as the default criterion.
- Adam is used as the default optimizer.
- The script supports resuming training from a saved model.
    - if you pass in `-r` or `--resume` then your must specify what type of resume
        - `full` will train from the end of the best previous model
        - `f1` will start train from the last best f1
        - `loss` will start from the last best loss





## Description for TrainLoop05 (tl05.py)

This script provides a basic framework for training neural network models using PyTorch with a custom dataset loader. We now use `getDataSet()` from <ins> custom module that must be imported and customized before running the script </ins> 

### Usage
```python3 tl05.py <training_id> [--resume] [-full|-f1|-loss] --train_batch_size <model_path> [optional args...] ```

### Arguments

- `training_id`: Unique identifier for the training run. (**Required**)
-  `model_path`: Path to the model definition or saved model. (**Required**)
-  `--train_batch_size` or `-trbs`: Batch size for training (required). Use -1 for full batch. (**Required**)
-    `--test_batch_size` or `-tebs`: Batch size for testing (default: full batch (-1)).
-  `--train_percent`: Percentage of data to use for training (default: 0.7)
- `--random_state`: Random state for dataset split (default: 69)
-    `--epochs`:  Number of training epochs (default: 15).
-    `--resume` or `-r`: Flag to resume training from a saved model. Requires one of the following:
        - `-full`: Resume from the very end of the last model's training.
        - `-f1`: Resume from the best F1 score of the model.
        - `-loss`: Resume from the best loss of the model.

### Data Loading
- Uses a custom getDataSet function to load and split the dataset. **Must import**
- <span style="color:pink;"> NOTE </span> The dataset is now dynamically split into train and test sets using the getDataSet function.
- <span style="color:pink;"> NEED </span> to ensure that the getDataSet function is properly implemented and imported.


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
