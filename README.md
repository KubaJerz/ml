# ml_framework

A streamlined framework for training PyTorch models with configurable experiment management.

## Installation
To install run ```pip install -e .``` when in the *ml/* dir.

## Quick Start
```python
from ml_framework.ExperimentRunner import ExperimentRunner

path = "/some/absolute/path/to/training/config.yaml"
runner = ExperimentRunner(path_to_config=path)

runner.run()
```
This is all the code that is needed ever for any training. The deails of what is in the config are in the config section. 

## Directory Structure
```python
ml_base_framework/          # Base framework code
├── src/
│   └── ml_framework/      
        ├── data/          # Base data handlers
        ├── models/        # Base model classes
        ├── training/      # Training logic
        └── utils/         # Utilities
└── setup.py              
```
```python
your_project/              # Your specific project
├── src/
│   └── models/           # Project specific models
├── configs/              # YAML configs
├── experiments/          # Results directory
└── requirements.txt
```

## Data Scripts
- Inherit from BaseDataScript

**MUST IMPLIMENT:**
```python
    @abstractmethod
    def get_datasets(self):
        """
        Get datasets based on configuration.
        
        Returns:
            If split_type is "train,dev":
                tuple: (train_dataset, dev_dataset)
            If split_type is "train,dev,test":
                tuple: (train_dataset, dev_dataset, test_dataset)
        """
    
    @abstractmethod
    def get_data_loaders(self):
        """
        Get loaders based on configuration.
        
        Returns:
            If split_type is "train,dev":
                tuple: (train_dataset, dev_dataset)
            If split_type is "train,dev,test":
                tuple: (train_dataset, dev_dataset, test_dataset)
        """
```
## Model Structure
**THE CONSTRUCTOR MUST HAVE:**

``` (input_size, input_channels, output_size, num_classes) ```

These will be passed to the model
```python
def __init__(self, input_size, input_channels, output_size, num_classes, hyperparam00(optional)):
    super().__init__()
```

## Modes 
### Single Mode 
Basic training of a single model:

```yaml
experiment:
  mode: "single"
  # other config options...
```

### Resume mode
Continue training from a checkpoint:
```yaml
experiment:
  mode: "resume"

resume:
  model_path: "/path/to/model.pth"
  metrics_path: "/path/to/metrics.json"
  resume_type: "full"  # Options: "full", "f1", "loss"
```

### Random Search Mode
Hyperparameter optimization through random search:

```yaml
experiment:
  mode: "random_search"

search_space:
  learning_rate:
    min: 0.0001
    max: 0.01
  train_batch_size: [32, 64, 128]
  # other hyperparameters...

sampling_control:
  num_trials: 5
  seed: 45
```

## Config Guide

### Experiment Settings

```yaml
experiment:
  name: "experiment_name"
  mode: "single"  # or "resume", "random_search"
  project_root: "/absolute/path/to/project"
```

### Data Configuration

```yaml
data:
  data_absolute_path: "/path/to/data"
  script_absolute_path: "/path/to/data_script.py"
  use_full: True
  use_percent: 0.10

  split_type: "train,dev"  # or "train,dev,test"
  split_ratios: [0.8, 0.2] # or [0.8, 0.1. 0.1]
  shuffle: True
  prevent_data_leakage: True
  seed: 69
  train_batch_size: 512
  dev_batch_size: -1
  test_batch_size: -1
  #num_workers: 2
  #pin_memory: true

  input_size: 1000
  input_channels: 1
  output_size: 10
  num_classes: 10
```
### Model Configuration

```yaml
model:
  absolute_path: "/path/to/basic_mlp.py"
```

### Training Configuration

```yaml
training:
  epochs: 100
  optimizer: "Adam"
  learning_rate: 0.001
  criterion: "CrossEntropyLoss"
  device: "cuda"  # or "cpu"
```

### Hyperparameter Configuration:
These get unpacked in the model constructor

```yaml
parameters:
  param: none
```

### Callbacks
```yaml
callbacks:
  plot_metrics_live: True
  best_dev_f1: True
  best_dev_loss: True
  early_stopping: True
  early_stopping_patience: 3
```

## Model Evaluation
The framework provides two evaluation tools for analyzing training results: ```SingleModelEvaluator``` and ```MultiModelEvaluator```

### Single Model Evaluation

```python
from ml_framework.evaluation.single_eval import SingleModelEvaluator

# Point to experiment directory
evaluator = SingleModelEvaluator("/path/to/experiment/directory")
evaluator.evaluate()
```

### Multi-Model Evaluation

```python
from ml_framework.evaluation.multi_eval import MultiModelEvaluator

# Point to directory containing multiple model runs
evaluator = MultiModelEvaluator("/path/to/search/experiments")
evaluator.evaluate()
```

# IMPORTANT NOTES

## Random Search Configuration

- When using random search mode, any configuration parameters set in the main config will be shared across all models
- For example, setting ```num_workers: 5``` in the main config will apply to all sampled models
- Only parameters defined in ```search_space``` will be randomly sampled and then overwritten; all other parameters remain constant

## Data Scripts

- The data script component is highly customizable
- Any additional parameters added to the data section of the config will be passed to your data script
- This allows for custom functionality without modifying the framework
- The ```use_full``` and ```use_percent``` parameters for partial dataset training are example implementations from the EEG data script

## Callbacks
    
### Live Plotting

- ```plot_metrics_live: True``` enables real-time metric plotting
- Plots are updated every other epoch when enabled
- If disabled, plots are only generated at training completion

### Early Stopping
- When resuming training, the early stopping counter resets to 0
- The best metric value is preserved from the previous training session
- This allows for proper continuation of training while maintaining historical best performance

## Model Requirements

### Model Outputs

- When using CrossEntropyLoss, models must output logits
- If using different loss functions, adjust model outputs accordingly
- Always check loss function requirements when designing model architecture

### Model Initialization

``` def __init__(self, input_size, input_channels, output_size, num_classes, hyperparams): ```
- All models must implement this initialization signature
- The hyperparams dictionary can be empty or contain custom parameters
- Custom hyperparameter names in dict and as param should match. They will be unpacked so they must match

## Metric Naming Conventions

- Validation metrics are always prefixed with ```"dev_"```
Examples:

```dev_loss```: Loss on validation set

```dev_f1```: F1 score on validation set


- This naming convention applies regardless of whether using train/val or train/val/test splits

