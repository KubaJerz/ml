# tensorMaker Dataset  Templates

## Description for 01 Train Wrapper (tw01.py )

This python script will need to be customized for each uscase.
The general idea is that the user specifys the criteria and how many models then the script randomly samples and builds and trains the models 

```python
python3 tw01.py hyper_search_id --num_models 8 --epochs 100
````
### Key Components

#### Configuration

Adjust these variables according to your specific wants:

```python
    hyper_ranges = {
        'train_batch_size': [32,64,128,256,512],
        'learning_rate': (0.0001, 0.1),
        'hidden_blocks': 4,
        'layer_depth': [[4, 8, 16, 32],[2,4,8,16]],
        'dropout_rate': (0.1, 0.5),
        'activation': ['ReLU'], #['ReLU', 'tanh', 'sigmoid'],
        'normalization': ['batch', 'layer']#['none', 'batch', 'layer']
    }
```

**MAKE SURE** to import the right training loop and the right model(model skeleton):

```python
'''MAKE SURE TO UPDATE THESE'''
from model import MODEL #this is the base model that when passed in new hyper params will create new models
from tl06 import run_training
```

Adjust the ``tl`` you are importing to your needs. If it need a ```tensor_builder``` file (file that defines the dataset) make sure to have that file in the working dir


### Usage Instructions

Copy this scrpit to your project directory.
Modify the configuration variables to match your dataset:

- update hyper_ranges
- ensure that the correct tl is imported
- make sure to customize and import that customized base model def
- pass in # of epochs to train for
- pass in # of models to sample 


Run the script to get the train/test split:
```python
python3 tw01.py hyper_search_id --num_models 8 --epochs 100
````


---

