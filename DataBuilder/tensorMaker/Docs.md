# tensorMaker Dataset  Templates

## Description for 0 Tensor Builder (0_tensor_builder.py )

This python script will need to be customized for each uscase.
The genrla idea is tpo abstract away the dataset creation so in the training loop you can just do the following:

```python
from 0_tensor_builder.py import getDataSet
train_dataset, test_dataset = getDataSet()
````
### Key Components

#### Configuration

Adjust these variables according to your specific dataset:

```python
raw_data_dir = '/home/kuba/Documents/Data/Raw/RatEEG_R/' #THIS NEEDS TO BE ABSOLUTE PATH
```
### Usage Instructions

Copy this scrpit to your project directory.
Modify the configuration variables to match your dataset:

- Update raw_data_path to the absolute path of your data dir
- Adjust the way the dataset creation/initalizationis done


Run the script to get the train/test split:
```python
from 0_tensor_builder.py import getDataSet
train_dataset, test_dataset = getDataSet()
#also valid is 
train_dataset, test_dataset = getDataSet(randomState=59, trainPercent=0.8)
````


### Notes

- Default for `randomState=69`
- Default for `trainPercent=0.7`

---

