# auto-correct-labels
Automatically correct mislabelled data

## Instructions to run
### Step 1
Put your data in the `data` folder.  
Structure should be as follows:  
```
.
├── data
│   ├── class1
│       ├── img1.png
│       ├── img2.png
|       └── ...
│   ├── class2 
│       ├── img3.png
│       ├── img4.png
|       └── ...
|   └── ...
└── ...
```

### Step 2
Change the config file `configs/config.yaml` to your liking.  
Parameters:  
- `model_name`: Name of model architecture. Currently, only supports `efficientnet` and `alexnet`. Else, a custom small CNN will be used.  
- `data_path`: Path to the data in Step 1.
- `n_repeats`: Number of simulations to run. 1 simulation trains 1 model.  
- `batch_size`: Batch size for training, validation, and test  
- `learning_rate`: Learning rate for model training.  
- `num_epochs`: Number of epochs to train per simulation.  
- `test_split`: Ratio of test set to total dataset in each simulation run.  
- `val_split`: Ratio of validation set to total dataset in each simulation run.  
- `results_path`: Path to write csv results to
- `pred_tracker` (optional): Path to pickle file of dictionary tracking the predictions. Used if previous has stopped midway. Pickle files are automatically saved to `temp/temp_pred_tracker.pickle`  

### Step 3  
Run `python3 correct_labels.py` in docker container.  