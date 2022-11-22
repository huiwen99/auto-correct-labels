# auto-correct-labels
Automatically correct mislabelled image data for custom datasets. Based on this [article](https://medium.com/@yalcinmurat1986/auto-correcting-mislabeled-data-7a4098c77357). Improved with faster training via FFCV.  

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

Create a `temp` folder if it does not exist yet.  

### Step 2
Change the config file `configs/config.yaml` to your liking.  
Parameters:  
- `model_name`: (str) Name of model architecture. Currently, only supports `efficientnet` and `alexnet`. Else, a custom small CNN will be used.  
- `model_weights` (optional): (str) Path to torch pretrained weights. If not used, model will download pretrained weights from the internet.  
- `data_path`: (str) Path to the data in Step 1.
- `n_repeats`: (int) Number of simulations to run. 1 simulation trains 1 model.  
- `batch_size`: (int) Batch size for training, validation, and test  
- `learning_rate`: (float) Learning rate for model training.  
- `num_epochs`: (int) Number of epochs to train per simulation.  
- `track_eval`: (bool) Evaluate model against train and val set after every epoch and print the results. If True, each epoch will take longer but earlystopping will be performed.  
- `test_split`: (float between 0-1) Ratio of test set to total dataset in each simulation run.  
- `val_split`: (float between 0-1) Ratio of validation set to total dataset in each simulation run.  
- `ffcv`: (bool) Whether to use FFCV to speed up training or not.
- `results_path`: (str) Path to write csv results to
- `pred_tracker` (optional): (str) Path to pickle file of dictionary tracking the predictions. Used if previous has stopped midway. Pickle files are automatically saved to `temp/temp_pred_tracker.pickle`  
- `seed`(optional): (int) Random seed.  

### Step 3  
Run `python3 correct_labels.py` in docker container.  

Tip: Set `track_eval` to True to check that train/val accuracy is decent for the first few rounds before restarting with `track_eval`=False.  

Output will be saved as a csv file in the `results` folder, which includes a list of filenames, their original and corrected labels, the number of times the image has been predicted in total and the percentage of predictions that were of the corrected label. 

### Step 4
(optional)
If you would like to automatically relabel your data based on the results in Step 3, edit the `sensitivity` value in `autocorrect_config.yaml` to your liking. This will be the threshold value and any rows above this value will be relabelled.  


Run `python3 autocorrect.py` in docker container.

