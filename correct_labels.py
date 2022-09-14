import utils.train_utils as train_utils
from utils.dataset import CustomDataset
from utils.dataset import Mask
from torch.utils.data import DataLoader
import torch
import yaml
import pickle

# set cpu / gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

with open(r"configs/config.yaml") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

model_name = cfg['model_name']
data_path = cfg['data_path']
mask_generator = Mask(data_path)

batch_size = cfg['batch_size']
learning_rate = cfg['learning_rate']
num_epochs = cfg['num_epochs']

test_split = cfg['test_split']
val_split = cfg['val_split']

n_repeats = cfg['n_repeats']
results_path = cfg['results_path']

if 'pred_tracker' in cfg:
    with open(cfg['pred_tracker'], 'rb') as f:
        pred_tracker = pickle.load(f)
else:
    pred_tracker = {}
    
for i in range(n_repeats):
    print(f"Round {i}")
    mask = mask_generator.generate_mask(test_split, val_split)

    train_dataset = CustomDataset(data_path, mask, mode=0)
    val_dataset = CustomDataset(data_path, mask, mode=1)
    num_class = train_dataset.num_class

    train_ld = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = train_utils.train_model(
        model_name, num_class, 
        device, 
        train_ld, val_ld, 
        learning_rate, num_epochs
    )

    test_dataset = CustomDataset(data_path, mask, mode=2)
    test_ld = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    predictions = train_utils.get_predictions(
        model, device, test_ld
    )
    pred_tracker = train_utils.store_predictions(
        pred_tracker, test_dataset, predictions
    )
    
    with open('temp/temp_pred_tracker.pickle', 'wb') as f:
        pickle.dump(pred_tracker, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    

train_utils.write_results(results_path, pred_tracker)