import utils.train_utils as train_utils
import utils.ffcv_utils as ffcv_utils
from utils.dataset import CustomDataset
from utils.dataset import Mask
from torch.utils.data import DataLoader
import torch
import yaml
import pickle
import numpy as np

from tqdm import tqdm



# set cpu / gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# load configs
with open(r"configs/config.yaml") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)

model_name = cfg['model_name']
data_path = cfg['data_path']
mask_generator = Mask(data_path)

batch_size = cfg['batch_size']
num_workers = cfg['num_workers']
learning_rate = cfg['learning_rate']
num_epochs = cfg['num_epochs']
seed = cfg['seed']

test_split = cfg['test_split']
val_split = cfg['val_split']

n_repeats = cfg['n_repeats']
results_path = cfg['results_path']

ffcv = cfg['ffcv']

# set seeds
np.random.seed(seed)
torch.manual_seed(seed)

# load prediction tracker if continuing previous run
if 'pred_tracker' in cfg:
    with open(cfg['pred_tracker'], 'rb') as f:
        pred_tracker = pickle.load(f)
else:
    pred_tracker = {}

# write ffcv dataset if using ffcv
if ffcv:
    mask = None
    dataset = CustomDataset(data_path, mask=None, mode=0, ffcv=True)
    num_class = dataset.num_class
    write_path = './temp/dataset.beton'
    ffcv_utils.write_ffcv_dataset(dataset, write_path)
    
    train_ld = ffcv_utils.ffcv_loader(
        write_path,
        device,
        batch_size,
        num_workers,
        shuffle=True
    )
    val_ld = ffcv_utils.ffcv_loader(
        write_path,
        device,
        batch_size,
        num_workers,
        shuffle=True
    )
    
# monte carlo simulation
for i in range(n_repeats):
    print(f"Round {i}")  
    
    mask = mask_generator.generate_mask(test_split, val_split)
    test_dataset = CustomDataset(data_path, mask, mode=2, ffcv=False)
    test_ld = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if not ffcv:
        train_dataset = CustomDataset(data_path, mask, mode=0, ffcv=False)
        val_dataset = CustomDataset(data_path, mask, mode=1, ffcv=False)
        
        num_class = train_dataset.num_class
        
        train_ld = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_ld = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
    else:
        train_idx = np.where(mask==0)[0]
        val_idx = np.where(mask==1)[0]
        test_idx = np.where(mask==2)[0]
        
        
        train_ld.indices = train_idx
        train_ld.traversal_order.indices = train_idx
        val_ld.indices = val_idx
        val_ld.traversal_order.indices = val_idx
        
        

    model = train_utils.train_model(
        model_name, num_class, 
        device, 
        train_ld, val_ld, 
        learning_rate, num_epochs,
        ffcv
    )
    
    predictions = train_utils.get_predictions(
        model, device, test_ld
    )
    pred_tracker = train_utils.store_predictions(
        pred_tracker, test_dataset, predictions
    )
    
    with open('temp/temp_pred_tracker.pickle', 'wb') as f:
        pickle.dump(pred_tracker, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    

train_utils.write_results(results_path, pred_tracker)