from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, NDArrayField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage, Convert
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, NDArrayDecoder
import numpy as np
import torch
import torchvision.transforms as T

from utils.dataset import CustomDataset
import albumentations as A

def get_datasets(data_path, mask):
    train_dataset = CustomDataset(data_path, mask, mode=0)
    val_dataset = CustomDataset(data_path, mask, mode=1)
    test_dataset = CustomDataset(data_path, mask, mode=2, ffcv=False)
    return train_dataset, val_dataset, test_dataset

def write_ffcv_dataset(dataset, write_path):
    """
    Converts your dataset into ffcv format
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        (image, label) pairs
    write_path: str
        Path to write ffcv dataset. '.beton' extension
    """
    # Pass a type for each data field
    # writer = DatasetWriter(write_path, {
    #     # Tune options to optimize dataset size, throughput at train-time
    #     'image': NDArrayField(
    #         shape=(len(dataset), dataset.img_size[0], dataset.img_size[1]), 
    #         dtype=np.dtype('float32')
    #     ),
    #     'label': IntField()})
    writer = DatasetWriter(write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(max_resolution=256),
        'label': IntField()
    })
    # Write dataset
    writer.from_indexed_dataset(dataset)
    

def ffcv_loader(write_path, device, batch_size, num_workers, shuffle=True):
    mean = [x/255 for x in [125.30691805, 122.95039414, 113.86538318]]
    std = [x/255 for x in [62.99321928, 62.08870764, 66.70489964]]
    
    image_pipeline = [
        SimpleRGBImageDecoder(),
        ToTensor(),
        ToDevice(device),
        ToTorchImage(),
        Convert(torch.float32),
        T.Normalize(mean, std)
    ]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device)]
    
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }
    if shuffle:
        order = OrderOption.RANDOM
    else:
        order = OrderOption.SEQUENTIAL
    
    loader = Loader(write_path, batch_size=batch_size, num_workers=num_workers,
                order=order, pipelines=pipelines)

    return loader

