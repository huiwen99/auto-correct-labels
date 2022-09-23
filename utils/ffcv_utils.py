from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, NDArrayField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage, Convert
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder, NDArrayDecoder
import numpy as np
import torch
import torchvision.transforms as T
import gc

from utils.dataset import CustomDataset
import albumentations as A

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
    writer = DatasetWriter(write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': NDArrayField(dtype=np.dtype(np.float32), shape=(224,224,3)),
        'label': IntField()
    })
    # Write dataset
    writer.from_indexed_dataset(dataset)
    
def ffcv_loader(write_path, device, batch_size, num_workers, shuffle=True):
    """Get FFCV dataloader from .beton file"""
    mean = [125.30691805, 122.95039414, 113.86538318]
    std = [62.99321928, 62.08870764, 66.70489964]
    
    
    image_pipeline = [
        NDArrayDecoder(),
        ToTensor(),
        ToTorchImage(convert_back_int16=False),
        ToDevice(device)
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
                order=order, pipelines=pipelines, drop_last=False)

    return loader


def gc_get_objects():
    gc.get_objects()


def check_tensors():
    num_parameters = 0
    num_tensors = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if type(obj).__name__ == 'Parameter':
                    num_parameters += 1
                elif type(obj).__name__ == 'Tensor':
                    # print(f"{type(obj).__name__}, {obj.device}, {obj.dtype}, {obj.size()}")
                    num_tensors += 1
                else:
                    print(f"{type(obj).__name__}, {type(obj)}, {obj.device} {obj.dtype}, {obj.size()}")
        except:
            pass
    print(f'num_parameters: {num_parameters}')
    print(f'num_tensors: {num_tensors}')