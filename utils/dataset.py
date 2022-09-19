import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pathlib import Path
import os
from tqdm import tqdm
import collections


class Mask():
    def __init__(self, data_path):
        self.total_len = CustomDataset.get_total_data_size(
            Path(data_path)
        )
        
    def generate_mask(self, test_split, val_split):
        """
        Generates a mask to split the dataset into train, val, test
        e.g. test_split = 0.1, val_split = 0.2
        then test 10%, val 20%, train 70%
        """
        mask = np.zeros(self.total_len)
        test_idx = int(self.total_len * test_split)
        mask[:test_idx] = 2
        val_idx = int(self.total_len * val_split) + test_idx
        mask[test_idx:val_idx] = 1
        mask[val_idx:] = 0
        np.random.shuffle(mask)
        return mask
        

class CustomDataset(Dataset):
    """
    Custom dataset class.
    """
    def __init__(self, data_path, mask=None, mode=0, img_size=(224,224), ffcv=True):
        """
        Constructor for dataset class. 
        Uses the mask to return only the desired images 
        (corresponding to mode)

        Parameters
        ----------
        data_path: str or Path
            path to data directory
        mask: list or array of integers
            e.g. [0 1 0 0 1 2]
            length has to correspond to total number of data samples in data_path
        mode: int
            i.e. 0 for train, 1 for val, 2 for test
        """
        self.img_size = img_size
        self.data_path = Path(data_path)
        self.classes = os.listdir(str(data_path))
        self.num_class = len(self.classes)
        self.mode = mode
        self.ffcv = ffcv

        self.class_map = {k: v for v, k in enumerate(self.classes)}
        self.reverse_class_map = {v: k for v, k in enumerate(self.classes)}
        
        self.x, self.y = self.get_x_and_y(self.data_path, mask)

    @staticmethod
    def get_all_x_and_y(data_path):
        """
        Returns a list of x (image filenames) and y (labels)
        """
        img_exts = [
            ".jpg",
            ".jpeg",
            ".png"
        ]

        x = []
        y = []

        for ext in img_exts:
            for elem in tqdm(data_path.rglob("*" + ext)):
                x.append(elem)
                y.append(elem.parent.stem)
        return x, y

    
    def get_x_and_y(self, data_path, mask):
        """
        Returns a list of filtered x (image filenames) and y (labels)
        """
        x, y = CustomDataset.get_all_x_and_y(data_path)
        
        if mask is not None:
            x, y = self.do_mask(x, y, mask, self.mode)

        cnt = collections.Counter(y)
        print(cnt)

        return x, y
    

    @staticmethod
    def do_mask(x, y, mask, mode):
        """
        Returns only the desired x and y based on the mask
            eg. x = ['a', 'b', 'c']
                mask = [1, 0, 1]
                mode = 1
            returns ['a', 'c']
        """

        mask = np.array(mask)
        idxs = np.where(mask==mode)[0]
        
        masked_x = [x[i] for i in idxs]
        masked_y = [y[i] for i in idxs]
        return masked_x, masked_y


    def __len__(self):
        return len(self.x)
    

    def get_transform(self):
        """
        Returns transform function based on whether the dataset is train or test
        """
        mean = [x/255 for x in [125.30691805, 122.95039414, 113.86538318]]
        std = [x/255 for x in [62.99321928, 62.08870764, 66.70489964]]
    
        
        if not self.ffcv:
            transform = A.Compose(
                [
                    A.Resize(self.img_size[0],self.img_size[1]),
                    A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
                    ToTensorV2()
                ]
            )
        else:
            transform = A.Compose(
                [
                    A.Resize(self.img_size[0],self.img_size[1]),
                    # A.Normalize(mean=mean, std=std, max_pixel_value=1.0)
                ]
            )
        
        return transform

        
    def int_to_one_hot(self, x):
        """Converts integer to one hot tensor"""
        one_hot = np.zeros(self.num_class)
        one_hot[x] = 1
        return torch.Tensor(one_hot)


    def __getitem__(self, index):
        """
        Returns the image and its label as one hot vector
        """
        filename = self.x[index]
        label = self.y[index]
        label = self.class_map[label]
        # label = self.int_to_one_hot(label) 

        img = cv2.imread(str(filename))
        transform = self.get_transform()
        img = transform(image=img)
        img = img['image']
        if self.ffcv:
            img = img.astype(np.uint8)
        
        return img, label

    @staticmethod
    def get_total_data_size(data_path):
        x, _ = CustomDataset.get_all_x_and_y(data_path)
        return len(x)
        