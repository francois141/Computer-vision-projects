import numpy as np

import os

import torch
from torch.utils.data import Dataset


class Simple2DDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter using np.load.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.

        data_train = np.load('data/train.npz')
        self.samples_train = data_train['samples']
        self.annotations_train = data_train['annotations']
            
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples_train.shape[0]
    
    def __getitem__(self, idx):
        sample = self.samples_train[idx]
        annotation = self.annotations_train[idx]
        
        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


class Simple2DTransformDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.

        data_train = np.load('data/train.npz')
        self.samples_train = data_train['samples']
        self.annotations_train = data_train['annotations']

    
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples_train.shape[0]
    
    def __getitem__(self, idx):
        sample = transform(self.samples_train[idx])
        annotation = self.annotations_train[idx]

        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }



def transform(sample):
    value = np.sqrt(np.power(sample[0],2) + np.power(sample[1],2))
    return np.array([value,0])
