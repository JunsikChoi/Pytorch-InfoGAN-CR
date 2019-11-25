import torch
import os
import sys
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from urllib import request


def get_loader(batch_size, root, dataset):

    data_dir = os.path.join(root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    if dataset == 'mnist':
        save_dir = f'{data_dir}/mnist'
        os.makedirs(save_dir, exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                f'{data_dir}/mnist',
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor()]
                ),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    elif dataset == 'dsprites':
        save_dir = f'{data_dir}/dsprites'
        os.makedirs(save_dir, exist_ok=True)
        dset = DSpriteDataset(save_dir)
        dataloader = torch.utils.data.DataLoader(
            dset, batch_size=batch_size, shuffle=True)

    return dataloader


class DSpriteDataset(Dataset):
    """
    A PyTorch wrapper for the dSprites dataset by
    Matthey et al. 2017. The dataset provides a 2D scene
    with a sprite under different transformations:
    * color
    * shape
    * scale
    * orientation
    * x-position
    * y-position
    """

    def __init__(self, save_dir, transform=None):
        self.transform = transform
        self.file_loc = f'{save_dir}/dsprites.npz'
        url = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        try:
            dataset_zip = np.load(
                self.file_loc, encoding='bytes', allow_pickle=True)
        except FileNotFoundError:
            print("Dsprite Dataset Not Found, Downloading...")
            request.urlretrieve(url, self.file_loc)
            dataset_zip = np.load(
                self.file_loc, encoding='bytes', allow_pickle=True)
        print("Dsprites Dataset Loaded")
        self.imgs = dataset_zip['imgs']
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = np.expand_dims(self.imgs[idx], axis=0).astype(np.float32)
        label = self.latents_values[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label
