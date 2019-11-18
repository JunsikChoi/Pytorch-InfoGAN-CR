import torch
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets


def get_loader(batch_size, root):
    # Configure data loader
    # os.makedirs("../data", exist_ok=True)
    # transforms.Normalize([0.5], [0.5])
    data_dir = os.path.join(root, 'data/mnist')
    os.makedirs(data_dir, exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor()]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return dataloader
