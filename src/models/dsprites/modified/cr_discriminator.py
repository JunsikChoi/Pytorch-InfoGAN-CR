import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

'''
Vanila InfoGAN CR Discriminator Model Definition for dSprites dataset
'''


class CRDiscriminator(nn.Module):
    '''Shared Part of Discriminator and Recognition Model'''

    def __init__(self, dim_c_cont):
        super(CRDiscriminator, self).__init__()
        # self.dim_c_disc = dim_c_disc
        self.dim_c_cont = dim_c_cont
        # self.n_c_disc = n_c_disc
        # Shared layers
        self.module = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=2,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Reshape(-1, 64*4*4),
            spectral_norm(nn.Linear(in_features=64*4*4, out_features=128)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(in_features=128, out_features=self.dim_c_cont),
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        out = self.module(x)
        return out


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
