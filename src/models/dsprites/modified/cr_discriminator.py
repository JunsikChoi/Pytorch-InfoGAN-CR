import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Discriminator Model Definition
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
            nn.Conv2d(in_channels=2,
                      out_channels=64,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Reshape(-1, 128*7*7),
            nn.Linear(in_features=128*7*7, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(in_features=1024, out_features=self.dim_c_cont),
            # nn.Softmax(dim=1)
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
