import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Generator Model Definition
'''


class Generator(nn.Module):
    def __init__(self, dim_z, n_c_disc, dim_c_disc, dim_c_cont):
        super(Generator, self).__init__()
        self.dim_latent = dim_z + n_c_disc * dim_c_disc + dim_c_cont
        self.fc1 = nn.Linear(in_features=self.dim_latent,
                             out_features=1024,
                             bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=1024)
        self.fc2 = nn.Linear(in_features=1024,
                             out_features=7*7*128,
                             bias=False)
        self.bn2 = nn.BatchNorm1d(7*7*128)
        self.upconv3 = nn.ConvTranspose2d(in_channels=128,
                                          out_channels=64,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv4 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=1,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)

    def forward(self, z):
        # Layer 1: [-1, dim_latent] -> [-1, 1024]
        z = F.relu(self.bn1(self.fc1(z)))

        # Layer 2: [-1, 1024] -> [-1, 7*7*128]
        z = F.relu(self.bn2(self.fc2(z)))

        # Shape Change: [-1, 7*7*128] -> [-1, 128, 7, 7]
        z = z.view(-1, 128, 7, 7)

        # Layer 3: [-1, 128, 7, 7] -> [-1, 64, 14, 14]
        z = F.relu(self.bn3(self.upconv3(z)))

        # Layer 4: [-1, 64, 14, 14] -> [-1, 1, 28, 28]
        img = torch.sigmoid(self.upconv4(z))

        return img
