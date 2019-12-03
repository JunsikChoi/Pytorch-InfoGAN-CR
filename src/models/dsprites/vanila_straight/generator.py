import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Vanila InfoGAN Generator Model Definition for dSprites dataset
'''


class Generator(nn.Module):
    def __init__(self, dim_z, n_c_disc, dim_c_disc, dim_c_cont):
        super(Generator, self).__init__()
        self.dim_latent = dim_z + n_c_disc * dim_c_disc + dim_c_cont
        self.fc1 = nn.Linear(in_features=self.dim_latent,
                             out_features=128,
                             bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.fc2 = nn.Linear(in_features=128,
                             out_features=4*4*64,
                             bias=False)
        self.bn2 = nn.BatchNorm1d(4*4*64)
        self.upconv3 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=64,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv4 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=32,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.upconv5 = nn.ConvTranspose2d(in_channels=32,
                                          out_channels=32,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.upconv6 = nn.ConvTranspose2d(in_channels=32,
                                          out_channels=1,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1)

    def forward(self, z):
        # Layer 1: [-1, dim_latent] -> [-1, 128]
        z = F.relu(self.bn1(self.fc1(z)))

        # Layer 2: [-1, 1024] -> [-1, 4*4*64]
        z = F.relu(self.bn2(self.fc2(z)))

        # Shape Change: [-1, 4*4*64] -> [-1, 64, 4, 4]
        z = z.view(-1, 64, 4, 4)

        # Layer 3: [-1, 64, 4, 4] -> [-1, 64, 8, 8]
        z = F.relu(self.bn3(self.upconv3(z)))

        # Layer 4: [-1, 64, 8, 8] -> [-1, 32, 16, 16]
        z = F.relu(self.bn4(self.upconv4(z)))

        # Layer 5: [-1, 32, 16, 16] -> [-1, 32, 32, 32]
        z = F.relu(self.bn5(self.upconv5(z)))

        # Layer 6: [-1, 32, 32, 32] -> [-1, 1, 64, 64]
        img = torch.sigmoid(self.upconv6(z))

        return img
