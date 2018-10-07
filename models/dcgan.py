"""
PyTorch DCGAN model code.
"""

# Externals
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator module for the GAN.
    """

    def __init__(self, noise_dim, output_channels=1, n_filters=16, threshold=0):
        super(Generator, self).__init__()
        # Number of filters in final generator layer
        ngf = n_filters
        # Construct the model as a sequence of layers
        self.network = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            nn.Threshold(threshold, 0)
            # state size. (nc) x 64 x 64
        )

    def forward(self, inputs):
        """Computes the forward pass of the generator network"""
        return self.network(inputs)

class Discriminator(nn.Module):
    """
    Discriminator module for the GAN.
    """

    def __init__(self, input_channels=1, n_filters=16):
        super(Discriminator, self).__init__()
        # Number of initial filters of discriminator network
        ndf = n_filters
        self.network = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        return self.network(inputs).squeeze()

def get_gan(n_channels=1, n_filters=16, noise_dim=64, threshold=0):
    g = Generator(noise_dim=noise_dim, output_channels=n_channels,
                  n_filters=n_filters, threshold=threshold)
    d = Discriminator(input_channels=n_channels, n_filters=n_filters)
    return g, d
