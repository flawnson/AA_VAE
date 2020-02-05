import torch.nn as nn

from models.vae_template import VaeTemplate


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, size=1024):
        self.size = size
        super().__init__()

    def forward(self, input):
        return input.view(input.size(0), self.size, 1, 1)


class ConvolutionalVAE(VaeTemplate):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        encoder = nn.Sequential(
            nn.Conv1d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        decoder = nn.Sequential(
            UnFlatten(size=h_dim),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        super(ConvolutionalVAE, self).__init__(encoder, decoder, h_dim, z_dim)
