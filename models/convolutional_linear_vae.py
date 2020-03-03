import torch.nn as nn

from utils.model_common import *


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, bias=False, padding=0, groups=1),
            nn.BatchNorm1d(out_c),
            nn.ELU())

    def forward(self, x):
        return self.conv_block(x)


class ConvolutionalTransposeBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(in_c),
            nn.ELU(),
            nn.ConvTranspose1d(in_c, out_c, kernel_size=kernel_size, bias=False, padding=0, groups=1)
        )

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, layers, kernel_size, input_size, input_channels: int, scale_factor):
        super().__init__()

        conv_layers = []

        output_channels = 2 ** ((input_channels - 1).bit_length())
        out_size = input_size

        for n in range(layers):
            block = ConvolutionalBlock(input_channels, output_channels, kernel_size)
            conv_layers.append(block)

            input_channels = output_channels
            output_channels = int(output_channels * scale_factor)
            out_size = out_size_conv(out_size, 0, 1, kernel_size, 1)

        self.out_size = out_size
        self.out_channels = int(input_channels)

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        return x


class Decoder(nn.Module):
    def __init__(self, layers, kernel_size, output_expected, input_size, input_channels: int, output_channels_expected,
                 scale_factor):
        super().__init__()
        conv_layers = []

        output_channels = int((2 ** ((input_channels - 1).bit_length()))/2)
        out_size = input_size

        for n in range(layers - 1):
            block = ConvolutionalTransposeBlock(input_channels, output_channels, kernel_size)
            conv_layers.append(block)

            input_channels = output_channels
            output_channels = int(output_channels / scale_factor)
            out_size = out_size_transpose(out_size, 0, 1, kernel_size, 1)

        block = ConvolutionalTransposeBlock(input_channels, output_channels_expected, kernel_size)
        out_size = out_size_transpose(out_size, 0, 1, kernel_size, 1)
        conv_layers.append(block)

        self.conv_layers = nn.Sequential(*conv_layers)

        self.out_size = out_size
        assert out_size == output_expected

    def forward(self, x):
        return self.conv_layers(x)


class ConvolutionalBaseVAE(nn.Module):
    def __init__(self, model_config, h_dim, z_dim, input_size, device, embeddings_static):
        super(ConvolutionalBaseVAE, self).__init__()
        self.name = "convolutional_basic"
        kernel_size = model_config["kernel_size"]
        layers = model_config["layers"]
        scale = model_config["scale"]

        self.device = device
        self.encoder = Encoder(layers, kernel_size, input_size, embeddings_static.shape[1], scale)
        h_dim = int(self.encoder.out_size * self.encoder.out_channels)
        self.decoder = Decoder(layers, kernel_size, input_size, self.encoder.out_size, self.encoder.out_channels,
                               embeddings_static.shape[0], scale)

        self.fc1: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc2: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc3: nn.Module = nn.Linear(z_dim, h_dim)
        embedding = nn.Embedding(embeddings_static.shape[0], embeddings_static.shape[1])
        embedding.weight.data.copy_(embeddings_static)

        self.embedding = embedding
        self.smax = nn.Sigmoid()

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = self.fc2(h)
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(self.embedding(x).transpose(1, 2))
        z, mu, log_var = self.bottleneck(h)
        z = self.fc3(z)
        z = z.view(z.shape[0], self.encoder.out_channels, -1)
        val = self.smax((self.decoder(z).transpose(1, 2)))
        return val.transpose(1, 2), mu, log_var
