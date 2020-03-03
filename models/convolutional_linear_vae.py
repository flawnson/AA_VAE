import torch
import torch.nn as nn

from models.vae_template import VaeTemplate


def reparameterization(mu, log_var: torch.Tensor, device):
    std = log_var.mul(0.5).exp_()
    std = std.to(device)
    esp = torch.randn(*mu.size()).to(device)
    z = mu + std * esp
    return z


def out_size_conv(current_layer, padding, dilation, kernel_size, stride):
    return ((current_layer + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


def out_size_transpose(current_layer, padding, dilation, kernel_size, stride):
    return (current_layer - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + padding + 1


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
            nn.BatchNorm1d(out_c),
            nn.ELU(),
            nn.ConvTranspose1d(in_c, out_c, kernel_size=kernel_size, bias=False, padding=0, groups=1)
        )

    def forward(self, x):
        return self.conv_block(x)


class Encoder(nn.Module):
    def __init__(self, layers, kernel_size, h_dim, input_size, input_channels: int, scale_factor):
        super(Encoder).__init__()
        self.conv_layers = nn.ModuleList()
        output_channels = 2 ** ((input_channels - 1).bit_length())
        out_size = input_size
        for n in range(layers):
            block = ConvolutionalBlock(input_channels, output_channels, kernel_size)
            self.conv_layers.append(block)
            input_channels = output_channels
            output_channels = output_channels * scale_factor
            out_size = out_size_conv(out_size, 0, 1, kernel_size, 1)
        self.out_size = out_size
        self.out_channels = output_channels
        self.linear_layer = nn.Linear(out_size * output_channels, h_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        return x


class Decoder(nn.Module):
    def __init__(self, layers, kernel_size, output_expected, input_size, input_channels: int, output_channels_expected, scale_factor):
        super(Decoder).__init__()
        self.conv_layers = nn.ModuleList()
        output_channels = 2 ** ((input_channels - 1).bit_length())
        out_size = input_size
        for n in range(layers - 1):
            block = ConvolutionalTransposeBlock(input_channels, output_channels, kernel_size)
            self.conv_layers.append(block)
            input_channels = output_channels
            output_channels = output_channels / scale_factor
            out_size = out_size_transpose(out_size, 0, 1, kernel_size, 1)
        block = ConvolutionalTransposeBlock(input_channels, output_channels_expected, kernel_size)
        out_size = out_size_transpose(out_size, 0, 1, kernel_size, 1)
        self.conv_layers.append(block)
        self.out_size = out_size
        self.out_channels = output_channels
        self.linear_layer = nn.Linear(out_size * output_channels_expected, output_expected)

    def forward(self, x):
        x = self.conv_layers(x)
        return self.linear_layer(x)


class ConvolutionalVAE(VaeTemplate, nn.Module):
    def __init__(self, model_config, h_dim, z_dim, out_dim, device, embeddings_static):
        sizes: list = model_config["sizes"]
        input_size = model_config["input_size"]
        kernel_size = model_config["kernel_size"]
        encoder = Encoder(4, kernel_size, h_dim, input_size, embeddings_static.shape[1], 2)

        decoder = Decoder(4, kernel_size, input_size, z_dim, out_dim, )

        super(ConvolutionalVAE, self).__init__(encoder, decoder, device, h_dim, z_dim)

        embedding = nn.Embedding(embeddings_static.shape[0], embeddings_static.shape[1])
        embedding.weight.data.copy_(embeddings_static)
        self.embedding = embedding
        self.deembed = nn.Linear(sizes[0], sizes[0])

        self.smax = nn.Sigmoid()

    def forward(self, x):
        h = self.encoder(self.embedding(x).transpose(1, 2))
        z, mu, log_var = self.bottleneck(h)
        z = self.fc3(z)
        val = self.smax((self.decoder(z).transpose(1, 2)))
        if self.postprocess is not None:
            val = self.postprocess(val)
        return val.transpose(1, 2), mu, log_var
