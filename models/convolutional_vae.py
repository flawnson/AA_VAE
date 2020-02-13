import torch.nn as nn

from models.vae_template import VaeTemplate


def out_size_conv(current_layer, padding, dilation, kernel_size, stride):
    return ((current_layer + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


def out_size_transpose(current_layer, padding, dilation, kernel_size, stride):
    return (current_layer - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + padding + 1


class ConvolutionalVAE(VaeTemplate):
    def __init__(self, model_config, h_dim, z_dim, out_dim, device):
        sizes: list = model_config["sizes"]
        kernel_sizes: list = model_config["kernel_sizes"]
        padding_sizes: list = model_config["kernel_sizes"]
        stride_sizes: list = model_config["kernel_sizes"]
        encoder = nn.Sequential(
            nn.Conv1d(in_channels=sizes[0], out_channels=sizes[1], kernel_size=kernel_sizes[0], stride=stride_sizes[0],
                      padding=padding_sizes[0], groups=1),
            nn.ReLU(sizes[1]),
            nn.Conv1d(in_channels=sizes[1], out_channels=sizes[2], kernel_size=kernel_sizes[1], stride=stride_sizes[1],
                      padding=padding_sizes[1], groups=1),
            nn.ReLU(sizes[2]),
            nn.Conv1d(in_channels=sizes[2], out_channels=sizes[3], kernel_size=kernel_sizes[2], stride=stride_sizes[2],
                      padding=padding_sizes[2], groups=1),
            nn.ReLU(sizes[3]),
            nn.Conv1d(in_channels=sizes[3], out_channels=1, kernel_size=kernel_sizes[3], stride=stride_sizes[3],
                      padding=padding_sizes[3], groups=1),
            nn.ReLU()  # ,
        )

        decoder = nn.Sequential(
            nn.ConvTranspose1d(1, sizes[3], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(sizes[3], sizes[2], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(sizes[2], sizes[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(sizes[1], out_dim, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )
        super(ConvolutionalVAE, self).__init__(encoder, decoder, device, h_dim, z_dim)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        z = self.fc3(z)
        val = self.decoder(z)
        if self.postprocess is not None:
            val = self.postprocess(val)
        return val, mu, log_var
