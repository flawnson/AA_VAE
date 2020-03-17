import torch
import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padded=False):
        super().__init__()
        padding = 0
        if padded:
            padding = (kernel_size - 1) / 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, bias=False, padding=padding, groups=1),
            nn.ELU(),
            nn.BatchNorm1d(out_c)
        )

    def forward(self, x):
        return self.conv_block(x)


class ConvolutionalTransposeBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padded=False):
        super().__init__()
        padding = 0
        if padded:
            padding = (kernel_size - 1) / 2
        self.conv_block = nn.Sequential(
            nn.ELU(),
            nn.BatchNorm1d(in_c),
            nn.ConvTranspose1d(in_c, out_c, kernel_size=kernel_size, bias=False, padding=padding, groups=1)
        )

    def forward(self, x):
        return self.conv_block(x)


def reparameterization(mu, log_var: torch.Tensor, device):
    std = log_var.mul(0.5).exp_().to(device)
    esp = torch.rand_like(std).to(device)
    return mu + std * esp


def out_size_conv(current_layer, padding, dilation, kernel_size, stride):
    return ((current_layer + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


def out_size_transpose(current_layer, padding, dilation, kernel_size, stride):
    return (current_layer - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + padding + 1
