import torch


def reparameterization(mu, log_var: torch.Tensor, device):
    std = log_var.mul(0.5).exp_().to(device)
    esp = torch.rand_like(std).to(device)
    return mu + std * esp


def out_size_conv(current_layer, padding, dilation, kernel_size, stride):
    return ((current_layer + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1


def out_size_transpose(current_layer, padding, dilation, kernel_size, stride):
    return (current_layer - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + padding + 1
