import torch
import torch.nn as nn

import models.mish as mish


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padded=False):
        super().__init__()
        padding = 0
        if padded:
            padding = int((kernel_size - 1) / 2)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=kernel_size, bias=False, padding=padding, groups=1),
            nn.ELU(),
            nn.BatchNorm1d(out_c)
        )

    def forward(self, x):
        return self.conv_block(x)


class GCNContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='att',
                 fusion_types=('channel_mul',)):
        super().__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv1d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv1d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1]),
                nn.ELU(inplace=True),  # yapf: disable
                nn.Conv1d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv1d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1]),
                nn.ELU(inplace=True),  # yapf: disable
                nn.Conv1d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def spatial_pool(self, x):
        batch, channel, height = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class ConvolutionalTransposeBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padded=False):
        super().__init__()
        padding = 0
        if padded:
            padding = int((kernel_size - 1) / 2)
        self.conv_block = nn.Sequential(
            nn.ELU(inplace=True),
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


def out_size_transpose(current_layer, padding, dilation, kernel_size, stride, output_padding=0):
    return (current_layer - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
