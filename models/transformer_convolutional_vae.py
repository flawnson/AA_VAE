import copy

import torch.nn.modules.activation

from models.model_common import *


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv1d(**kwargs), eps=eps)


def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


def sn_embedding(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(**kwargs), eps=eps)


class SelfAttnHuggingFace(nn.Module):

    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttnHuggingFace, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels // 8,
                                        kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels // 8,
                                      kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels // 2,
                                    kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_o_conv = snconv2d(in_channels=in_channels // 2, out_channels=in_channels,
                                         kernel_size=1, bias=False, eps=eps)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, ch, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(b, ch // 8, w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(b, ch // 8, w // 4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(b, ch // 2, w // 4)
        # Attn_g - o_conv
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(b, ch // 2, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        # Out
        out = x + self.gamma * attn_g
        return out


class TransformerLayer(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerLayer, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    """

    def __init__(self, channels, dropout=0.1, kernel_size=3):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SelfAttnHuggingFace(channels)
        self.dropout1 = nn.Dropout(dropout)
        out_c = int(channels / 2)
        self.mutate = nn.Sequential(
            ConvolutionalBlock(in_c=channels, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalBlock(in_c=out_c, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalBlock(in_c=out_c, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalBlock(in_c=out_c, out_c=channels, padded=True, kernel_size=kernel_size),
        )

    def forward(self, src):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        residual = src
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.mutate(src) + residual
        return src


class TransformerDecoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        channels: the number of channels in the convolution.
        dropout: the dropout value (default=0.1).
        kernel_size: The size of the kernel of the convolutions.

    """

    def __init__(self, channels, dropout=0.1, kernel_size=1):
        """

        :param channels:
        :param dropout:
        :param kernel_size:
        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = SelfAttnHuggingFace(channels)
        self.dropout1 = nn.Dropout(dropout)
        # out_c = channels
        out_c = int(channels / 2)
        self.mutate = nn.Sequential(
            ConvolutionalTransposeBlock(in_c=channels, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalTransposeBlock(in_c=out_c, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalTransposeBlock(in_c=out_c, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalTransposeBlock(in_c=out_c, out_c=channels, padded=True, kernel_size=kernel_size)
        )

    def forward(self, src):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).

        Shape:
            see the docs in Transformer class.
        """
        residual = src
        src = self.mutate(src) + residual
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        return src


class TransformerConvVAEModel(nn.Module):
    def __init__(self, model_config, h_dim, z_dim, input_size, device, embeddings_static, requires_grad=True):
        torch.manual_seed(0)
        self.device = device

        self.name = "transformer_conv_vae"
        super(TransformerConvVAEModel, self).__init__()
        self.model_type = 'Transformer_Convolutional'
        self.src_mask = None
        layers = model_config["layers"]
        self.channels = model_config["channels"]
        kernel_dimension = model_config["kernel_size"]
        self.embedder = nn.Conv1d(kernel_size=3, in_channels=embeddings_static.shape[1],
                                  out_channels=self.channels, stride=1, padding=1, bias=False)
        self.deembed = nn.ConvTranspose1d(kernel_size=3, in_channels=self.channels,
                                          out_channels=embeddings_static.shape[0], padding=1, bias=False)
        self.encoder = nn.Embedding(embeddings_static.shape[0], embeddings_static.shape[1])
        self.encoder.weight.data.copy_(embeddings_static)
        self.encoder.weight.requires_grad = False

        encoder_layers = TransformerEncoderLayer(channels=self.channels, kernel_size=kernel_dimension)
        self.transformer_encoder = TransformerLayer(encoder_layers, layers)
        self.resize_channels = ConvolutionalTransposeBlock(in_c=self.channels + 1, out_c=self.channels, padded=True,
                                                           kernel_size=1)

        decoder_layer = TransformerDecoderLayer(channels=self.channels, kernel_size=kernel_dimension)
        self.transformer_decoder = TransformerLayer(decoder_layer, layers)
        h_dim = input_size * self.channels
        self.fc1: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc2: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc3: nn.Module = nn.Linear(z_dim, h_dim)
        self.activation = nn.LogSoftmax(dim=0)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedder.weight.data.uniform_(-initrange, initrange)
        self.deembed.weight.data.uniform_(-initrange, initrange)

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = self.fc2(h)
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def forward(self, x):
        input_len = x.shape[1]
        mask = x.le(20).unsqueeze(1).float()
        z, mu, log_var = self.bottleneck(
            self.transformer_encoder(self.embedder(self.encoder(x).transpose(1, 2))).view(x.shape[0], -1))
        # z, mu, log_var = self.bottleneck(output)
        data = self.fc3(z).view(z.shape[0], -1, input_len)
        data = self.resize_channels(torch.cat((data, mask), 1))
        return self.activation(self.deembed(self.transformer_decoder(data))), mu, log_var

    def representation(self, x):
        x = self.transformer_encoder(self.embedder(self.encoder(x).transpose(1, 2))).view(x.shape[0], -1)
        return self.bottleneck(x)[1]
