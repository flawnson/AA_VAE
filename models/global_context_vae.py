import copy

import torch.nn.modules.activation

from models.model_common import *


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


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
    r"""TransformerEncoderLayer is made up of channel self-attn using Squeeze expand network along with
    a spatial Non linear SAGAN using a gaussian  network.
    This encoder is based on the GCNet paper.

    Args:
        dropout: the dropout value (default=0.1).
        kernel_size: size of the kernel

    """

    def __init__(self, channels, dropout=0.1, kernel_size=3):
        super(TransformerEncoderLayer, self).__init__()
        self.spatial_attention = GCNContextBlock(inplanes=channels, ratio=8)
        self.dropout1 = nn.Dropout(dropout)
        out_c = int(channels / 2)
        self.channel_attention = nn.Sequential(
            ConvolutionalBlock(in_c=channels, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalBlock(in_c=out_c, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalBlock(in_c=out_c, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalBlock(in_c=out_c, out_c=channels, padded=True, kernel_size=kernel_size),
        )

    def forward(self, src):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
        Shape:
            see the docs in Transformer class.
        """
        residual = src
        src = src + self.dropout1(self.spatial_attention(src))
        src = self.channel_attention(src) + residual
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
        self.spatial_attention = GCNContextBlock(channels, 8)
        self.dropout1 = nn.Dropout(dropout)
        # out_c = channels
        out_c = int(channels / 2)
        self.channel_attention = nn.Sequential(
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
        src = self.channel_attention(src) + residual
        src2 = self.spatial_attention(src)
        src = src + self.dropout1(src2)
        return src


class GlobalContextVAEModel(nn.Module):
    def __init__(self, model_config, h_dim, z_dim, input_size, device, embeddings_static, requires_grad=True):
        torch.manual_seed(0)
        self.device = device

        self.name = "global_context_vae"
        super(GlobalContextVAEModel, self).__init__()
        self.model_type = 'GCA_vae'
        self.src_mask = None
        layers = model_config["layers"]
        self.channels = model_config["channels"]
        kernel_dimension = model_config["kernel_size"]
        self.triple_encoder = nn.Conv1d(kernel_size=3, in_channels=embeddings_static.shape[1],
                                        out_channels=self.channels, stride=1, padding=1, bias=False)
        self.deembed = nn.ConvTranspose1d(kernel_size=3, in_channels=self.channels,
                                          out_channels=embeddings_static.shape[0], padding=1, bias=False)
        self.protein_embedding = nn.Embedding(embeddings_static.shape[0], embeddings_static.shape[1])
        self.protein_embedding.weight.data.copy_(embeddings_static)
        self.protein_embedding.weight.requires_grad = False

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
        self.activation = nn.LogSoftmax(dim=1)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.triple_encoder.weight.data.uniform_(-initrange, initrange)
        self.deembed.weight.data.uniform_(-initrange, initrange)

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = (self.fc2(h))
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def forward(self, x):
        input_len = x.shape[1]
        mask = x.le(20).unsqueeze(1).float()
        z, mu, log_var = self.bottleneck(
            self.transformer_encoder(self.triple_encoder(self.protein_embedding(x).transpose(1, 2)))
                .view(x.shape[0], -1))
        data = self.fc3(z).view(z.shape[0], -1, input_len)
        data = self.resize_channels(torch.cat((data, mask), 1))
        return self.activation(self.deembed(self.transformer_decoder(data))), mu, log_var

    def representation(self, x):
        x = self.transformer_encoder(self.triple_encoder(self.protein_embedding(x).transpose(1, 2))).view(x.shape[0],
                                                                                                          -1)
        return self.bottleneck(x)
