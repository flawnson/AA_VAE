import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, model_config, h_dim, z_dim, input_size, device, embeddings_static, requires_grad=True):
        torch.manual_seed(0)
        self.name = "transformer_vae"
        # def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
        self.model_type = 'Transformer'
        self.src_mask = None
        nheads = model_config["heads"]
        layers = model_config["layers"]
        moving_dimension = model_config["internal_dimension"]
        feed_forward_dim = model_config["feed_forward"]
        self.pos_encoder = PositionalEncoding(moving_dimension)
        encoder_layers = TransformerEncoderLayer(d_model=moving_dimension, nhead=nheads,
                                                 dim_feedforward=feed_forward_dim)
        self.transformer_encoder = TransformerEncoder(encoder_layers, layers)
        self.encoder = nn.Embedding(embeddings_static.shape[0], embeddings_static.shape[1])
        self.encoder.weight.data.copy_(embeddings_static)
        self.encoder.weight.requires_grad = False

        self.embedder = nn.Conv1d(kernel_size=3, in_channels=embeddings_static.shape[1],
                                  out_channels=moving_dimension, stride=1, padding=1, bias=False)
        decoder_layer = TransformerDecoderLayer(moving_dimension, nhead=nheads, dim_feedforward=feed_forward_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, layers)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_encoding(self, x):
        return self.transformer_encoder(x)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src_mask = src.le(20)
        src = self.encoder(src)
        src = self.embedder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(output, src_mask)
        return output
