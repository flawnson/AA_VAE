import torch.nn as nn

from models.model_common import ConvolutionalBlock, ConvolutionalTransposeBlock, out_size_conv, out_size_transpose, \
    reparameterization


class Encoder(nn.Module):
    def __init__(self, layers, kernel_size, input_size, input_channels: int):
        super().__init__()
        conv_layers = []
        output_channels = 2 ** ((input_channels - 1).bit_length())
        out_size = input_size
        base_kernel_size = kernel_size
        padded = True
        padding = int((base_kernel_size - 1) / 2)

        for n in range(layers):
            block = ConvolutionalBlock(input_channels, output_channels, base_kernel_size, padded=padded)
            conv_layers.append(block)
            out_size = int(out_size_conv(out_size, padding, 1, base_kernel_size, 1))

        self.out_size = int(out_size)
        self.out_channels = int(input_channels)
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        return x


class Decoder(nn.Module):
    def __init__(self, layers, kernel_size, input_size, input_channels: int, output_channels_expected,
                 channel_scale_factor, max_channels=256, kernel_expansion_factor=1):
        super().__init__()
        conv_layers = []

        output_channels = input_channels
        out_size = input_size
        base_kernel_size = kernel_size
        padded = True
        padding = 0

        for n in range(layers - 1):
            block = ConvolutionalTransposeBlock(input_channels, output_channels, base_kernel_size, padded=padded)
            conv_layers.append(block)
            kernel_size = int(kernel_size / kernel_expansion_factor)
            input_channels = output_channels
            output_channels = int(output_channels / channel_scale_factor)
            if output_channels > max_channels:
                output_channels = max_channels
            if padded:
                padding = int((base_kernel_size - 1) / 2)

            out_size = out_size_transpose(out_size, padding, 1, base_kernel_size, 1)
            if kernel_expansion_factor > 1:
                base_kernel_size = kernel_size + 1
        if padded:
            padding = int((base_kernel_size - 1) / 2)
        block = ConvolutionalTransposeBlock(input_channels, output_channels_expected, base_kernel_size, padded=padded)
        out_size = out_size_transpose(out_size, padding, 1, base_kernel_size, 1)
        conv_layers.append(block)

        self.conv_layers = nn.Sequential(*conv_layers)

        self.out_size = out_size
        self.residue = 200

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class LSTMConvolutionalVae(nn.Module):
    def __init__(self, model_config, z_dim, input_size, device, embeddings_static):
        self.name = "lstm_convolutional"
        lstm_layers = model_config["lstm_layers"]
        cnn_layers = model_config["cnn_layers"]
        self.channels = model_config["channels"]
        kernel_dimension = model_config["kernel_size"]
        hidden_embedding_size = embeddings_static[1]
        vocab = embeddings_static[0]
        super(LSTMConvolutionalVae, self).__init__()
        hidden_embedding_size = int(hidden_embedding_size / 2) * 2
        bidirectional = True
        self.device = device
        lstm_features = int(self.channels / (2 if bidirectional else 1))
        self.embeds = nn.Embedding(vocab, hidden_embedding_size)
        self.rectifier = nn.ReLU()
        self.encoder_rnn = nn.LSTM(hidden_embedding_size, lstm_features, num_layers=lstm_layers,
                                   bidirectional=bidirectional,
                                   batch_first=True)
        self.encoder_cnn = Encoder(cnn_layers, kernel_dimension, input_size, self.channels)
        self.decoder_cnn = Decoder(cnn_layers, kernel_dimension, input_size, self.channels, self.channels, 1)
        self.decoder_rnn = nn.LSTM(self.channels,
                                   hidden_embedding_size,
                                   num_layers=lstm_layers, bidirectional=bidirectional,
                                   batch_first=True)
        self.deembed = nn.Linear(int(hidden_embedding_size * (2 if bidirectional else 1)), vocab)
        h_dim = input_size * self.channels
        self.fc1: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc2: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc3: nn.Module = nn.Linear(z_dim, h_dim)
        self.smax = nn.Softmax(dim=2)

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = self.fc2(h)
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def forward(self, x):
        batch_size = x.shape[0]
        self.encoder_rnn.flatten_parameters()
        self.decoder_rnn.flatten_parameters()
        h = self.encoder_cnn(self.rectifier(self.encoder_rnn(self.embeds(x))[0]).transpose(1, 2)).view(batch_size, -1)
        z, mu, log_var = self.bottleneck(h)
        z = self.fc3(z).view(batch_size, self.channels, -1)
        v1 = self.decoder_cnn(z).transpose(1, 2)
        v2 = self.decoder_rnn(v1)
        val = self.deembed(v2[0])
        return self.smax(val.transpose(1, 2)), mu, log_var
