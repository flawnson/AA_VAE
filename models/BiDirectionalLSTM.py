from models.model_common import *


class Encoder(nn.Module):
    def __init__(self, embeddings_static, lstm_layers, layers, z_dim, input_channels, kernel_size,
                 kernel_expansion_factor, channel_scale_factor, max_channels):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(embeddings_static.shape[0], embeddings_static.shape[1])
        self.embedding.weight.data.copy_(embeddings_static)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embeddings_static.shape[1], input_channels, lstm_layers, dropout=0.1, batch_first=True)
        conv_layers = []
        output_channels = 2 ** ((input_channels - 1).bit_length())
        out_size = z_dim
        base_kernel_size = kernel_size + 1

        for n in range(layers):
            base_kernel_size = kernel_size + 1
            block = ConvolutionalBlock(input_channels, output_channels, base_kernel_size, padded=True)
            conv_layers.append(block)
            kernel_size = kernel_size * kernel_expansion_factor
            input_channels = output_channels
            output_channels = int(output_channels * channel_scale_factor)
            if output_channels > max_channels:
                output_channels = max_channels
            padding = int((base_kernel_size - 1) / 2)

            out_size = int(out_size_conv(out_size, padding, 1, base_kernel_size, 1))

        self.out_size = int(out_size)
        self.out_channels = int(input_channels)
        self.final_kernel_size = base_kernel_size
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        x, _ = self.lstm(self.embedding(x))
        x = x.transpose(0,1).transpose(1,2)
        x = self.conv_layers(x)
        return x


class Decoder(nn.Module):
    def __init__(self, lstm_layers, layers, kernel_size, output_expected, input_size, input_channels: int, output_channels_expected,
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
        self.lstm = nn.LSTM(output_channels_expected, output_channels_expected, lstm_layers, dropout=0.1, batch_first=True)
        assert out_size == output_expected
        self.residue = 200

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.lstm(x)
        return x


class BiDirectionalLSTM(nn.Module):
    def __init__(self, model_config,  z_dim, input_size, device, embeddings_static):
        torch.manual_seed(0)
        super(BiDirectionalLSTM, self).__init__()
        self.name = "bidirectional_lstm"
        lstm_layers = model_config["lstm_layers"]
        conv_layers = model_config["conv_layers"]
        kernel_size = model_config["kernel_size"]
        max_channels = model_config["max_channels"]
        channels = model_config["channels"]
        channel_scale_factor = model_config["channel_scale_factor"]
        kernel_expansion_factor = model_config["kernel_expansion_factor"]
        if kernel_expansion_factor == 1:
            kernel_size = kernel_size - 1

        self.device = device
        self.encoder = Encoder(embeddings_static, lstm_layers, conv_layers, z_dim, channels,
                               kernel_size, kernel_expansion_factor,
                               channel_scale_factor, max_channels)

        h_dim = int(self.encoder.out_size * self.encoder.out_channels)

        self.decoder = Decoder(embeddings_static, lstm_layers, conv_layers, self.encoder.final_kernel_size, input_size,
                               self.encoder.out_size,
                               self.encoder.out_channels,
                               embeddings_static.shape[0], channel_scale_factor,
                               kernel_expansion_factor=kernel_expansion_factor)

        self.fc1: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc2: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc3: nn.Module = nn.Linear(z_dim, h_dim)

        self.predictor = nn.Linear(z_dim, 7)

        self.smax = nn.Softmax(dim=1)

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = self.fc2(h)
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def representation(self, x):
        x = self.encoder(self.embedding(x).transpose(1, 2))
        return x, self.bottleneck(x)

    def forward(self, x):
        h = self.encoder(self.embedding(x).transpose(1, 2))
        z, mu, log_var = self.bottleneck(h)
        class_prediction = self.predictor(z)
        z = self.fc3(z)
        z = z.view(z.shape[0], self.encoder.out_channels, -1)
        val = self.decoder(z).transpose(1, 2)
        return class_prediction, val.transpose(1, 2), mu, log_var
