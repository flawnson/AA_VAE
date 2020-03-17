from utils.model_common import *


def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.ConvTranspose1d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class Encoder(nn.Module):
    def __init__(self, layers, kernel_size, input_size, input_channels: int, channel_scale_factor, max_channels=256,
                 kernel_expansion_factor=1):
        super().__init__()

        conv_layers = []

        output_channels = 2 ** ((input_channels - 1).bit_length())
        out_size = input_size
        base_kernel_size = kernel_size + 1

        for n in range(layers):
            base_kernel_size = kernel_size + 1
            block = ConvolutionalBlock(input_channels, output_channels, base_kernel_size)
            conv_layers.append(block)
            kernel_size = kernel_size * kernel_expansion_factor
            input_channels = output_channels
            output_channels = int(output_channels * channel_scale_factor)
            if output_channels > max_channels:
                output_channels = max_channels
            out_size = out_size_conv(out_size, 0, 1, base_kernel_size, 1)

        self.out_size = int(out_size)
        self.out_channels = int(input_channels)
        self.final_kernel_size = base_kernel_size
        self.conv_layers = nn.ModuleList(conv_layers)
        self.residue = 200

    def forward(self, x):
        inv = x
        i = 1
        for convolution_layer in self.conv_layers:
            x = convolution_layer(x)
            if i % self.residue == 0:
                out = x + inv
                inv = out
                x = out
            i = i + 1
        x = x.view(x.shape[0], -1)
        return x


class Decoder(nn.Module):
    def __init__(self, layers, kernel_size, output_expected, input_size, input_channels: int, output_channels_expected,
                 channel_scale_factor, max_channels=256, kernel_expansion_factor=1):
        super().__init__()
        conv_layers = []

        output_channels = input_channels
        out_size = input_size
        base_kernel_size = kernel_size

        for n in range(layers - 1):
            block = ConvolutionalTransposeBlock(input_channels, output_channels, base_kernel_size)
            conv_layers.append(block)
            kernel_size = int(kernel_size / kernel_expansion_factor)
            input_channels = output_channels
            output_channels = int(output_channels / channel_scale_factor)
            if output_channels > max_channels:
                output_channels = max_channels
            out_size = out_size_transpose(out_size, 0, 1, base_kernel_size, 1)
            if kernel_expansion_factor > 1:
                base_kernel_size = kernel_size + 1

        block = ConvolutionalTransposeBlock(input_channels, output_channels_expected, base_kernel_size)
        out_size = out_size_transpose(out_size, 0, 1, base_kernel_size, 1)
        conv_layers.append(block)

        self.conv_layers = nn.ModuleList(conv_layers)

        self.out_size = out_size
        assert out_size == output_expected
        self.residue = 200

    def forward(self, x):
        inv = x
        i = 1
        for convolution_layer in self.conv_layers:
            x = convolution_layer(x)
            if i % self.residue == 0:
                out = x + inv
                inv = out
                x = out
            i = i + 1
        return x
        # return self.conv_layers(x)


class ConvolutionalBaseVAE(nn.Module):
    def __init__(self, model_config, h_dim, z_dim, input_size, device, embeddings_static, requires_grad=True):
        torch.manual_seed(0)
        super(ConvolutionalBaseVAE, self).__init__()
        self.name = "convolutional_basic"
        kernel_size = model_config["kernel_size"]
        layers = model_config["layers"]
        channel_scale_factor = model_config["channel_scale_factor"]
        kernel_expansion_factor = model_config["kernel_expansion_factor"]
        if kernel_expansion_factor == 1:
            kernel_size = kernel_size - 1

        self.device = device
        self.encoder = Encoder(layers, kernel_size, input_size, embeddings_static.shape[1], channel_scale_factor,
                               kernel_expansion_factor=kernel_expansion_factor)
        h_dim = int(self.encoder.out_size * self.encoder.out_channels)
        self.decoder = Decoder(layers, self.encoder.final_kernel_size, input_size, self.encoder.out_size,
                               self.encoder.out_channels,
                               embeddings_static.shape[0], channel_scale_factor,
                               kernel_expansion_factor=kernel_expansion_factor)

        self.encoder.apply(init_weights)
        self.decoder.apply(init_weights)
        self.fc1: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc2: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc3: nn.Module = nn.Linear(z_dim, h_dim)
        self.fc1.apply(init_weights)
        self.fc2.apply(init_weights)
        self.fc3.apply(init_weights)
        embedding = nn.Embedding(embeddings_static.shape[0], embeddings_static.shape[1])
        embedding.weight.data.copy_(embeddings_static)
        embedding.weight.requires_grad = False

        self.embedding = embedding
        self.smax = nn.Softmax(dim=2)

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = self.fc2(h)
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(self.embedding(x).transpose(1, 2))
        z, mu, log_var = self.bottleneck(h)
        z = self.fc3(z)
        z = z.view(z.shape[0], self.encoder.out_channels, -1)
        val = self.decoder(z).transpose(1, 2)
        return val.transpose(1, 2), mu, log_var
