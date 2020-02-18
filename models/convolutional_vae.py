import torch.nn as nn

from models.vae_template import VaeTemplate


def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.ConvTranspose1d:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class ConvolutionalVAE(VaeTemplate):
    def __init__(self, model_config, h_dim, z_dim, out_dim, device):
        sizes: list = model_config["sizes"]
        encoder = nn.Sequential(
            nn.Conv1d(in_channels=sizes[0], out_channels=sizes[1], kernel_size=5, stride=1, padding=2, groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=sizes[1], out_channels=sizes[2], kernel_size=5, stride=1, padding=2, groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=sizes[2], out_channels=sizes[3], kernel_size=5, stride=1, padding=2, groups=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=sizes[3], out_channels=1, kernel_size=5, stride=1, padding=2, groups=1),
            nn.ReLU()  # ,
            # nn.LSTM()
            # Flatten()
        )
        encoder.apply(init_weights)

        decoder = nn.Sequential(
            # nn.LSTM(),
            # nn.ReLU(),
            # UnFlatten(size=h_dim),
            nn.ConvTranspose1d(1, sizes[3], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(sizes[3], sizes[2], kernel_size=5, stride=1, padding= 2),
            nn.ReLU(),
            nn.ConvTranspose1d(sizes[2], sizes[1], kernel_size=5, stride=1, padding= 2),
            nn.ReLU(),
            nn.ConvTranspose1d(sizes[1], out_dim, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )
        decoder.apply(init_weights)
        super(ConvolutionalVAE, self).__init__(encoder, decoder, device, h_dim, z_dim)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        z = self.fc3(z)
        val = self.decoder(z)
        if self.postprocess is not None:
            val = self.postprocess(val)
        return val, mu, log_var
