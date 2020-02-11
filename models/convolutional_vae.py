import torch.nn as nn

from models.vae_template import VaeTemplate


class ConvolutionalVAE(VaeTemplate):
    def __init__(self, model_config, h_dim, z_dim, out_dim, device):
        sizes: list = model_config["sizes"]
        encoder = nn.Sequential(
            nn.Conv1d(in_channels=sizes[0], out_channels=sizes[1], kernel_size=5, stride=1, padding=2, groups=1),
            nn.ReLU(),
            # nn.Conv1d(in_channels=sizes[1], out_channels=sizes[2], kernel_size=5, stride=1, padding=2, groups=1),
            # nn.ReLU(),
            # nn.Conv1d(in_channels=sizes[2], out_channels=sizes[3], kernel_size=5, stride=1, padding=2, groups=1),
            # nn.ReLU(),
            nn.Conv1d(in_channels=sizes[1], out_channels=1, kernel_size=5, stride=1, padding=2, groups=1),
            nn.ReLU()  # ,
            # nn.LSTM()
            # Flatten()
        )

        decoder = nn.Sequential(
            # nn.LSTM(),
            # nn.ReLU(),
            # UnFlatten(size=h_dim),
            nn.ConvTranspose1d(1, sizes[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # nn.ConvTranspose1d(sizes[3], sizes[2], kernel_size=5, stride=1, padding= 2),
            # nn.ReLU(),
            # nn.ConvTranspose1d(sizes[2], sizes[1], kernel_size=5, stride=1, padding= 2),
            # nn.ReLU(),
            nn.ConvTranspose1d(sizes[1], out_dim, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )
        super(ConvolutionalVAE, self).__init__(encoder, decoder, device, h_dim, z_dim)

    def forward(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        z = self.fc3(z)
        val = self.decoder(z)
        if self.postprocess is not None:
            val = self.postprocess(val)
        return val, mu, log_var
