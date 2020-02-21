import torch as torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.ConvTranspose1d or type(m) ==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def reparameterization(mu, log_var: torch.Tensor, device):
    std = log_var.mul(0.5).exp_()
    std = std.to(device)
    esp = torch.randn(*mu.size()).to(device)
    z = mu + std * esp
    return z


class Convolutional_Linear_VAE(nn.Module):
    def __init__(self, model_config, h_dim, z_dim, out_dim, device, embeddings_static):
        super(Convolutional_Linear_VAE, self).__init__()
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
        )
        linear_config = model_config["linear"]
        self.linear1 = nn.Linear(linear_config[0], linear_config[1])
        encoder.apply(init_weights)
        self.linear1.apply(init_weights)

        decoder = nn.Sequential(
            # nn.LSTM(),
            # nn.ReLU(),
            # UnFlatten(size=h_dim),
            nn.ConvTranspose1d(1, sizes[3], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(sizes[3], sizes[2], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(sizes[2], sizes[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.ConvTranspose1d(sizes[1], out_dim, kernel_size=5, stride=1, padding=2)
        )
        decoder.apply(init_weights)
        embedding = nn.Embedding(23, sizes[0])
        embedding.weight.data.copy_(embeddings_static)
        embedding.weight.requires_grad = False
        self.embedding = embedding
        self.encoder: nn.Module = encoder
        self.fc1: nn.Module = nn.Linear(linear_config[1], z_dim)
        self.fc2: nn.Module = nn.Linear(linear_config[1], z_dim)
        self.fc3: nn.Module = nn.Linear(z_dim, linear_config[1])
        self.linear2 = nn.Linear(linear_config[1], linear_config[0])
        self.linear2.apply(init_weights)
        self.decoder: nn.Module = decoder
        self.device = device

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = self.fc2(h)
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        x = x.long()
        x1 = self.embedding(x).transpose(1, 2)
        h = self.linear1(self.encoder(x1))
        z, _, _ = self.bottleneck(h)
        z = self.linear2(self.fc3(z))
        val = self.decoder(z)
        return val
