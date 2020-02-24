import torch as torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Conv1d or type(m) == nn.ConvTranspose1d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def reparameterization(mu, log_var: torch.Tensor, device):
    std = log_var.mul(0.5).exp_()
    std = std.to(device)
    esp = torch.randn(*mu.size()).to(device)
    z = mu + std * esp
    return z


class ConvolutionalVAE(nn.Module):
    def __init__(self, model_config, h_dim, z_dim, out_dim, device, embeddings_static):
        super(ConvolutionalVAE, self).__init__()
        self.name = "ConvolutionalVAE"
        encoder_sizes: list = model_config["encoder_sizes"]
        # encoder = nn.Sequential(
        self.bne1 = nn.BatchNorm1d(encoder_sizes[0])
        self.ce1 = nn.Conv1d(in_channels=encoder_sizes[0], out_channels=encoder_sizes[1], kernel_size=5, stride=1, padding=2, groups=1)
        self.ce1.apply(init_weights)
        self.re1 = nn.ReLU()
        self.bne2 = nn.BatchNorm1d(encoder_sizes[1])
        self.ce2 = nn.Conv1d(in_channels=encoder_sizes[1], out_channels=encoder_sizes[2], kernel_size=5, stride=1, padding=2, groups=1)
        self.ce2.apply(init_weights)
        self.re2 = nn.ReLU()
        self.bne3 = nn.BatchNorm1d(encoder_sizes[2])
        self.ce3 = nn.Conv1d(in_channels=encoder_sizes[2], out_channels=encoder_sizes[3], kernel_size=5, stride=1, padding=2, groups=1)
        self.ce3.apply(init_weights)
        self.re3 = nn.ReLU()
        self.bne4 = nn.BatchNorm1d(encoder_sizes[3])
        self.ce4 = nn.Conv1d(in_channels=encoder_sizes[3], out_channels=1, kernel_size=5, stride=1, padding=2, groups=1)
        self.ce4.apply(init_weights)
        self.re4 = nn.ReLU()  # ,
        self.smax = nn.Softmax()
        # )

        decoder_sizes: list = model_config["decoder_sizes"]
        # decoder = nn.Sequential(
            # nn.LSTM(),
            # nn.ReLU(),
            # UnFlatten(size=h_dim),
        self.cd1 = nn.ConvTranspose1d(1, decoder_sizes[3], kernel_size=5, stride=1, padding=2)
        self.cd1.apply(init_weights)
        self.rd1 = nn.ReLU()
        self.bnd1 = nn.BatchNorm1d(decoder_sizes[3])
        self.cd2 = nn.ConvTranspose1d(decoder_sizes[3], decoder_sizes[2], kernel_size=5, stride=1, padding=2)
        self.cd2.apply(init_weights)
        self.rd2 = nn.ReLU()
        self.bnd2 = nn.BatchNorm1d(decoder_sizes[2])
        self.cd3 = nn.ConvTranspose1d(decoder_sizes[2], decoder_sizes[1], kernel_size=5, stride=1, padding=2)
        self.cd3.apply(init_weights)
        self.rd3 = nn.ReLU()
        self.bnd3 = nn.BatchNorm1d(decoder_sizes[1])
        self.cd4 = nn.ConvTranspose1d(decoder_sizes[1], decoder_sizes[0], kernel_size=5, stride=1, padding=2)
        self.rd4 = nn.ReLU()
        self.cd4.apply(init_weights)
        # )
        # decoder.apply(init_weights)
        embedding = nn.Embedding(23, encoder_sizes[0], max_norm=1)
        embedding.weight.data.copy_(embeddings_static)
        embedding.weight.requires_grad = False
        self.embedding = embedding
        self.fc1: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc2: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc3: nn.Module = nn.Linear(z_dim, h_dim)
        self.device = device

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = self.fc2(h)
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def representation(self, x):
        return self.bottleneck(self.encoder(self.embedding(x.long()).transpose(1, 2)))[0]

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        h = self.re1(self.ce1((self.bne1(x))))
        h = self.re2(self.ce2((self.bne2(h))))
        h = self.re3(self.ce3((self.bne3(h))))
        h = self.re4(self.ce4((self.bne4(h))))
        z, _, _ = self.bottleneck(h)
        z = self.fc3(z)
        x = self.rd1(self.cd1(z))
        x = self.rd2(self.cd2((self.bnd1(x))))
        x = self.rd3(self.cd3((self.bnd2(x))))
        x = self.rd4(self.cd4((self.bnd3(x))))
        # x = self.smax(x)
        return x
