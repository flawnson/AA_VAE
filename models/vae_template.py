import torch
import torch.nn as nn


def reparameterization(mu, log_var):
    std = log_var.mul(0.5).exp_()
    esp = torch.randn(*mu.size())
    z = mu + std * esp
    return z


class VaeTemplate(nn.Module):
    def __init__(self, encoder, decoder, h_dim=1024, z_dim=32, preprocessing_func = None, post_processing_func = None):
        super(VaeTemplate, self).__init__()
        self.preprocess = preprocessing_func
        self.postprocess = post_processing_func
        self.encoder: nn.Module = encoder
        self.fc1: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc2: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc3: nn.Module = nn.Linear(z_dim, h_dim)
        self.decoder: nn.Module = decoder

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = self.fc2(h)
        z = reparameterization(mu, log_var)
        return z, mu, log_var

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        z = self.fc3(z)
        val = self.decoder(z)
        if self.postprocess is not None:
            val = self.postprocess(val)
        return val, mu, log_var
