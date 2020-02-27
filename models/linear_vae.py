import torch
import torch.nn as nn

from models.vae_template import VaeTemplate


class LinearVAE(VaeTemplate, nn.Module):
    def __init__(self, model_config, hidden_size, embedding_size, data_length, device, embeddings_static):
        self.data_length = data_length
        self.name = "linear_vae"
        encoder_sizes: list = model_config["encoder_sizes"]
        encoder = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(encoder_sizes[0] * data_length),
            torch.nn.Linear(encoder_sizes[0] * data_length, encoder_sizes[1] * data_length),  # 2 for bidirection
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(encoder_sizes[1] * data_length),
            torch.nn.Linear(encoder_sizes[1] * data_length, encoder_sizes[2] * data_length),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(encoder_sizes[2] * data_length),
            torch.nn.Linear(encoder_sizes[2] * data_length, encoder_sizes[3] * data_length),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(encoder_sizes[3] * data_length),
            torch.nn.Linear(encoder_sizes[3] * data_length, hidden_size),
            torch.nn.LeakyReLU()
        )

        decoder_sizes: list = model_config["decoder_sizes"]
        decoder = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(hidden_size),
            torch.nn.Linear(hidden_size, decoder_sizes[3] * data_length),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(decoder_sizes[3] * data_length),
            torch.nn.Linear(decoder_sizes[3] * data_length, decoder_sizes[2] * data_length),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(decoder_sizes[2] * data_length),
            torch.nn.Linear(decoder_sizes[2] * data_length, decoder_sizes[1] * data_length),
            torch.nn.LeakyReLU(),
            # torch.nn.BatchNorm1d(decoder_sizes[1] * data_length),
            torch.nn.Linear(decoder_sizes[1] * data_length, decoder_sizes[0] * data_length),
            torch.nn.LeakyReLU()
        )
        embedding = torch.nn.Embedding(23, 30, max_norm=1)
        embedding.weight.data.copy_(embeddings_static)
        embedding.weight.requires_grad = False
        super(LinearVAE, self).__init__(encoder, decoder, device, hidden_size, embedding_size, embedding=embedding)

    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)
        h = self.encoder(x)
        z, _, _ = self.bottleneck(h)
        z = self.fc3(z)
        val = self.decoder(z)
        val = val.view(x.shape[0], 23, -1)
        return val
