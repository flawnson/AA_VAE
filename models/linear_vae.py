import torch
from models.vae_template import VaeTemplate


class LinearVAE(VaeTemplate):
    def __init__(self, model_config, hidden_size, embedding_size, device, embeddings_static):

        encoder_sizes: list = model_config["encoder_sizes"]
        encoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(encoder_sizes[0]),
            torch.nn.Linear(encoder_sizes[0], encoder_sizes[1]),  # 2 for bidirection
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(encoder_sizes[1]),
            torch.nn.Linear(encoder_sizes[1], encoder_sizes[2]),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(encoder_sizes[2]),
            torch.nn.Linear(encoder_sizes[2], encoder_sizes[3]),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(encoder_sizes[3]),
            torch.nn.Linear(encoder_sizes[3], hidden_size),
            torch.nn.LeakyReLU()
        )

        decoder_sizes: list = model_config["decoder_sizes"]
        decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.Linear(embedding_size, decoder_sizes[3]),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(decoder_sizes[3]),
            torch.nn.Linear(decoder_sizes[3], decoder_sizes[2]),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(decoder_sizes[2]),
            torch.nn.Linear(decoder_sizes[2], decoder_sizes[1]),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(decoder_sizes[1]),
            torch.nn.Linear(decoder_sizes[1], decoder_sizes[0]),
            torch.nn.LeakyReLU()
        )
        embedding = torch.nn.Embedding(23, 30, max_norm=1)
        embedding.weight.data.copy_(embeddings_static)
        embedding.weight.requires_grad = False
        super(LinearVAE, self).__init__(encoder, decoder, device, hidden_size, embedding_size, embedding=embedding)

    def forward(self, x):
        x = self.embedding(x).view(x.shape[0], -1)
        if self.preprocess is not None:
            x = self.preprocess(x)
        h = self.encoder(x)
        z, _, _ = self.bottleneck(h)
        z = self.fc3(z)
        val = self.decoder(z)
        if self.postprocess is not None:
            val = self.postprocess(val)
        return val
