import torch
from models.vae_template import VaeTemplate


class LinearVAE(VaeTemplate):
    def __init__(self, input_size, hidden_sizes, device, embeddings_static):
        encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_sizes[0]),  # 2 for bidirection
            torch.nn.BatchNorm1d(hidden_sizes[0]),
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.BatchNorm1d(hidden_sizes[1]),
            torch.nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            torch.nn.BatchNorm1d(hidden_sizes[2])
        )

        decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_sizes[3], hidden_sizes[2]),
            torch.nn.BatchNorm1d(hidden_sizes[2]),
            torch.nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            torch.nn.BatchNorm1d(hidden_sizes[1]),
            torch.nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            torch.nn.BatchNorm1d(hidden_sizes[0]),
            torch.nn.Linear(hidden_sizes[0], input_size)
        )
        embedding = torch.nn.Embedding(23, 30, max_norm=1)
        embedding.weight.data.copy_(embeddings_static)
        embedding.weight.requires_grad = False
        super(LinearVAE, self).__init__(encoder, decoder, device, hidden_sizes[2], hidden_sizes[3], embedding=embedding)

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
