import torch
import torch.nn as nn


class LinearPredictor(nn.Module):
    """
    Concatenate the two embeddings for a protein and predict whether an interaction between them wil happen or not.
    """

    def __init__(self, data_length, layer_sizes,targets):
        super().__init__()
        self.name = "linear_vae"
        self.predictor = torch.nn.Sequential(
            torch.nn.BatchNorm1d(data_length),
            torch.nn.Linear(data_length, layer_sizes[0]),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(layer_sizes[0]),
            torch.nn.Linear(layer_sizes[0], layer_sizes[1]),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(layer_sizes[1]),
            torch.nn.Linear(layer_sizes[1], layer_sizes[2]),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(layer_sizes[2]),
            torch.nn.Linear(layer_sizes[2], targets)
        )

    def forward(self,  embeddings):
        return self.predictor(embeddings)