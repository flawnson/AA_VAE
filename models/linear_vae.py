import torch
from models.vae_template import VaeTemplate


class LinearVAE(VaeTemplate):
    def __init__(self, input_size, hidden_sizes, device):
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

        super(LinearVAE, self).__init__(encoder, decoder, device, hidden_sizes[2], hidden_sizes[3])
