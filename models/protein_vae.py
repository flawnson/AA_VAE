import torch
import torch.nn.functional as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import argparse
import utils

from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split as tts

class VAE(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, batch_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size

        self.fc = torch.nn.Linear(input_size, hidden_sizes[0])  # 2 for bidirection
        self.BN = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc1 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.BN1 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc2 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.BN2 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc3_mu = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc3_sig = torch.nn.Linear(hidden_sizes[2], hidden_sizes[3])

        self.fc4 = torch.nn.Linear(hidden_sizes[3] + 8, hidden_sizes[2])
        self.BN4 = torch.nn.BatchNorm1d(hidden_sizes[2])
        self.fc5 = torch.nn.Linear(hidden_sizes[2], hidden_sizes[1])
        self.BN5 = torch.nn.BatchNorm1d(hidden_sizes[1])
        self.fc6 = torch.nn.Linear(hidden_sizes[1], hidden_sizes[0])
        self.BN6 = torch.nn.BatchNorm1d(hidden_sizes[0])
        self.fc7 = torch.nn.Linear(hidden_sizes[0], input_size - 8)

    def sample_z(self, x_size, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn(x_size, self.hidden_sizes[-1])
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x, code, struc=None):
        ###########
        # Encoder #
        ###########

        # get the code from the tensor
        # add the conditioned code
        x = torch.cat((x, code), 1)
        # Layer 0
        out1 = self.fc(x)
        out1 = nn.relu(self.BN(out1))
        # Layer 1
        out2 = self.fc1(out1)
        out2 = nn.relu(self.BN1(out2))
        # Layer 2
        out3 = self.fc2(out2)
        out3 = nn.relu(self.BN2(out3))
        # Layer 3 - mu
        mu = self.fc3_mu(out3)
        # layer 3 - sig
        sig = nn.softplus(self.fc3_sig(out3))

        ###########
        # Decoder #
        ###########

        # sample from the distro
        sample = self.sample_z(x.size(0), mu, sig)
        # add the conditioned code
        sample = torch.cat((sample, code), 1)
        # Layer 4
        out4 = self.fc4(sample)
        out4 = nn.relu(self.BN4(out4))
        # Layer 5
        out5 = self.fc5(out4)
        out5 = nn.relu(self.BN5(out5))
        # Layer 6
        out6 = self.fc6(out5)
        out6 = nn.relu(self.BN6(out6))
        # Layer 7
        out7 = nn.sigmoid(self.fc7(out6))

        return out7, mu, sig


# =============================================================================
# Create and Load model into memory
# =============================================================================

vae = VAE(X_dim, hidden_size, batch_size)