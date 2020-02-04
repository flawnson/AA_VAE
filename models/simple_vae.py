import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, input_dim, hidden_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        hidden = F.relu(self.linear(x))
        return hidden


class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, hidden_dim, output_dim):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        predicted = torch.sigmoid(self.out(x))
        return predicted


class VAE(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM):
        ''' This the VAE, which takes a encoder and decoder.

        '''
        super().__init__()

        self.enc = Encoder(INPUT_DIM, HIDDEN_DIM)
        self.dec = Decoder(HIDDEN_DIM, INPUT_DIM)

    def forward(self, x):
        x_sample = self.enc(x)
        predicted = self.dec(x_sample)

        return predicted
