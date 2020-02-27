import torch
import torch.nn as nn


class LinearLayer(torch.nn.Module):
    def __init__(self, in_size, out_size, layer_sizes, dropout=.25):
        """

        :param in_size: Model input size (embedding size)
        :param out_size: Model output size (target size)
        :param layer_sizes: List of layer sizes
        :param dropout: Dropout percentage (default: .25)
        """
        super(LinearLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout

    def input_layer(self) -> nn.Linear:
        return nn.Linear(self.in_size, self.layer_sizes[0])

    def hidden_layers(self, layer_in, layer_out, *args, **kwargs) -> nn.Sequential:
        """

        :param layer_in: Layer input size
        :param layer_out: Layer output size
        :param args: Linear arguments (bias, etc.)
        :param kwargs: Linear arguments (bias, etc.)
        :return:
        """
        return nn.Sequential(
            nn.Linear(layer_in, layer_out, *args, **kwargs),
            nn.BatchNorm1d(layer_out),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

    def output_layer(self) -> nn.Linear:
        return nn.Linear(self.layer_sizes[-1], self.out_size)

    def forward(self, x):
        return self.model(x)


class LinearModel(torch.nn.Module):
    def __init__(self, in_size, out_size, layer_sizes, dropout=.25):
        super(LinearModel, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout

        self.first_layer = LinearLayer(self.in_size, self.out_size, self.layer_sizes, self.dropout).input_layer()
        self.hidden_layers = [LinearLayer(self.in_size,
                                          self.out_size,
                                          self.layer_sizes,
                                          self.dropout).hidden_layers(in_size, out_size, bias=True)
                                          for in_size, out_size in zip(self.layer_sizes, self.layer_sizes[1:])]
        self.final_layer = LinearLayer(self.in_size, self.out_size, self.layer_sizes).output_layer()

        self.full_model = nn.Sequential(self.first_layer, *self.hidden_layers, self.final_layer)

    def forward(self, data):
        return self.full_model(data)

