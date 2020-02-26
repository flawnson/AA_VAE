import json
import torch
import argparse
import numpy as np
import os.path as osp

from torch import nn
from torch.nn import functional as f
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from models.validation.data_processing import *
from models.validation.trainer import TrainLinear


class LinearModel(torch.nn.Module):
    def __init__(self, in_size, out_size, layer_sizes, dropout=.25):
        """

        :param in_size: Model input size (embedding size)
        :param out_size: Model output size (target size)
        :param layer_sizes: List of layer sizes
        :param dropout: Dropout percentage (default: .25)
        """
        super(LinearModel, self).__init__()

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

    def model(self) -> nn.Sequential:
        first_layer = self.input_layer()
        hidden_layers = [self.hidden_layers(in_size, out_size, bias=True)
                         for in_size, out_size in zip(self.layer_sizes, self.layer_sizes[1:])]
        final_layer = self.output_layer()

        full_model = nn.Sequential(first_layer, *hidden_layers, final_layer)

        return full_model


if __name__ == "__main__":
    # print(LinearModel(1, 10, [22, 33, 44, 55, 66]).model())
    # print(list(LinearModel(1, 10, [22, 33, 44, 55, 66]).model().parameters()))
    path = osp.join('simple-vae', 'configs')  # Implicitly used to get config file?
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-f", "--config", help="json config file", type=str)
    args = parser.parse_args()

    embed_file = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "exports", "embeddings.json")
    json_embed = open(embed_file)
    json_data = json.load(json_embed)

    json_file = open(args.config)
    json_config = json.load(json_file)

    data_config = json_config.get('data_config')
    data = ProteinLabels(json_data)
    dataset = DataLoader(dataset=data, batch_size=data_config.get('batch_size'), pin_memory=True)

    model_config = json_config.get('model_config')
    print(model_config)
    print(model_config.get('layer_sizes'))
    # print(model_config['layer_sizes'])
    exit()
    model = LinearModel(model_config.get('in_size'),
                        model_config.get('out_size'),
                        model_config.get('layer_sizes'),
                        model_config.get('dropout'))
    print(list(model.parameters()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_config = json_config.get('train_config')
    TrainLinear(train_config, dataset, model, device).run()
