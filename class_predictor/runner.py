import copy

import torch.nn.modules.activation

from models.linear_predictor import LinearPredictor
from models.model_common import *
from utils import amino_acid_loader
from utils.logger import log


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerLayer(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerLayer, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequnce to the encoder (required).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of channel self-attn using Squeeze expand network along with
    a spatial Non linear SAGAN using a gaussian  network.
    This encoder is based on the GCNet paper.

    Args:
        dropout: the dropout value (default=0.1).
        kernel_size: size of the kernel

    """

    def __init__(self, channels, dropout=0.1, kernel_size=3):
        super(TransformerEncoderLayer, self).__init__()
        self.spatial_attention = GCNContextBlock(inplanes=channels, ratio=8)
        self.dropout1 = nn.Dropout(dropout)
        out_c = int(channels / 2)
        self.channel_attention = nn.Sequential(
            ConvolutionalBlock(in_c=channels, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalBlock(in_c=out_c, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalBlock(in_c=out_c, out_c=out_c, padded=True, kernel_size=kernel_size),
            ConvolutionalBlock(in_c=out_c, out_c=channels, padded=True, kernel_size=kernel_size),
        )

    def forward(self, src):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
        Shape:
            see the docs in Transformer class.
        """
        residual = src
        src = src + self.dropout1(self.spatial_attention(src))
        src = self.channel_attention(src) + residual
        return src


class GlobalContextVAEModel(nn.Module):
    def __init__(self, model_config, z_dim, input_size, device, embeddings_static):
        torch.manual_seed(0)
        self.device = device

        self.name = "global_context_vae"
        super(GlobalContextVAEModel, self).__init__()
        self.model_type = 'GCA_vae'
        self.src_mask = None
        layers = model_config["layers"]
        self.channels = model_config["channels"]
        kernel_dimension = model_config["kernel_size"]
        self.triple_encoder = nn.Conv1d(kernel_size=3, in_channels=embeddings_static.shape[1],
                                        out_channels=self.channels, stride=1, padding=1, bias=False)
        self.protein_embedding = nn.Embedding(embeddings_static.shape[0], embeddings_static.shape[1])
        self.protein_embedding.weight.data.copy_(embeddings_static)
        self.protein_embedding.weight.requires_grad = False

        encoder_layers = TransformerEncoderLayer(channels=self.channels, kernel_size=kernel_dimension)
        self.transformer_encoder = TransformerLayer(encoder_layers, layers)

        h_dim = input_size * self.channels
        self.fc1: nn.Module = nn.Linear(h_dim, z_dim)
        self.fc2: nn.Module = nn.Linear(h_dim, z_dim)
        self.activation = nn.Softmax(dim=1)
        # self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.triple_encoder.weight.data.uniform_(-initrange, initrange)
        self.deembed.weight.data.uniform_(-initrange, initrange)

    def bottleneck(self, h):
        mu = self.fc1(h)
        log_var = (self.fc2(h))
        z = reparameterization(mu, log_var, self.device)
        return z, mu, log_var

    def forward(self, x):
        z, mu, log_var = self.bottleneck(
            self.transformer_encoder(self.triple_encoder(self.protein_embedding(x).transpose(1, 2)))
                .view(x.shape[0], -1))
        return z, mu, log_var

    def representation(self, x):
        x = self.transformer_encoder(self.triple_encoder(self.protein_embedding(x).transpose(1, 2))).view(x.shape[0],
                                                                                                          -1)
        return self.bottleneck(x)


class Predictor(nn.Module):
    def __init__(self, model_config, z_dim, input_size, device, embeddings_static, requires_grad=True):
        torch.manual_seed(0)
        self.device = device

        self.name = "predictor"
        super(Predictor, self).__init__()
        self.embedder = GlobalContextVAEModel(model_config, z_dim, input_size, device, embeddings_static)
        self.predictor = LinearPredictor(448, [128, 32, 8])

    def forward(self, x):
        return self.predictor(self.embedder(x))


import torch

from utils.amino_acid_loader import process_sequence
import pandas as pd
from torch.utils.data import DataLoader, Dataset

import math
from utils.training.common import reconstruction_accuracy


def inner_iteration(x, labels, training: bool, model, device, optimizer, criterion, reconstruction_accuracy):
    """
    This method runs a single inner iteration or passes a single batch through the model
    :param labels:
    :param reconstruction_accuracy:
    :param criterion:
    :param optimizer:
    :param device:
    :param model:
    :param x:
    :param training:
    :return:
    """

    x = x.long().to(device)

    # update the gradients to zero
    if training:
        optimizer.zero_grad()
    # forward pass
    predicted = model(x)

    recon_loss = criterion(predicted, labels)
    # reconstruction accuracy
    recon_accuracy = reconstruction_accuracy(predicted, labels)

    # backward pass
    if training:
        if not math.isnan(recon_loss):
            recon_loss.backward()
        else:
            v1 = torch.isnan(x).any()
            v2 = torch.isnan(predicted).any()
            log.error("recon: {} x:{} predicted:{}".format(recon_loss, v1, v2))
            recon_loss = criterion(predicted, x)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)

        optimizer.step()
        optimizer.zero_grad()

    return recon_loss.item(), recon_accuracy


def train(model, device, optimizer, criterion, reconstruction_accuracy):
    """
    Run a training iteration over the entire dataset
    """
    # set the train mode
    model.train()

    # Statistics of the epoch
    train_recon_loss = 0
    valid_loop = True
    for i, data in enumerate(train_iterator):
        x, labels, _, _ = data
        recon_loss, accuracy = inner_iteration(x, labels, True, model, device, optimizer, criterion,
                                               reconstruction_accuracy)
        total_loss = recon_loss
        if math.isnan(total_loss):
            log.error("Loss was nan, loop is breaking, change parameters")
            valid_loop = False
            break

        train_recon_loss += recon_loss
        if (i % 100) == 0:
            log.debug("Recon: {} ".format(train_recon_loss))
    return train_recon_loss, valid_loop


class IDMapper:
    def __init__(self):
        self.id = 0
        self.mapper = {}

    def class_to_number(self, c: str):
        if self.mapper.get(c, -1) == -1:
            self.mapper[c] = self.id
            self.id = self.id + 1
        return self.mapper[c]


class LabeledProtein(Dataset):

    def __init__(self, sequence, protein_class, protein_fold, protein_super):
        self.sequence = sequence
        self.protein_class = protein_class
        self.protein_fold = protein_fold
        self.protein_super = protein_super

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, idx):
        return self.sequence[idx], self.protein_class[idx], self.protein_fold[idx], self.protein_super[idx]


if __name__ == "__main__":
    class_id_mapper = IDMapper()
    fold_id_mapper = IDMapper()
    super_family_id_mapper = IDMapper()

    train_data = pd.read_csv("../data/scop_from_domain/train.csv")

    train_protein_class = [class_id_mapper.class_to_number(x) for x in train_data["class"]]
    train_protein_fold = [fold_id_mapper.class_to_number(x) for x in train_data["fold"]]
    train_protein_super = [super_family_id_mapper.class_to_number(x) for x in (train_data["superfamily"])]
    train_sequence_dataset = [process_sequence(x) for x in train_data["sequence"]]
    train_iterator = DataLoader(
        LabeledProtein(train_sequence_dataset, train_protein_class, train_protein_fold, train_protein_super),
        shuffle=True,
        batch_size=100)
    embeddings_map = amino_acid_loader.get_embedding_matrix(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_data = pd.read_csv("../data/scop_from_domain/test.csv")
    test_protein_class = [class_id_mapper.class_to_number(x) for x in test_data["class"]]
    test_protein_fold = [fold_id_mapper.class_to_number(x) for x in test_data["fold"]]
    test_protein_super = [super_family_id_mapper.class_to_number(x) for x in (test_data["superfamily"])]
    test_sequence_dataset = [process_sequence(x) for x in test_data["sequence"]]

    test_iterator = DataLoader(
        LabeledProtein(test_sequence_dataset, test_protein_class, test_protein_fold, test_protein_super))
    gcn_config = {"layers": 4, "channels": 16, "kernel_size": 5}
    model = Predictor(gcn_config, 448, 1500, device, embeddings_map).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    for x in range(100):
        train(model, device, optimizer, loss_func, reconstruction_accuracy)
