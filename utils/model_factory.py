import torch
from torch import optim as optim

from models.convolutional_vae import ConvolutionalVAE
from models.lstm_vae import LSTMVae
from models.simple_vae import VAE


def get_optimizer(optimizer_config, model):
    return optim.Adam(model.parameters(), **optimizer_config)


def create_model(config, model_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fixed_protein_length = config["protein_length"]
    input_dim = fixed_protein_length * (config["feature_length"] + config["added_length"])  # size of each input

    if model_config["model_name"] == "convolutional_vae":
        model = ConvolutionalVAE(model_config["convolutional_parameters"], config["hidden_size"],
                                 config["embedding_size"], config["feature_length"], device)
    else:
        if model_config["model_name"] == "lstm_vae":
            model = LSTMVae(model_config["convolutional_parameters"], config["hidden_size"],
                            config["embedding_size"], config["feature_length"], device)
        else:
            model = VAE(input_dim, 20).to(device)  # 20 is number of hidden dimension

    # optimizer
    return model, get_optimizer(model_config["optimizer_config"], model), device
