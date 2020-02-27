import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.convolutional_linear import Convolutional_Linear_VAE
from models.convolutional_vae import ConvolutionalVAE
from models.linear_vae import LinearVAE
from models.lstm_vae import LSTMVae
from models.gated_cnn import GatedCNN
from utils import data


def load_data(_config, max_length=-1):
    data_length = _config["protein_length"]
    batch_size = _config["batch_size"]  # number of data points in each batch
    train_dataset_name = _config["train_dataset_name"]
    test_dataset_name = _config["test_dataset_name"]

    print(f"Loading the sequence for train data: {train_dataset_name} and test data: {test_dataset_name}")
    _train_dataset = data.read_sequences(train_dataset_name,
                                         fixed_protein_length=data_length, add_chemical_features=True,
                                         sequence_only=True, pad_sequence=True, fill_itself=False,
                                         max_length=max_length)
    _test_dataset = data.read_sequences(test_dataset_name,
                                        fixed_protein_length=data_length, add_chemical_features=True,
                                        sequence_only=True, pad_sequence=True, fill_itself=False, max_length=max_length)
    print(f"Loading the iterator for train data: {train_dataset_name} and test data: {test_dataset_name}")
    _train_iterator = DataLoader(_train_dataset, shuffle=True, batch_size=batch_size)
    _test_iterator = DataLoader(_test_dataset, batch_size=batch_size)
    return _train_dataset, _test_dataset, _train_iterator, _test_iterator


def get_optimizer(optimizer_config, model):
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_model(config, model_config):
    models = {"convolutional_vae": ConvolutionalVAE,
              "lstm_vae": LSTMVae,
              "linear_vae": LinearVAE,
              "convolutional_linear": Convolutional_Linear_VAE,
              "gated_cnn": GatedCNN}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.get(model_config["model_name"])(model_config, config["hidden_size"],
                                                   config["embedding_size"], config["protein_length"], device,
                                                   data.get_embedding_matrix()).to(device)

    # optimizer
    return model, get_optimizer(model_config, model), device
