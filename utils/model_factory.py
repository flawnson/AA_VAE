import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.convolutional_linear import Convolutional_Linear_VAE
from models.convolutional_vae import ConvolutionalVAE
from models.linear_vae import LinearVAE
from models.lstm_vae import LSTMVae
from utils import data


def load_data(_config, tuning: bool = False):
    dataset_type = _config["dataset"]  # (small|medium|large)
    if tuning:
        max_length = 10000
    else:
        max_length = -1
    data_length = _config["protein_length"]
    batch_size = _config["batch_size"]  # number of data points in each batch
    if _config["class"] != "mammalian":
        train_dataset_name = f"data/train_set_{dataset_type}_{data_length}.json"
        test_dataset_name = f"data/test_set_{dataset_type}_{data_length}.json"
    else:
        train_dataset_name = "data/train_set_large_1500_mammalian.json"
        test_dataset_name = "data/test_set_large_1500_mammalian.json"
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
    return optim.Adam(model.parameters(), **optimizer_config)


def create_model(config, model_config):
    models = {"convolutional_vae": ConvolutionalVAE,
              "lstm_vae": LSTMVae,
              "linear_vae": LinearVAE,
              "convolutional_linear": Convolutional_Linear_VAE}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.get(model_config["model_name"])(model_config["model_parameters"], config["hidden_size"],
                                                   config["embedding_size"], config["feature_length"], device,
                                                   data.get_embedding_matrix()).to(device)

    # optimizer
    return model, get_optimizer(model_config["optimizer_config"], model), device
