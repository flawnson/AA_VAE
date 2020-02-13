import torch
from torch import optim as optim
from torch.utils.data import DataLoader

from models.convolutional_vae import ConvolutionalVAE
from models.lstm_vae import LSTMVae
from models.simple_vae import VAE
from utils import data
from utils.train import Trainer


def create_model(config, model_config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATASET_LENGTH = config["dataset"]  # (small|medium|large)
    FIXED_PROTEIN_LENGTH = config["protein_length"]
    INPUT_DIM = FIXED_PROTEIN_LENGTH * (config["feature_length"] + config["added_length"])  # size of each input

    BATCH_SIZE = model_config["batch_size"]  # number of data points in each batch
    N_EPOCHS = model_config["epochs"]  # times to run the model on complete data

    lr = model_config["optimizer_config"]["learning_rate"]  # learning rate

    train_dataset = data.read_sequences(f"data/train_set_{DATASET_LENGTH}_{FIXED_PROTEIN_LENGTH}.json",
                                        fixed_protein_length=FIXED_PROTEIN_LENGTH, add_chemical_features=False)
    test_dataset = data.read_sequences(f"data/test_set_{DATASET_LENGTH}_{FIXED_PROTEIN_LENGTH}.json",
                                       fixed_protein_length=FIXED_PROTEIN_LENGTH, add_chemical_features=False)

    train_iterator = DataLoader(train_dataset, shuffle=True)
    test_iterator = DataLoader(test_dataset)

    if model_config["model_name"] == "convolutional_vae":
        model = ConvolutionalVAE(model_config["convolutional_parameters"], config["hidden_size"],
                                 config["embedding_size"], config["feature_length"], device)
    else:
        if model_config["model_name"] == "lstm_vae":
            model = LSTMVae(model_config["convolutional_parameters"], config["hidden_size"],
                            config["embedding_size"], config["feature_length"], device)
        else:
            model = VAE(INPUT_DIM, 20).to(device)  # 20 is number of hidden dimension

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return Trainer(model, config["protein_length"], train_iterator, test_iterator, config["feature_length"], device,
                   optimizer,
                   train_dataset,
                   test_dataset, N_EPOCHS), model, optimizer, device