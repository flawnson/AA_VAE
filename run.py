"""
This code is a variation of simple VAE from https://graviraja.github.io/vanillavae/
"""
import argparse
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.convolutional_vae import ConvolutionalVAE
from models.simple_vae import VAE
from utils import data
from utils.train import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-m", "--model", help="model config file", type=str)
    parser.add_argument("-b", "--benchmarking", help="benchmarking run config", type=str)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config: dict = json.load(open(args.config))
    model_config: dict = json.load(open(args.model))

    DATASET_LENGTH = config["dataset"]  # (small|medium|large)
    FIXED_PROTEIN_LENGTH = config["protein_length"]
    INPUT_DIM = FIXED_PROTEIN_LENGTH * (config["feature_length"] + config["added_length"])  # size of each input

    BATCH_SIZE = config["batch_size"]  # number of data points in each batch
    N_EPOCHS = config["epochs"]  # times to run the model on complete data

    lr = model_config["optimizer_config"]["learning_rate"]  # learning rate

    if config["class"] != "mammalian":
        train_dataset_name = f"data/train_set_{DATASET_LENGTH}_{FIXED_PROTEIN_LENGTH}.json"
        test_dataset_name = f"data/test_set_{DATASET_LENGTH}_{FIXED_PROTEIN_LENGTH}.json"
    else:
        train_dataset_name = "data/train_set_large_1500_mammalian.json"
        test_dataset_name = "data/test_set_large_1500_mammalian.json"

    print(f"Loading the sequence for train data: {train_dataset_name} and test data: {test_dataset_name}")

    train_dataset = data.read_sequences(train_dataset_name,
                                        fixed_protein_length=FIXED_PROTEIN_LENGTH, add_chemical_features=True,
                                        sequence_only=True)
    test_dataset = data.read_sequences(test_dataset_name,
                                       fixed_protein_length=FIXED_PROTEIN_LENGTH, add_chemical_features=True,
                                       sequence_only=True)

    train_iterator = DataLoader(train_dataset, shuffle=True, batch_size= BATCH_SIZE)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    if model_config["model_name"] == "convolutional_vae":
        model = ConvolutionalVAE(model_config["convolutional_parameters"], config["hidden_size"],
                                 config["embedding_size"], config["feature_length"], device,
                                 data.get_embedding_matrix())
    else:
        model = VAE(INPUT_DIM, 20).to(device)  # 20 is number of hidden dimension

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    Trainer(model, config["protein_length"], train_iterator, test_iterator, config["feature_length"], device, optimizer,
            train_dataset,
            test_dataset, N_EPOCHS).trainer()

    SAVE_SNAPSHOT = False
    if SAVE_SNAPSHOT:
        # save a snapshot of the model
        from datetime import datetime

        now = datetime.now()
        date_time = now.strftime("%m_%d-%Y_%H_%M_%S")
        torch.save(model.state_dict(), f"saved_models/{model.name}_{date_time}")
