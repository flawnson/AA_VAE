"""
This code is a variation of simple VAE from https://graviraja.github.io/vanillavae/
"""
import argparse
import json

import torch

from utils.model_factory import create_model, get_optimizer, load_data
from utils.train import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-m", "--model", help="model config file", type=str)
    parser.add_argument("-b", "--benchmarking", help="benchmarking run config", type=str)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"
    config: dict = json.load(open(args.config))
    model_config: dict = json.load(open(args.model))

    data_length = config["protein_length"]
    number_of_epochs = config["epochs"]  # times to run the model on complete data

    train_dataset, test_dataset, train_iterator, test_iterator = load_data(config)

    print(f"Creating the model")
    model = create_model(config, model_config)

    print(f"Start the training")
    # optimizer
    optimizer = get_optimizer(model_config["optimizer_config"], model)
    Trainer(model, config["protein_length"], train_iterator, test_iterator, config["feature_length"], device, optimizer,
            len(train_dataset),
            len(test_dataset), number_of_epochs, vocab_size=data_length).trainer()
