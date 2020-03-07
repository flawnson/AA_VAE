"""
This code is a variation of simple VAE from https://graviraja.github.io/vanillavae/
"""
import argparse
import json
import torch
import os
from utils.model_factory import create_model
from utils.data import load_data
from utils.train import Trainer

if __name__ == "__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-m", "--model", help="model config file", type=str)
    parser.add_argument("-b", "--benchmarking", help="benchmarking run config", type=str)
    args = parser.parse_args()
    config: dict = json.load(open(args.config))

    model_config: dict = json.load(open(args.model))

    data_length = config["protein_length"]
    number_of_epochs = config["epochs"]  # times to run the model on complete data
    dataset_type = config["dataset"]  # (small|medium|large)
    if config["class"] != "mammalian":
        train_dataset_name = f"data/train_set_{dataset_type}_{data_length}.json"
        test_dataset_name = f"data/test_set_{dataset_type}_{data_length}.json"
    else:
        train_dataset_name = "data/train_set_large_1500_mammalian.json"
        test_dataset_name = "data/test_set_large_1500_mammalian.json"
    config["train_dataset_name"] = os.getcwd() + "/" + train_dataset_name
    config["test_dataset_name"] = os.getcwd() + "/" + test_dataset_name
    train_dataset, test_dataset, train_iterator, test_iterator, c, score = load_data(config)

    print(f"Creating the model")
    model, optimizer, device = create_model(config, model_config)

    print(f"Start the training")
    # optimizer
    Trainer(model, config["protein_length"], train_iterator, test_iterator, config["feature_length"], device, optimizer,
            len(train_dataset),
            len(test_dataset), number_of_epochs, vocab_size=data_length, weights=score).trainer()
