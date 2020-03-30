import argparse
import json
import os

import torch

from utils.data import load_data
from utils.logger import log
from models.model_factory import create_model
from utils.train import Trainer

if __name__ == "__main__":
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-m", "--model", help="model config file", type=str)
    parser.add_argument("-g", "--multigpu", help="multigpu mode", action="store_true")
    parser.add_argument("-p", "--pretrained", help="pretrained", type=str)
    parser.add_argument("-s", "--save", help="Save the model", action="store_true")
    args = parser.parse_args()
    config: dict = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
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
    log.info("Creating the model")
    model, optimizer, device, model_name = create_model(config, model_config, args.pretrained, args.multigpu)
    log.info("Loading the data")
    train_dataset, test_dataset, train_iterator, test_iterator, c, score = load_data(config)
    log.info("Start the training")
    iteration_freq = config["iteration_freq"]
    # optimizer
    Trainer(model, config["protein_length"], train_iterator, test_iterator, device, optimizer,
            len(train_dataset),
            len(test_dataset), number_of_epochs, vocab_size=data_length, weights=score, model_name=model_name,
            freq=iteration_freq, save_best=args.save).trainer()
