import argparse
import json
import subprocess

import torch

import utils.logger as logger
from utils.data_load import load_data
from utils.logger import log
from utils.model_factory import create_model
from utils.training.train import Trainer


def get_data_set_default(class_name):
    if class_name != "mammalian":
        train_dataset_name = f"data/train_set_{dataset_type}_{data_length}.json"
        test_dataset_name = f"data/test_set_{dataset_type}_{data_length}.json"
    else:
        train_dataset_name = "data/train_set_large_no_ofr_no_trim_1500_mammalian.json"
        test_dataset_name = "data/test_set_large_no_ofr_no_trim_1500_mammalian.json"
    return train_dataset_name, test_dataset_name


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
    model_config: dict = json.load(open(args.model))
    logger.set_file_logger()
    data_length = config["protein_length"]
    number_of_epochs = config["epochs"]  # times to run the model on complete data
    if config.get("train_dataset_name", "NA") == "NA":
        dataset_type = config["dataset"]  # (small|medium|large)
        train_dataset_name, test_dataset_name = get_data_set_default(config["class"])
        config["train_dataset_name"] = train_dataset_name
        config["test_dataset_name"] = test_dataset_name

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    log.info("Creating the model")
    model, optimizer, device, model_name = create_model(config, model_config, args.pretrained, args.multigpu)

    log.info("Loading the data")
    train_iterator, test_iterator, c, score, length_scores = load_data(config)

    log.info(f"Model config:{model_config}")
    log.info(f"General config:{config}")
    log.info(f"{args}")

    git_hash = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
    log.info(f"Git hash: {git_hash}, branch: {git_branch}")

    log.info("Start the training")
    Trainer(model, config["protein_length"], train_iterator, test_iterator, device, optimizer,
            number_of_epochs, weights=score, model_name=model_name,
            save_best=args.save, length_stats=length_scores).trainer()
