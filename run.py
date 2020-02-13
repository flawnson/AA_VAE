"""
This code is a variation of simple VAE from https://graviraja.github.io/vanillavae/
"""
import argparse
import json

import numpy as np
import torch
import torch.optim as optim
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler
from torch.utils.data import DataLoader

from models.convolutional_vae import ConvolutionalVAE
from models.lstm_vae import LSTMVae
from models.simple_vae import VAE
from utils import data
from utils.train import Trainer


def tunerRun(config):
    train_dataset = data.read_sequences(f"data/train_set_{DATASET_LENGTH}_{FIXED_PROTEIN_LENGTH}.json",
                                        fixed_protein_length=FIXED_PROTEIN_LENGTH, add_chemical_features=False)
    test_dataset = data.read_sequences(f"data/test_set_{DATASET_LENGTH}_{FIXED_PROTEIN_LENGTH}.json",
                                       fixed_protein_length=FIXED_PROTEIN_LENGTH, add_chemical_features=False)

    train_iterator = DataLoader(train_dataset, shuffle=True)
    test_iterator = DataLoader(test_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config["model_name"] == "convolutional_vae":
        model = ConvolutionalVAE(config["convolutional_parameters"], config["hidden_size"],
                                 config["embedding_size"], config["feature_length"], device)
    if config["model_name"] == "lstm_vae":
        model = LSTMVae(config["convolutional_parameters"], config["hidden_size"],
                        config["embedding_size"], config["feature_length"], device)
    else:
        model = VAE(config["feature_length"], 20).to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])

    trainer = Trainer(model, config["protein_length"], train_iterator, test_iterator, config["feature_length"], device, optimizer,
                      train_dataset,
                      test_dataset, config["epochs"])
    while True:
        trainer.train(model, optimizer, device)
        loss, acc = trainer.test(model, device)
        track.log(mean_accuracy=acc)


def tuner(smoke_test: bool):
    import multiprocessing

    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_accuracy")
    analysis = tune.run(
        tunerRun,
        name="exp",
        scheduler=sched,
        stop={
            "mean_accuracy": 0.80,
            "training_iteration": 5 if smoke_test else 10000
        },
        resources_per_trial={
            "cpu": cpus,
            "gpu": gpus
        },
        num_samples=1 if smoke_test else 3,
        config={
            "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
            "momentum": tune.uniform(0.1, 0.9),
            "use_gpu": True
        })
    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
def create_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config: dict = json.load(open(args.config))
    model_config: dict = json.load(open(args.model))

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
    return Trainer(model, config["protein_length"], train_iterator, test_iterator, config["feature_length"], device, optimizer,
            train_dataset,
            test_dataset, model_config["epochs"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-m", "--model", help="model config file", type=str)
    parser.add_argument("-b", "--benchmarking", help="benchmarking run config", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--training', dest='training', action='store_true')
    feature_parser.add_argument('--tuning', dest='training', action='store_false')
    parser.set_defaults(training=True)
    args = parser.parse_args()

    if args.training:
        trainer = create_model()
        trainer.trainer()

        SAVE_SNAPSHOT = False

        if SAVE_SNAPSHOT:
            trainer.save_snapshot()
    else:



