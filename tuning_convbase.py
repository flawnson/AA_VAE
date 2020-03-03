import argparse
import json
import multiprocessing
import os
import os.path as osp

import numpy as np
import torch
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler

from utils.model_factory import create_model
from utils.data import load_data, read_sequences
from utils.train import Trainer


def tuner_run(config):
    track.init()
    print(config)
    tuning: bool = config["tuning"]
    model, optimizer, device = create_model(config, config)
    max_dataset_length = 20000

    data_length = config["protein_length"]
    batch_size = config["batch_size"]  # number of data points in each batch
    train_dataset_name = config["train_dataset_name"]

    print(f"Loading the sequence for train data: {train_dataset_name}")
    train_dataset, c, score = read_sequences(train_dataset_name,
                                             fixed_protein_length=data_length, add_chemical_features=True,
                                             sequence_only=True, pad_sequence=True, fill_itself=False,
                                             max_length=10000)
    train = Trainer(model, config["protein_length"], train_iterator, test_iterator, config["feature_length"], device,
                    optimizer,
                    len(train_dataset),
                    len(test_dataset), 0, vocab_size=data_length, weights=score)
    train_dataset_len = train_dataset.shape[0]
    epochs = config["epochs"]
    for e in range(epochs):
        train_loss, train_recon_accuracy = train.train()

        train_loss /= train_dataset_len
        print(f'Epoch {e}, Train Loss: {train_loss:.8f} Train accuracy {train_recon_accuracy * 100.0:.2f}%')
        if tuning:
            track.log(mean_accuracy=train_recon_accuracy * 100)


def tuner(smoke_test: bool, config_):
    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()

    model_config = {
        "model_name": "convolutional_basic",
        "kernel_size": {"grid_search": [11, 21, 31]},
        "scale": {"grid_search": [2, 3]},
        "layers": {"grid_search": [4, 6, 8, 16]},
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "weight_decay": tune.uniform(0, 0.9)
    }

    dataset_type = config["dataset"]  # (small|medium|large)
    data_length = config["protein_length"]
    if config["class"] != "mammalian":
        train_dataset_name = f"data/train_set_{dataset_type}_{data_length}.json"
        test_dataset_name = f"data/test_set_{dataset_type}_{data_length}.json"
    else:
        train_dataset_name = "data/train_set_large_1500_mammalian.json"
        test_dataset_name = "data/test_set_large_1500_mammalian.json"
    config_["train_dataset_name"] = os.getcwd() + "/" + train_dataset_name
    config_["test_dataset_name"] = os.getcwd() + "/" + test_dataset_name
    config_["epochs"] = 150
    config_tune = {**config_, **model_config}
    local_dir = osp.join(os.getcwd(), "logs")
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_accuracy")
    analysis = tune.run(
        tuner_run,
        name="exp",
        scheduler=sched,
        stop={
            "training_iteration": 5 if smoke_test else 100
        },
        resources_per_trial={
            "cpu": cpus,
            "gpu": gpus
        },
        local_dir=local_dir,
        num_samples=1 if smoke_test else 3,
        config=config_tune)
    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-d", "--debug", help="Debugging or full scale", type=str)
    args = parser.parse_args()
    config: dict = json.load(open(args.config))
    if args.debug == "True":
        config_ = {'dataset': 'small', 'protein_length': 50, 'class': 'bacteria', 'batch_size': 20, 'epochs': 4,
                   'feature_length': 30, 'added_length': 0, 'hidden_size': 50, 'embedding_size': 20,
                   'train_dataset_name': '/home/jyothish/PycharmProjects/simple-vae/data/train_set_small_50.json',
                   'test_dataset_name': '/home/jyothish/PycharmProjects/simple-vae/data/test_set_small_50.json',
                   "model_name": "gated_cnn",
                   "layers": 5,
                   "kernel_size_0": 11,
                   "kernel_size_1": 30,
                   "channels": 6,
                   "residual": 2,
                   "lr": 0.00020297,
                   "weight_decay": 0.0,
                   "tuning": False}
        tuner_run(config_)
    else:
        tuner(False, config)
