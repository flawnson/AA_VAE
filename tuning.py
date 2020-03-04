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
from torch.utils.data import DataLoader

import utils.data as data
from utils.data import read_sequences
from utils.model_factory import create_model
from utils.train import Trainer

model_tuning_configs = {
    "convolutionalBasic": {
        "model_name": "convolutional_basic",
        "kernel_size": {"grid_search": [11, 21, 31, 51, 101]},
        "scale": {"grid_search": [1, 2]},
        "layers": {"grid_search": [4, 6, 8, 16]},
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "weight_decay": tune.uniform(0, 0.9)
    },
    "gated_conv": {
        "model_name": "gated_cnn",
        "layers": {"grid_search": [4, 6, 8, 16]},
        "kernel_size_0": {"grid_search": [7, 9, 11, 17]},
        "kernel_size_1": 30,
        "channels": {"grid_search": [4, 6, 8]},
        "residual": {"grid_search": [2, 4]},
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "weight_decay": tune.uniform(0, 0.9)
    },
    "convolutional_old": {
        "model_name": "convolutional_vae",
        "encoder_sizes": [30, 16, 8, 4, 1],
        "decoder_sizes": [23, 16, 8, 4, 1],
        "kernel_sizes_encoder": tune.grid_search([5, 10, 20, 50, 100, 150]),
        "stride_sizes_encoder": tune.grid_search([2, 5, 10, 15, 30]),
        "kernel_sizes_decoder": tune.grid_search([5, 10, 20, 50, 100, 150]),
        "stride_sizes_decoder": tune.grid_search([2, 5, 10, 15, 30]),
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.000001, 1)),
        "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.0, 0.1)),
        "tuning": True
    }
}


def tuner_run(config):
    track.init()
    print(config)
    tuning: bool = config["tuning"]
    model, optimizer, device = create_model(config, config)

    data_length = config["protein_length"]
    batch_size = config["batch_size"]  # number of data points in each batch
    train_dataset_name = config["tuning_dataset_name"]
    weights_name = config["tuning_weights"]

    print(f"Loading the sequence for train data: {train_dataset_name}")

    train_dataset = data.load_from_saved_tensor(train_dataset_name)
    weights = data.load_from_saved_tensor(weights_name)
    train_iterator = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    train = Trainer(model, config["protein_length"], train_iterator, None, config["feature_length"], device,
                    optimizer,
                    len(train_dataset),
                    0, 0, vocab_size=data_length, weights=weights)

    train_dataset_len = train_dataset.shape[0]
    epochs = config["epochs"]
    for e in range(epochs):
        train_loss, train_recon_accuracy = train.train()

        train_loss /= train_dataset_len
        print(f'Epoch {e}, Train Loss: {train_loss:.8f} Train accuracy {train_recon_accuracy * 100.0:.2f}%')
        if tuning:
            track.log(mean_accuracy=train_recon_accuracy * 100)


def tuner(smoke_test: bool, config_, model):
    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()

    model_config = model_tuning_configs[model]

    dataset_type = config["dataset"]  # (small|medium|large)
    data_length = config["protein_length"]
    if config["class"] != "mammalian":
        train_dataset_name = f"data/train_set_{dataset_type}_{data_length}.json"
    else:
        train_dataset_name = "data/train_set_large_1500_mammalian.json"

    max_dataset_length = 10000

    train_dataset, _, _ = read_sequences(train_dataset_name,
                                         fixed_protein_length=data_length, add_chemical_features=True,
                                         sequence_only=True, pad_sequence=True, fill_itself=False,
                                         max_length=max_dataset_length)

    _, c, score = read_sequences(train_dataset_name,
                                 fixed_protein_length=data_length, add_chemical_features=True,
                                 sequence_only=True, pad_sequence=True, fill_itself=False,
                                 max_length=-1)

    tensor_filename = "{}_{}_{}_tuning.pt".format(config["class"], data_length, max_dataset_length)
    weights_filename = "{}.{}.{}.wt".format(config["class"], data_length, max_dataset_length)
    data.save_tensor_to_file(tensor_filename, train_dataset)
    data.save_tensor_to_file(weights_filename, score)
    config_["tuning_dataset_name"] = os.getcwd() + "/" + tensor_filename
    config_["tuning_weights"] = os.getcwd() + "/" + weights_filename

    config_["epochs"] = 150
    config_tune = {**config_, **model_config}
    local_dir = osp.join(os.getcwd(), "logs", "model")

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
    parser.add_argument("-m", "--model",
                        help="Name of the model, options are : convolutionalBasic, gated_conv, convolutional_old",
                        type=str)
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
        model = args.model
        tuner(False, config, model)
