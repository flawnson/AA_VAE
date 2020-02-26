import argparse
import json
import multiprocessing

import numpy as np
import torch
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler

from utils.model_factory import create_model
from utils.model_factory import load_data
from utils.train import Trainer

import tensorflow as tf  # Needed to prevent get_global_worker attribute error


def tuner_run(config):
    print(config)
    model, optimizer, device = create_model(config, config)
    max_dataset_length = 20000
    data_length = config["protein_length"]
    train_dataset, test_dataset, train_iterator, test_iterator = load_data(config, max_dataset_length)
    train = Trainer(model, config["protein_length"], train_iterator, test_iterator, config["feature_length"], device,
                    optimizer,
                    len(train_dataset),
                    len(test_dataset), 0, vocab_size=data_length)
    train_dataset_len = train_dataset.shape[0]
    test_dataset_len = test_dataset.shape[0]
    epochs = config["epochs"]
    for e in range(epochs):
        train_loss, train_recon_accuracy = train.train()
        test_loss, test_recon_accuracy = train.test()

        train_loss /= train_dataset_len
        test_loss /= test_dataset_len
        print(
            f'Epoch {e}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Train accuracy {train_recon_accuracy * 100.0:.2f}%, Test accuracy {test_recon_accuracy * 100.0:.2f}%')
        track.log(mean_accuracy=test_recon_accuracy)


def tuner(smoke_test: bool, config_):
    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()
    model_config = {
        "model_name": "convolutional_vae",
        "encoder_sizes": [30, tune.grid_search([30, 16, 8]), tune.grid_search([16, 8, 4]),
                          tune.grid_search([16, 8, 4, 2]), 1],
        "decoder_sizes": [23, tune.grid_search([16, 8]), tune.grid_search([16, 8, 4]), tune.grid_search([8, 4, 2]),
                          1],
        "kernel_sizes_encoder": tune.grid_search([2, 4, 8, 16, 32, 64, 128]),
        "stride_sizes_encoder": tune.grid_search([2, 4, 8, 16, 32]),
        "kernel_sizes_decoder": tune.grid_search([2, 4, 8, 16, 32, 64, 128]),
        "stride_sizes_decoder": tune.grid_search([2, 4, 8, 16, 32]),
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.0001, 1))
    }
    config_["epochs"] = 400
    config_tune = {**config_, **model_config}
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_accuracy")
    analysis = tune.run(
        tuner_run,
        name="exp",
        scheduler=sched,
        stop={
            "mean_accuracy": 0.70,
            "training_iteration": 5 if smoke_test else 100
        },
        resources_per_trial={
            "cpu": cpus,
            "gpu": gpus
        },
        num_samples=1 if smoke_test else 3,
        config=config_tune)
    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    args = parser.parse_args()
    config: dict = json.load(open(args.config))
    tuner(False, config)
