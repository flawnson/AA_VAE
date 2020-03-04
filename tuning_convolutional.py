import argparse
import json
import multiprocessing
import os
import os.path as osp

import torch
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler

from utils.data import load_data
from utils.model_factory import create_model
from utils.train import Trainer


def tuner_run(config__):
    track.init()
    print(config__)
    tuning: bool = config__["tuning"]
    model, optimizer, device = create_model(config__, config__)
    max_dataset_length = 20000
    data_length = config__["protein_length"]
    train_dataset, test_dataset, train_iterator, test_iterator = load_data(config__, max_dataset_length)
    train = Trainer(model, config__["protein_length"], train_iterator, test_iterator, config__["feature_length"],
                    device,
                    optimizer,
                    len(train_dataset),
                    len(test_dataset), 0, vocab_size=data_length)
    train_dataset_len = train_dataset.shape[0]
    test_dataset_len = test_dataset.shape[0]
    epochs = config__["epochs"]
    for e in range(epochs):
        train_loss, train_recon_accuracy = train.train()
        test_loss, test_recon_accuracy = train.test()

        train_loss /= train_dataset_len
        test_loss /= test_dataset_len
        print(
            f'Epoch {e}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Train accuracy {train_recon_accuracy * 100.0:.2f}%, Test accuracy {test_recon_accuracy * 100.0:.2f}%')
        if tuning:
            track.log(mean_accuracy=test_recon_accuracy * 100)


def tuner(smoke_test: bool, config_):
    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()

    model_config = {
        "model_name": "convolutional_vae",
        "encoder_sizes": [30, 16, 8, 4, 1],
        "decoder_sizes": [23, 16, 8, 4, 1],
        "kernel_sizes_encoder": tune.grid_search([5, 10, 20, 50, 100, 150]),
        "stride_sizes_encoder": tune.grid_search([2, 5, 10, 15, 30]),
        "kernel_sizes_decoder": tune.grid_search([5, 10, 20, 50, 100, 150]),
        "stride_sizes_decoder": tune.grid_search([2, 5, 10, 15, 30]),
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.00001, 1)),
        "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.0001, 0.1)),
        "tuning": True
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
    config_["epochs"] = 50
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
                   'model_name': 'convolutional_vae', 'encoder_sizes': [30, 16, 8, 4, 1],
                   'decoder_sizes': [23, 16, 8, 4, 1], 'kernel_sizes_encoder': 5, 'stride_sizes_encoder': 2,
                   'kernel_sizes_decoder': 5, 'stride_sizes_decoder': 2, 'lr': 0.00939812052381224,
                   'weight_decay': 0.004428770037319564,
                   "tuning": False}
        tuner_run(config_)
    else:
        tuner(False, config)
