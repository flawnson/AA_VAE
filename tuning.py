import argparse
import multiprocessing
import os
import os.path as osp

import numpy as np
import torch
import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.utils import pin_in_object_store, get_pinned_object
from torch.utils.data import DataLoader

import utils.data as data
from utils.model_factory import create_model
from utils.train import Trainer

config_common = {
    'dataset': 'small', 'protein_length': 1500, 'class': 'mammalian', 'batch_size': 1000, 'epochs': 150,
    'feature_length': 23, 'added_length': 0, 'hidden_size': 1500, 'embedding_size': 600, "tuning": True
}

model_tuning_configs = {
    "convolutionalBasic": {
        "model_name": "convolutional_basic",
        "kernel_size": {"grid_search": [11, 21, 31, 51]},
        "scale": {"grid_search": [1, 2]},
        "layers": {"grid_search": [4, 6, 8, 10]},
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "weight_decay": 0.0
    },
    "gated_conv": {
        "model_name": "gated_cnn",
        "layers": {"grid_search": [6, 8, 16]},
        "kernel_size_0": {"grid_search": [11, 21, 31, 51]},
        "channels": {"grid_search": [8, 16, 32]},
        "residual": {"grid_search": [2, 4, 6]},
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "weight_decay": 0.0
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
        "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.0, 0.05)),
        "tuning": True
    }
}


def tuner_run(config):
    track.init()
    print(config)

    model, optimizer, device = create_model(config, config)

    data_length = config["protein_length"]
    batch_size = config["batch_size"]  # number of data points in each batch
    train_dataset_name = config["tuning_dataset_name"]
    weights_name = config["tuning_weights"]

    print(f"Loading the sequence for train data: {train_dataset_name}")

    # train_dataset = data.load_from_saved_tensor(train_dataset_name)
    train_dataset = get_pinned_object(pinned_dataset)
    weights = data.load_from_saved_tensor(weights_name)
    train_iterator = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    train = Trainer(model, config["protein_length"], train_iterator, None, config["feature_length"], device,
                    optimizer,
                    len(train_dataset),
                    0, 0, vocab_size=data_length, weights=weights)

    train_dataset_len = train_dataset.shape[0]
    epochs = config["epochs"]
    for e in range(epochs):
        train_loss, train_recon_accuracy = train.train(e)

        train_loss /= train_dataset_len
        print(f'Epoch {e}, Train Loss: {train_loss:.8f} Train accuracy {train_recon_accuracy * 100.0:.2f}%')
        if not debug:
            track.log(mean_accuracy=train_recon_accuracy * 100)


def tuner(smoke_test: bool, model):
    ray.init()
    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()

    model_config = model_tuning_configs[model]

    dataset_type = config_common["dataset"]  # (small|medium|large)
    data_length = config_common["protein_length"]
    if config_common["class"] != "mammalian":
        train_dataset_name = f"data/train_set_{dataset_type}_{data_length}.json"
    else:
        train_dataset_name = "data/train_set_large_1500_mammalian.json"

    max_dataset_length = 20000

    train_dataset, c, score = data.read_sequences(train_dataset_name,
                                                  fixed_protein_length=data_length, add_chemical_features=True,
                                                  sequence_only=True, pad_sequence=True, fill_itself=False,
                                                  max_length=-1)

    train_dataset = data.get_shuffled_sample(train_dataset, max_dataset_length)

    tensor_filename = "{}_{}_{}_tuning.pt".format(config_common["class"], data_length, max_dataset_length)
    weights_filename = "{}.{}.{}.wt".format(config_common["class"], data_length, max_dataset_length)
    data.save_tensor_to_file(tensor_filename, train_dataset)
    data.save_tensor_to_file(weights_filename, score)
    config_common["tuning_dataset_name"] = os.getcwd() + "/" + tensor_filename
    config_common["tuning_weights"] = os.getcwd() + "/" + weights_filename

    config_common["epochs"] = 150
    config_tune = {**config_common, **model_config}
    local_dir = osp.join(os.getcwd(), "logs", "model")
    global pinned_dataset
    pinned_dataset = pin_in_object_store(train_dataset)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_accuracy")

    analysis = tune.run(
        tuner_run,
        name="exp",
        scheduler=sched,
        stop={
            "training_iteration": 5 if smoke_test else 15
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
    parser.add_argument("-m", "--model",
                        help="Name of the model, options are : convolutionalBasic, gated_conv, convolutional_old",
                        type=str)
    parser.add_argument("-d", "--debug", help="Debugging or full scale", type=str)
    args = parser.parse_args()
    debug = False
    if debug:
        ray.init()
        train_dataset = data.load_from_saved_tensor(
            '/home/jyothish/PycharmProjects/simple-vae/mammalian_1500_10000_tuning.pt')
        train_dataset = data.get_shuffled_sample(train_dataset, 10000)
        pinned_dataset = pin_in_object_store(train_dataset)
        config_ = {'dataset': 'small', 'protein_length': 1500, 'class': 'mammalian', 'batch_size': 1000, 'epochs': 5,
                   'feature_length': 23, 'added_length': 0, 'hidden_size': 1500, 'embedding_size': 600,
                   'tuning_dataset_name': '/home/jyothish/PycharmProjects/simple-vae/mammalian_1500_10000_tuning.pt',
                   'tuning_weights': '/home/jyothish/PycharmProjects/simple-vae/mammalian.1500.10000.wt',
                   'model_name': 'gated_cnn', 'layers': 4,
                   'kernel_size_0': 7, 'channels': 4,
                   'residual': 2,
                   "lr": 0.005,
                   "weight_decay": 0.0,
                   "tuning": True
                   }

        tuner_run(config_)
    else:
        tuner(False, args.model)
