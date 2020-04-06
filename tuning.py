import argparse
import math
import multiprocessing
import os
import os.path as osp

import numpy as np
import ray
import torch
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.utils import pin_in_object_store, get_pinned_object
from torch.utils.data import DataLoader

import utils.data.common
from models.model_factory import create_model
from utils.train import Trainer

config_common_mammalian = {
    'dataset': 'medium', 'protein_length': 1500, 'class': 'mammalian', 'batch_size': 150, 'epochs': 150,
    "iteration_freq": 1000,
    "chem_features": "False",
    'added_length': 0, 'hidden_size': 1000, 'embedding_size': 640, "tuning": True
}

config_common_bacteria = {
    'dataset': 'medium', 'protein_length': 200, 'class': 'bacteria', 'batch_size': 200, 'epochs': 150,
    "iteration_freq": 10,
    'added_length': 0, 'hidden_size': 200, 'embedding_size': 128, "tuning": True
}

model_tuning_configs = {
    "convolutionalBasic": {
        "model_name": "convolutional_basic",
        "kernel_size": {"grid_search": [2]},
        "kernel_expansion_factor": {"grid_search": [2]},
        "channel_scale_factor": {"grid_search": [2]},
        "layers": {"grid_search": [5]},
        "embedding_gradient": "False",
        "chem_features": "False",
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.000000001, 0.001)),
        "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.000001, 0.0001))
    },
    "gated_conv": {
        "model_name": "gated_cnn",
        "layers": {"grid_search": [6, 8]},
        "kernel_size_0": {"grid_search": [21, 33, 49, 65]},
        "channels": {"grid_search": [64, 128, 256]},
        "residual": {"grid_search": [2, 4]},
        "chem_features": "False",
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.01, 0.05))
    },
    "convolutional_old": {
        "model_name": "convolutional_vae",
        "encoder_sizes": [30, 16, 8, 4, 1],
        "decoder_sizes": [23, 16, 8, 4, 1],
        "kernel_sizes_encoder": {"grid_search": [5, 10, 20, 50, 100, 150]},
        "stride_sizes_encoder": {"grid_search": [2, 5, 10, 15, 30]},
        "kernel_sizes_decoder": {"grid_search": [5, 10, 20, 50, 100, 150]},
        "stride_sizes_decoder": {"grid_search": [2, 5, 10, 15, 30]},
        "chem_features": "False",
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.000001, 1)),
        "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.0, 0.05)),
        "tuning": True
    },
    "transformer_convolutional": {
        "model_name": "transformer_convolutional",
        "heads": 8,
        "layers": {"grid_search": [6]},
        "channels": {"grid_search": [128]},
        "kernel_size": {"grid_search": [3]},
        "embedding_gradient": "False",
        "chem_features": "False",
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.00001, 0.001)),
        # "lr": 0.0005279379246234669,
        "weight_decay": 1.6459309598386149e-06,
        "wrap": "True",
        "sched_freq": 400,
        "optimizer": "RAdam"
    },
    "transformer": {
        "model_name": "transformer",
        "heads": {"grid_search": [8]},
        "layers": {"grid_search": [5]},
        "internal_dimension": {"grid_search": [64]},
        "feed_forward": {"grid_search": [64]},
        "embedding_gradient": "False",
        "chem_features": "False",
        # "lr": 1.710853307705144e-05,
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.00000001, 0.00001)),
        "weight_decay": 1.4412730806529451e-06
        # "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.000001, 0.0001))
    }
}


def tuner_run(config):
    track.init()
    print(config)

    model, optimizer, device, _ = create_model(config, config, multigpu=True)

    data_length = config["protein_length"]
    batch_size = config["batch_size"]  # number of data points in each batch
    train_dataset_name = config["tuning_dataset_name"]
    weights_name = config["tuning_weights"]

    print(f"Loading the sequence for train data: {train_dataset_name}")

    train_dataset = get_pinned_object(pinned_dataset)
    weights = utils.data.common.load_from_saved_tensor(weights_name)
    train_iterator = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    train = Trainer(model, config["protein_length"], train_iterator, None, device,
                    optimizer,
                    len(train_dataset),
                    0, 0, vocab_size=data_length, weights=weights, freq=config["iteration_freq"], save_best=False)

    train_dataset_len = train_dataset.shape[0]
    epochs = config["epochs"]
    for e in range(epochs):
        train_kl_loss, recon_loss, train_recon_accuracy, valid_loop = train.train()

        train_kl_loss /= train_dataset_len
        recon_loss /= train_dataset_len

        print_str = f'Epoch {e}, kl: {train_kl_loss:.3f}, recon: {recon_loss:.3f} accuracy {train_recon_accuracy:.2f}'
        total_loss = train_recon_accuracy + train_kl_loss
        if total_loss == math.nan or not valid_loop:
            break
        print(print_str)
        if not debug:
            track.log(mean_loss=(train_kl_loss + recon_loss), accuracy=train_recon_accuracy, kl_loss=train_kl_loss,
                      recon_loss=recon_loss)


def tuner(smoke_test: bool, model, config_type):
    ray.init()
    if config_type == "bacteria":
        config_common = config_common_bacteria
    else:
        config_common = config_common_mammalian
    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()

    model_config = model_tuning_configs[model]

    dataset_type = config_common["dataset"]  # (small|medium|large)
    data_length = config_common["protein_length"]
    if config_common["class"] != "mammalian":
        train_dataset_name = f"data/train_set_{dataset_type}_{data_length}.json"
    else:
        train_dataset_name = "data/train_set_large_1500_mammalian.json"

    max_dataset_length = 80000

    train_dataset, c, score = utils.data.common.load_data_from_file(train_dataset_name)

    train_dataset = utils.data.common.get_shuffled_sample(train_dataset, max_dataset_length)

    tensor_filename = "{}_{}_{}_tuning.pt".format(config_common["class"], data_length, max_dataset_length)
    weights_filename = "{}.{}.{}.wt".format(config_common["class"], data_length, max_dataset_length)
    utils.data.common.save_tensor_to_file(tensor_filename, train_dataset)
    utils.data.common.save_tensor_to_file(weights_filename, score)
    config_common["tuning_dataset_name"] = os.getcwd() + "/" + tensor_filename
    config_common["tuning_weights"] = os.getcwd() + "/" + weights_filename

    config_common["epochs"] = 150
    config_tune = {**config_common, **model_config}
    local_dir = osp.join(os.getcwd(), "logs", "model")
    global pinned_dataset
    pinned_dataset = pin_in_object_store(train_dataset)
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="accuracy")

    analysis = tune.run(
        tuner_run,
        name="exp",
        scheduler=sched,
        stop={
            "training_iteration": 5 if smoke_test else 10
        },
        resources_per_trial={
            "cpu": cpus,
            "gpu": gpus
        },
        local_dir=local_dir,
        num_samples=1 if smoke_test else 3,
        config=config_tune)
    print("Best config is:", analysis.get_best_config(metric="accuracy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-m", "--model",
                        help="Name of the model, options are : convolutionalBasic, gated_conv, convolutional_old",
                        type=str)
    parser.add_argument("-t", "--type", help="Bacteria or mammalian", type=str)
    args = parser.parse_args()

    debug = False

    tuner(False, args.model, args.type)
