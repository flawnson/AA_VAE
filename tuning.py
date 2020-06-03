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
from utils.amino_acid_loader import process_sequences
from utils.model_factory import create_model
from utils.training.train import Trainer

config_common_mammalian = {
    'dataset': 'medium', 'protein_length': 1500, 'class': 'mammalian', 'batch_size': 400, 'epochs': 150,
    "chem_features": "False",
    "optimizer": "RAdam",
    'added_length': 0, 'hidden_size': 1000, 'embedding_size': 448, "tuning": True
}

config_common_bacteria = {
    'dataset': 'medium', 'protein_length': 200, 'class': 'bacteria', 'batch_size': 200, 'epochs': 150,
    "iteration_freq": 10,
    'added_length': 0, 'hidden_size': 200, 'embedding_size': 128, "tuning": True
}

config_common_human = {
    'dataset': 'human', 'protein_length': 1500, 'class': 'human', 'batch_size': 10, 'epochs': 150,
    'embedding_size': 768, "tuning": True
}

model_tuning_configs = {
    "convolutional_basic": {
        "model_name": "convolutional_basic",
        "kernel_size": {"grid_search": [2]},
        "kernel_expansion_factor": {"grid_search": [2]},
        "channel_scale_factor": {"grid_search": [2]},
        "layers": {"grid_search": [5]},
        "embedding_gradient": "False",
        "chem_features": "False",
        "lr": {"grid_search": [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]},
        # tune.sample_from(lambda spec: tune.loguniform(0.000001, 0.1)),
        "weight_decay": 1.6459309598386149e-06
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
        "layers": {"grid_search": [6]},
        "channels": {"grid_search": [16]},
        "kernel_size": {"grid_search": [3]},
        "embedding_gradient": "False",
        "chem_features": "False",
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.00001, 0.01)),
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
        "internal_dimension": {"grid_search": [32]},
        "feed_forward": {"grid_search": [32]},
        "embedding_gradient": "False",
        "chem_features": "False",
        # "lr": 1.710853307705144e-05,
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.00000001, 0.001)),
        "weight_decay": 1.4412730806529451e-06,
        "LearningRateScheduler": "CosineWarmRestarts",
        "wrap": "False",
        "sched_freq": 20000,
        "optimizer": "RAdam"
        # "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.000001, 0.0001))
    },
    "gcn": {
        "model_name": "global_context_vae",
        "layers": 4,
        "channels": 16,
        "kernel_size": {"grid_search": [5]},
        "embedding_gradient": "False",
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.0001, 0.01)),
        "sched_freq": 40,
        "weight_decay": 1.6459309598386149e-06,
        "optimizer": "RAdam"
    },
    "lstm_convolutional": {
        "model_name": "lstm_convolutional",
        "lstm_layers": {"grid_search": [4, 8]},
        "cnn_layers": {"grid_search": [4, 8]},
        "channels": {"grid_search": [128, 256]},
        "kernel_size": {"grid_search": [17, 33]},
        "embedding_gradient": "False",
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.0000001, 0.01)),
        "weight_decay": 1.6459309598386149e-06,
        "optimizer": "Adam"
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
                    optimizer, n_epochs=0,
                    weights=weights, save_best=False)

    train_dataset_len = train_dataset.shape[0]
    epochs = config["epochs"]
    for e in range(epochs):
        train_kl_loss, recon_loss, train_recon_accuracy, valid_loop = train.train()

        train_kl_loss /= train_dataset_len
        recon_loss /= train_dataset_len

        print_str = f'Epoch {e}, kl: {train_kl_loss:.6f}, recon: {recon_loss:.6f} accuracy {train_recon_accuracy:.6f}'
        total_loss = train_recon_accuracy + train_kl_loss
        if total_loss == math.nan or not valid_loop:
            break
        print(print_str)
        if not debug:
            track.log(mean_loss=(train_kl_loss + recon_loss), accuracy=train_recon_accuracy, kl_loss=train_kl_loss,
                      recon_loss=recon_loss)


def tuner(smoke_test: bool, model, config_type):
    ray.init()
    config_classes = {
        "bacteria": config_common_bacteria,
        "mammalian": config_common_mammalian,
        "human": config_common_human
    }
    config_common = config_classes[config_type]
    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()

    model_config = model_tuning_configs[model]

    dataset_type = config_common["dataset"]  # (small|medium|large)
    data_length = config_common["protein_length"]
    dataset_map = {
        "bacteria": f"data/train_set_{dataset_type}_{data_length}.json",
        "mammalian": "data/validation_set_large_no_ofr_no_trim_1500_mammalian.json",
        "human": "data/human_proteins.json"
    }
    train_dataset_name = dataset_map[config_common["class"]]

    max_dataset_length = 80000

    train_dataset, c, score, _ = process_sequences(utils.data.common.load_data_from_file(train_dataset_name),
                                                   max_dataset_length, data_length, pad_sequence=True,
                                                   pt_file="validation_set_tuning.pt")

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
        time_attr="training_iteration", metric="mean_loss")

    analysis = tune.run(
        tuner_run,
        name="exp",
        scheduler=sched,
        stop={
            "training_iteration": 5 if smoke_test else 4
        },
        resources_per_trial={
            "cpu": cpus,
            "gpu": gpus
        },
        local_dir=local_dir,
        num_samples=3 if smoke_test else 1,
        config=config_tune)
    print("Best config is:", analysis.get_best_config(metric="mean_loss", mode="min"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-m", "--model",
                        help="Name of the model, options are : convolutionalBasic, gated_conv, convolutional_old, gcn,"
                             " lstm_convolutional",
                        type=str)
    parser.add_argument("-t", "--type", help="human, bacteria or mammalian", type=str)
    args = parser.parse_args()

    debug = False

    tuner(False, args.model, args.type)
