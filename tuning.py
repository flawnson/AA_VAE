import argparse
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

import utils.data as data
from utils.model_factory import create_model
from utils.train import Trainer

config_common_mammalian = {
    'dataset': 'medium', 'protein_length': 1500, 'class': 'mammalian', 'batch_size': 200, 'epochs': 150,
    'added_length': 0, 'hidden_size': 1500, 'embedding_size': 750, "tuning": True
}

config_common_bacteria = {
    'dataset': 'medium', 'protein_length': 200, 'class': 'bacteria', 'batch_size': 200, 'epochs': 150,
    'added_length': 0, 'hidden_size': 200, 'embedding_size': 40, "tuning": True
}

model_tuning_configs = {
    "convolutionalBasic": {
        "model_name": "convolutional_basic",
        "kernel_size": {"grid_search": [2, 4, 6, 8]},
        "expansion_factor": {"grid_search": [1, 2, 4]},
        "scale": {"grid_search": [1, 2]},
        "layers": {"grid_search": [4, 5, 6, 8]},
        "embedding_gradient": "True",
        "chem_features": {"grid_search": ["False", "True"]},
        "lr": tune.sample_from(lambda spec: tune.loguniform(0.000000001, 0.001)),
        "weight_decay": tune.sample_from(lambda spec: tune.loguniform(0.000001, 0.0001))
    },
    "gated_conv": {
        "model_name": "gated_cnn",
        "layers": {"grid_search": [6, 8, 16]},
        "kernel_size_0": {"grid_search": [11, 21, 31, 51]},
        "channels": {"grid_search": [8, 16, 32]},
        "residual": {"grid_search": [2, 4, 6]},
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
    train = Trainer(model, config["protein_length"], train_iterator, None, device,
                    optimizer,
                    len(train_dataset),
                    0, 0, vocab_size=data_length, weights=weights)

    train_dataset_len = train_dataset.shape[0]
    epochs = config["epochs"]
    for e in range(epochs):
        train_loss, recon_loss, train_recon_accuracy = train.train(e)

        train_loss /= train_dataset_len
        recon_loss /= train_dataset_len
        print(
            f'Epoch {e}, Train Loss: {train_loss:.8f}, {recon_loss:.8f} Train accuracy {train_recon_accuracy * 100.0:.2f}%')
        if not debug:
            track.log(mean_loss=(train_loss + recon_loss), accuracy=train_recon_accuracy, kl_loss=train_loss,
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

    max_dataset_length = 50000

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
        num_samples=1 if smoke_test else 2,
        config=config_tune)
    print("Best config is:", analysis.get_best_config(metric="accuracy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-m", "--model",
                        help="Name of the model, options are : convolutionalBasic, gated_conv, convolutional_old",
                        type=str)
    parser.add_argument("-d", "--debug", help="Debugging or full scale", type=str)
    parser.add_argument("-t", "--type", help="Bacteria or mammalian", type=str)
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
        tuner(False, args.model, args.type)
