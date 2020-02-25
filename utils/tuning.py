import multiprocessing

import numpy as np
import torch
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler

from utils.model_factory import create_model
from utils.model_factory import load_data
from utils.train import Trainer


def tuner_run(config):
    model, optimizer, device = create_model(config, config)

    data_length = config["protein_length"]
    train_dataset, test_dataset, train_iterator, test_iterator = load_data(config)
    train = Trainer(model, config["protein_length"], train_iterator, test_iterator, config["feature_length"], device,
                    optimizer,
                    len(train_dataset),
                    len(test_dataset), 0, vocab_size=data_length)
    while True:
        train.train()
        loss, acc = train.test()
        track.log(mean_accuracy=acc)


def tuner(smoke_test: bool, config, model_config):
    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()
    tune_config = {**config, **model_config}
    for k, v in config["tunable"]:
        if isinstance(v, dict):
            for k1, v1 in v:
                tune_config[k][k1] = tune.grid_search(v1)
        else:
            tune_config[k] = tune.grid_search(v)
    z2 = {
        "lr": tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "momentum": tune.uniform(0.1, 0.9),
    }

    config_tune = {**tune_config, **z2}
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration", metric="mean_accuracy")
    analysis = tune.run(
        tuner_run,
        name="exp",
        scheduler=sched,
        stop={
            "mean_accuracy": 0.70,
            "training_iteration": 5 if smoke_test else 10000
        },
        resources_per_trial={
            "cpu": cpus,
            "gpu": gpus
        },
        num_samples=1 if smoke_test else 3,
        config=config_tune)
    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
