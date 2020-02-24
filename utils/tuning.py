import multiprocessing

import numpy as np
import torch
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler

from utils.model_factory import create_model


def tuner_run(config):
    train, model, optimizer, device = create_model(config, config)
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
            "mean_accuracy": 0.80,
            "training_iteration": 5 if smoke_test else 10000
        },
        resources_per_trial={
            "cpu": cpus,
            "gpu": gpus
        },
        num_samples=1 if smoke_test else 3,
        config=config_tune)
    print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
