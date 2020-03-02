import multiprocessing

import torch
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler

from utils.model_factory import create_model
from utils.model_factory import load_data
from utils.train import Trainer
import os
import os.path as osp


def tuner_run(config__):
    track.init()
    print(config__)
    tuning: bool = config__["tuning"]
    model, optimizer, device = create_model(config__, config__)
    max_dataset_length = 20000
    data_length = config__["protein_length"]
    train_dataset, test_dataset, train_iterator, test_iterator = load_data(config__, max_dataset_length)
    train = Trainer(model, config__["protein_length"], train_iterator, test_iterator, config__["feature_length"], device,
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
            track.log(mean_accuracy=test_recon_accuracy.clone().detach().to("cpu").item() * 100)


def tuner(smoke_test: bool, config_, model_config_):
    cpus = int(multiprocessing.cpu_count())
    gpus = torch.cuda.device_count()

    model_config_["lr"] = tune.sample_from(lambda spec: tune.loguniform(0.00001, 1)),
    model_config_["weight_decay"] = tune.sample_from(lambda spec: tune.loguniform(0.0001, 0.1)),
    model_config_["tuning"] = True

    dataset_type = config_["dataset"]  # (small|medium|large)
    data_length = config_["protein_length"]
    if config_["class"] != "mammalian":
        train_dataset_name = f"data/train_set_{dataset_type}_{data_length}.json"
        test_dataset_name = f"data/test_set_{dataset_type}_{data_length}.json"
    else:
        train_dataset_name = "data/train_set_large_1500_mammalian.json"
        test_dataset_name = "data/test_set_large_1500_mammalian.json"
    config_["train_dataset_name"] = os.getcwd() + "/" + train_dataset_name
    config_["test_dataset_name"] = os.getcwd() + "/" + test_dataset_name
    config_["epochs"] = 50
    config_tune = {**config_, **model_config_}
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

