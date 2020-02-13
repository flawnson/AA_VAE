import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from ray import tune


def tune_model(config):
    tune.track.init()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=5e-5)

    loss_state, accs_state = 0, 0
    loss_no_improve, accs_no_improve = 0, 0

    for epoch in range(2001):
        model.train()
        optimizer.zero_grad()
        logits, latent_1, latent_2, latent_3 = model()

        loss = F.cross_entropy(logits[alpha],
                               config.get('data').y[alpha],
                               weight=weights)
        _loss = loss.clone().detach().to("cpu").item()

        if config.get("early_stopping_loss"):
            if _loss > loss_state:
                loss_no_improve += 1

            if loss_no_improve > config.get("loss_patience"):
                print('Loss failed to decrease for {} iter, early stopping current iter'.format(config['patience']))
                break

        loss.backward()
        optimizer.step()

        model.eval()
        logits, latent_1, latent_2, latent_3 = model()
        accs, auroc_scores, f1_scores = [], [], []


        train_acc, test_acc = accs
        tune.track.log(train_accuracy=train_acc, test_accuracy=test_acc)

        if config.get('early_stopping_accs'):
            if train_acc > accs_state:
                accs_no_improve += 1

            if loss_no_improve > config.get("accs_patience"):
                print('Accuracy failed to decrease for {} iter, early stopping current iter'.format(
                    config['accs_patience']))
                break


class Tuner:
    def __init__(self, data_set, task, masks: list, tuning_config, device):
        train, test = masks

        tuning_config["train_mask"] = train
        tuning_config["test_mask"] = test

        tuning_config["task"] = task
        tuning_config["device"] = device

        tuning_config["dataset"] = data_set
        tuning_config["data"] = data_set[0]
        tuning_config["layer_1"] = tune.grid_search(tuning_config.get("layer_1_size"))
        tuning_config["layer_2"] = tune.grid_search(tuning_config.get("layer_2_size"))
        tuning_config["learning_rate"] = tune.grid_search(tuning_config.get("lr"))

        self.tuning_config = tuning_config

    def run_tune(self):
        tuning_run_name = "tuning_quaternary"
        import multiprocessing

        cpus = int(multiprocessing.cpu_count())
        gpus = torch.cuda.device_count()
        analysis = tune.run(
            tune_model,
            config=self.tuning_config,
            num_samples=3,
            local_dir=osp.join("..", "logs", tuning_run_name),
            resources_per_trial={"cpu": cpus, "gpu": gpus}
        )

        tune_log("Best config: {}".format(analysis.get_best_config(metric="train_accuracy")))
        tune_log("Best config: {}".format(analysis.get_best_config(metric="test_accuracy")))
        tune_log("Best config: {}".format(analysis.get_best_config(metric="val_accuracy")))
        tune_log("Best config: {}".format(analysis.get_best_config(metric="loss", mode="min")))

        df = analysis.dataframe()

        df.to_csv(path_or_buf=osp.join("..", "logs", logging_run_name))

        return analysis.get_best_config(metric="train_accuracy")  # needs to return config for best model
