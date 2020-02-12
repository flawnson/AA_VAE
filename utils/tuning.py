import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from ray import tune
from sklearn.metrics import f1_score, roc_auc_score


def tune_model(config):
    tune.track.init()

    dataset = config.get('dataset')
    data = config.get("data")
    known_mask = data.y.cpu().numpy() != 0
    layer_1 = config["layer_1"]
    layer_2 = config["layer_2"]
    if config.get("task") == 'telological':
        model = TeleosGCN(dataset=dataset,
                          data=data,
                          layer_1_size=layer_1,
                          layer_2_size=layer_2).to(config.get('device'))
    else:
        model = GCNModel(dataset=dataset,
                         data=data,
                         layer_1_size=layer_1,
                         layer_2_size=layer_2).to(config.get('device'))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=5e-5)

    loss_state, accs_state = 0, 0
    loss_no_improve, accs_no_improve = 0, 0

    for epoch in range(2001):
        model.train()
        optimizer.zero_grad()
        logits, latent_1, latent_2, latent_3 = model()
        alpha = np.logical_and(config.get('train_mask'), known_mask)
        imb_Wc = torch.bincount(config.get('data').y[alpha]).float().clamp(min=1e-10, max=1e10) / \
                 config.get('data').y[alpha].shape[0]
        weights = (1 / imb_Wc) / (sum(1 / imb_Wc))

        loss = F.cross_entropy(logits[alpha],
                               config.get('data').y[alpha],
                               weight=weights)
        _loss = loss.clone().detach().to("cpu").item()
        tune.track.log(loss=_loss)

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
        s_logits = F.softmax(input=logits[:, 1:], dim=1)

        for mask in [config.get('train_mask'), config.get('test_mask')]:
            alpha = np.logical_and(mask, known_mask)
            pred = logits[alpha].max(1)[1]

            accs.append(pred.eq(config.get('data').y[alpha]).sum().item() / alpha.sum().item())

            f1_scores.append(f1_score(y_true=config.get('data').y[alpha].to('cpu'),
                                      y_pred=pred.to('cpu'),
                                      average='macro'))

            #  FIXME: Error occurs with fold_class labelset, due to it not being possible to represent all 508
            #         classes inside each of the splits. Fix either by creating a pruning the data to consolidate
            #         the sizes of y_true and y_score right before auroc calculation.
            if config.get("task") == 'binary':
                auroc_scores.append(roc_auc_score(y_true=config.get('data').y[alpha].to('cpu').numpy(),
                                                  y_score=np.amax(s_logits[alpha].to('cpu').data.numpy(), axis=1),
                                                  average=config.get('auroc_average'),
                                                  multi_class=None))
            else:
                auroc_scores.append(roc_auc_score(y_true=config.get('data').y[alpha].to('cpu').numpy(),
                                                  y_score=s_logits[alpha].to('cpu').data.numpy(),
                                                  average=config.get('auroc_average'),
                                                  multi_class=config.get('auroc_versus')))

        train_acc, test_acc = accs
        train_f1, test_f1 = f1_scores
        train_auc, test_auc = auroc_scores
        tune.track.log(train_accuracy=train_acc, test_accuracy=test_acc)
        tune.track.log(train_f1_score=train_f1, test_f1_score=test_f1)
        tune.track.log(train_f1_score=train_auc, test_f1_score=test_auc)

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
        logging_run_name = "quaternary_experiment"
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

        return analysis.get_best_config(metric="train_f1")  # needs to return config for best model