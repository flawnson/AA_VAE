import json
import torch
import datetime
import argparse
import numpy as np
import os.path as osp

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import functional as f
from sklearn.metrics import roc_auc_score


class LinearModel(torch.nn.Module):
    def __init__(self, in_size, out_size, layer_sizes, dropout=.25):
        """

        :param in_size: Model input size (embedding size)
        :param out_size: Model output size (target size)
        :param layer_sizes: List of layer sizes
        :param dropout: Dropout percentage (default: .25)
        """
        super(LinearModel, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.layer_sizes = layer_sizes
        self.dropout = dropout

    def input_layer(self) -> nn.Linear:
        return nn.Linear(self.in_size, self.layer_sizes[0])

    def hidden_layers(self, layer_in, layer_out, *args, **kwargs) -> nn.Sequential:
        """

        :param layer_in: Layer input size
        :param layer_out: Layer output size
        :param args: Linear arguments (bias, etc.)
        :param kwargs: Linear arguments (bias, etc.)
        :return:
        """
        return nn.Sequential(
            nn.Linear(layer_in, layer_out, *args, **kwargs),
            nn.BatchNorm1d(layer_out),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

    def output_layer(self) -> nn.Linear:
        return nn.Linear(self.layer_sizes[-1], self.out_size)

    def model(self) -> nn.Sequential:
        first_layer = self.input_layer()
        hidden_layers = [self.hidden_layers(in_size, out_size, bias=True)
                         for in_size, out_size in zip(self.layer_sizes, self.layer_sizes[1:])]
        final_layer = self.output_layer()

        full_model = nn.Sequential(first_layer, *hidden_layers, final_layer)

        return full_model


class TrainLinear:
    def __init__(self, config, targets, model, device):

        self.config = config.get('train_config')
        self.targets = targets
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=5e-5)

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model()

        imb_wc = torch.bincount(self.targets, minlength=int(self.targets.max())).float().clamp(
            min=1e-10, max=1e10) / self.targets.shape[0]
        weights = (1 / imb_wc) / (sum(1 / imb_wc))

        loss = f.cross_entropy(logits, self.targets, weight=weights)
        loss.backward()
        self.optimizer.step()

    def test(self) -> list:
        self.model.eval()
        logits = self.model()
        accs, auroc_scores, f1_scores = [], [], []
        s_logits = f.softmax(input=logits, dim=1)  # To account for the unknown class

        for mask in [self.train_mask, self.val_mask, self.test_mask]:
            # pred = logits.max(1)[1]

            # Add calculation for accuracy
            # Add calculation for f1_score

            if len(np.unique(self.targets)) > 2:
                average = self.config.get('auroc_average')
                multi_class = self.config.get('auroc_multi_class')
            else:
                average = None
                multi_class = None

            auroc = roc_auc_score(y_true=self.targets[mask].to('cpu').numpy(),
                                  y_score=np.amax(s_logits[mask].to('cpu').data.numpy(), axis=1),
                                  average=average,
                                  multi_class=multi_class)

            # accs.append(acc)
            # f1_scores.append(f1)
            auroc_scores.append(auroc)

        return [accs, f1_scores, np.asarray(auroc_scores)]

    def logs(self):
        pass

    def run(self) -> list:

        output = None
        for epoch in range(self.config.get('epochs') + 1):
            start = datetime.datetime.now()
            self.train()
            output = self.test()
            print(f'{datetime.datetime.now() - start} since epoch-{epoch - 1}')

        return output


class EmbeddingData(Dataset):
    def __init__(self, embeddings, targets):
        """

        :param embeddings: torch tensor of embeddings
        :param targets: torch tensor of corresponding targets
        """
        self.x = embeddings
        self.y = targets

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    # print(LinearModel(1, 10, [22, 33, 44, 55, 66]).model())
    path = osp.join('simple-vae', 'configs')  # Implicitly used to get config file?
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-f", "--config", help="json config file", type=str)
    args = parser.parse_args()

    json_file = open(args.config)
    json_config = json.load(json_file)

    data_config = json_config.get('data_config')
    data = EmbeddingData(_, _)
    dataset = DataLoader(dataset=data, batch_size=data_config.get('batch_size'))

    model_config = json_config.get('model_config')
    model = LinearModel(model_config.get('in_size'),
                        model_config.get('out_size'),
                        model_config.get('layer_sizes'),
                        model_config.get('dropout'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TrainLinear(json_config, dataset, model, device).run()
