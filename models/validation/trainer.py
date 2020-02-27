import torch
import datetime
import numpy as np

from torch.nn import functional as f
from sklearn.metrics import roc_auc_score

class TrainLinear:
    def __init__(self, config, dataset, model, device):

        self.config = config
        self.dataset = dataset
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.config.get('lr'),
                                          weight_decay=self.config.get('wd'))

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(torch.randn(2, 300))

        # imb_wc = torch.bincount(self.targets, minlength=int(self.targets.max())).float().clamp(
        #     min=1e-10, max=1e10) / self.targets.shape[0]
        # weights = (1 / imb_wc) / (sum(1 / imb_wc))

        # loss = f.cross_entropy(logits, self.targets, weight=weights)
        loss = f.cross_entropy(logits, )
        loss.backward()
        self.optimizer.step()

    def test(self) -> None:
        self.model.eval()
        logits = self.model.model()

        return None

    def logs(self):
        pass

    def run(self) -> list:

        output = None
        for epoch in range(self.config.get('epochs') + 1):
            for local_batch, local_labels in self.dataset:
                start = datetime.datetime.now()
                train_output = self.train()
                test_output = self.test()
                print(f'{datetime.datetime.now() - start} since epoch-{epoch - 1}')

        return train_output, test_output

