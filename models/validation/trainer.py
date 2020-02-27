import torch
import datetime
import numpy as np

from torch.nn import functional as f
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import roc_auc_score

class TrainLinear:
    def __init__(self, train_config, data_donfig, dataset, model, device):

        self.train_config = train_config
        self.dataset = dataset
        self.batch_size = data_donfig.get("batch_size")
        self.test_split = data_donfig.get("test_ratio")
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.train_config.get('lr'),
                                          weight_decay=self.train_config.get('wd'))

    def train(self, batch, labels):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(batch)

        # imb_wc = torch.bincount(self.targets, minlength=int(self.targets.max())).float().clamp(
        #     min=1e-10, max=1e10) / self.targets.shape[0]
        # weights = (1 / imb_wc) / (sum(1 / imb_wc))

        # loss = f.cross_entropy(logits, self.targets, weight=weights)
        loss = f.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()

    def test(self, batch, labels) -> None:
        self.model.eval()
        logits = self.model.model()

        return None

    def logs(self):
        pass

    def run(self) -> list:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.test_split * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_data = DataLoader(dataset=self.dataset,
                                shuffle=True,
                                batch_size=self.batch_size,
                                sampler=train_sampler)
        test_data = DataLoader(dataset=self.dataset,
                               shuffle=True,
                               batch_size=self.batch_size,
                               sampler=valid_sampler)

        for epoch in range(self.config.get('epochs') + 1):
            start = datetime.datetime.now()

            for train_batch, train_labels in train_data:
                train_output = self.train(train_batch.float(), train_labels)

            while torch.set_grad_enabled(False):
                for test_batch, test_labels in test_data:
                    test_output = self.test(test_batch, test_labels)

            print(f'{datetime.datetime.now() - start} since epoch-{epoch - 1}')

        return [train_output, test_output]

