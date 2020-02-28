import torch
import datetime
import numpy as np

from torch.nn import functional as f
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score

class TrainLinear:
    def __init__(self, train_config, data_donfig, dataset, model, device):

        self.train_config = train_config
        self.dataset = dataset
        self.batch_size = data_donfig.get("batch_size")
        self.test_split = data_donfig.get("test_ratio")
        self.device = device
        self.model = model.to(self.device)
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

        loss = f.cross_entropy(logits, labels)  # weight=weights)
        loss.backward()
        self.optimizer.step()
        print(f"Loss: {loss}")

    def test(self, batch, labels) -> None:
        self.model.eval()
        torch.set_grad_enabled(False)
        logits = self.model(batch)
        pred = logits.max(1)[1]

        accuracy = accuracy_score(pred.cpu(), labels.cpu())
        print(f"Accuracy: {accuracy}")

        return None

    def logs(self):
        pass

    def run(self) -> list:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.test_split * dataset_size))
        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_data = DataLoader(dataset=self.dataset,
                                batch_size=self.batch_size,
                                sampler=train_sampler)
        test_data = DataLoader(dataset=self.dataset,
                               batch_size=self.batch_size,
                               sampler=test_sampler)

        for epoch in range(self.train_config.get('epochs') + 1):
            # FIXME: Iteration occurs over batches, not epochs, restructuring needed
            start = datetime.datetime.now()

            for train_batch, train_labels in train_data:
                train_output = self.train(train_batch.float().to(self.device), train_labels.to(self.device))
                print(f'{datetime.datetime.now() - start} since epoch-{epoch}')

            for test_batch, test_labels in test_data:
                test_output = self.test(test_batch.float().to(self.device), test_labels)

        return [None]  # [train_output, test_output]

