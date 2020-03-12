import torch
import datetime
import numpy as np

from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score


class TrainLinear:
    def __init__(self, train_config, data_config, dataset, model, device):

        self.train_config = train_config
        self.dataset = dataset
        self.batch_size = data_config.get("batch_size")
        self.test_split = data_config.get("test_ratio")
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.train_config.get('lr'),
                                          weight_decay=self.train_config.get('wd'))
        if data_config.get("onehot"):
            self.imb_wc = torch.bincount(torch.argmax(torch.tensor(self.dataset.y), dim=1)).clamp(min=1e-10, max=1e10) \
                          / float(torch.tensor(self.dataset.y).shape[0])
        else:
            self.imb_wc = torch.bincount(torch.tensor(self.dataset.y)).clamp(min=1e-10, max=1e10) / float(
                torch.tensor(self.dataset.y).shape[0])
        self.weights = (1 / self.imb_wc) / (sum(1 / self.imb_wc))

    def train(self, batch, labels, mask):
        self.model.train()
        torch.set_grad_enabled(True)
        self.optimizer.zero_grad()
        logits = self.model(batch)

        loss = F.cross_entropy(logits, torch.argmax(labels, dim=1), weight=self.weights.float().to(self.device))
        loss.backward()
        self.optimizer.step()
        print(f"Loss: {loss}")

    def test(self, batch, labels, mask) -> None:
        self.model.eval()
        torch.set_grad_enabled(False)
        logits = self.model(batch)
        # s_logits = F.softmax(input=logits)
        pred = logits.max(1)[1]

        accuracy = accuracy_score(pred.cpu(), torch.argmax(labels, dim=1).cpu())
        # auroc = roc_auc_score(labels.cpu(),
        #                       s_logits,
        #                       average=self.train_config.get("auroc_average"),
        #                       multi_class=self.train_config.get("auroc_multi_class"))
        print(f"Accuracy: {accuracy}")
        # print(f"Accuracy: {auroc}")

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
                                sampler=train_sampler,
                                pin_memory=True)
        test_data = DataLoader(dataset=self.dataset,
                               batch_size=self.batch_size,
                               sampler=test_sampler,
                               pin_memory=True)

        for epoch in range(self.train_config.get('epochs') + 1):
            # FIXME: Iteration occurs over batches, not epochs, restructuring needed

            for (train_batch, train_labels, train_mask), (test_batch, test_labels, test_mask) in zip(train_data, test_data):
                train_output = self.train(train_batch.float().to(self.device),
                                          train_labels.to(self.device),
                                          train_mask.to(self.device))
                test_output = self.test(test_batch.float().to(self.device),
                                        test_labels,
                                        test_mask.to(self.device))

        return [None]  # [train_output, test_output]

