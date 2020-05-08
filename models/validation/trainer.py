import torch
import datetime
import numpy as np

from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


class TrainLinear:
    def __init__(self, run_config, data_config, dataset, model, device):

        self.run_config = run_config
        self.dataset = dataset
        self.batch_size = data_config.get("batch_size")
        self.test_split = data_config.get("test_ratio")
        self.onehot = data_config.get("onehot")
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.run_config.get('lr'),
                                          weight_decay=self.run_config.get('wd'))

        # The option to use integer labels in available but not recommended as classes have no relation to each other
        if self.onehot:
            self.imb_wc = torch.bincount(torch.argmax(torch.tensor(self.dataset.y), dim=1)).clamp(min=1e-10, max=1e10) \
                          / float(torch.tensor(self.dataset.y).shape[0])
            self.weights = (1 / self.imb_wc) / (sum(1 / self.imb_wc)).to(device)
        else:
            self.imb_wc = torch.bincount(torch.tensor(np.asarray(self.dataset.y))).clamp(min=1e-10, max=1e10) / float(
                torch.tensor(np.asarray(self.dataset.y)).shape[0])
            self.weights = (1 / self.imb_wc[1:]) / (sum(1 / self.imb_wc[1:])).to(device)

    def train(self, batch, labels):
        # TODO: Need to apply known mask to ignore unknowns (supervised task)
        self.model.train()
        torch.set_grad_enabled(True)
        self.optimizer.zero_grad()
        logits = self.model(batch)
        if self.onehot:
            labels = torch.argmax(labels, dim=1)
        loss = F.cross_entropy(logits, labels, weight=self.weights.float())
        loss.backward()
        self.optimizer.step()
        print(f"Loss: {loss:.4f}")
        return loss

    def test(self, batch, labels) -> float:
        self.model.eval()
        torch.set_grad_enabled(False)
        logits = self.model(batch)
        pred = logits.max(1)[1]

        accuracy = accuracy_score(pred.cpu(), torch.argmax(labels, dim=1).cpu())
        output_mask = np.isin(list(range(list(self.model.children())[-1][-1].out_features)),
                              np.unique(torch.argmax(labels, dim=1)))
        test = np.apply_along_axis(func1d=lambda arr: arr[output_mask], axis=1,
                                   arr=logits.to('cpu').data.numpy())
        s_logits = F.softmax(input=torch.from_numpy(test), dim=1)
        auroc = roc_auc_score(torch.argmax(labels, dim=1).cpu(),
                              s_logits,
                              average=self.run_config.get("auroc_average"),
                              multi_class=self.run_config.get("auroc_multi_class"))
        f1 = f1_score(torch.argmax(labels, dim=1).cpu(),
                      pred.cpu(),
                      average='macro')
        print(f"Accuracy: {accuracy:.3f}")
        print(f"auroc: {auroc:.3f}")
        print(f"F1: {f1:.3f}")

        return accuracy

    def logs(self):
        # TODO: Need to implement logging of metrics and scores
        pass

    def run(self) -> tuple:
        indices = list(range(len(self.dataset)))
        split = int(np.floor(self.test_split * len(self.dataset)))
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

        train_record, test_record = [], []

        epoch_num = 0
        for epoch in range(self.run_config.get('epochs') + 1):
            print("-"*10 + f"Epoch number: {epoch_num}" + "-"*10)
            epoch_num += 1

            batch_num = 0
            for (train_batch, train_labels), (test_batch, test_labels) in zip(train_data, test_data):
                print(f"Batch number: {batch_num}")
                batch_num += 1

                train_record.append(self.train(train_batch.float().to(self.device),
                                               train_labels.to(self.device)))
                train_record.append(self.test(test_batch.float().to(self.device),
                                              test_labels))

        return train_record, test_record

