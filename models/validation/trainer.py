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

    def train(self, batch, labels) -> torch.Tensor:
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
        return loss

    @torch.no_grad()
    def score(self, batchset: list, labelset: list) -> float:
        self.model.eval()
        torch.set_grad_enabled(False)

        logits = self.model(batchset)
        pred = logits.max(1)[1]

        accuracy = accuracy_score(pred.cpu(), torch.argmax(labelset, dim=1).cpu())
        output_mask = np.isin(list(range(list(self.model.children())[-1][-1].out_features)),
                              np.unique(torch.argmax(labelset, dim=1)))
        test = np.apply_along_axis(func1d=lambda arr: arr[output_mask], axis=1,
                                   arr=logits.to('cpu').data.numpy())
        s_logits = F.softmax(input=torch.from_numpy(test), dim=1)
        auroc = roc_auc_score(torch.argmax(labelset, dim=1).cpu(),
                              s_logits,
                              average=self.run_config.get("auroc_average"),
                              multi_class=self.run_config.get("auroc_multi_class"))
        f1 = f1_score(torch.argmax(labelset, dim=1).cpu(),
                      pred.cpu(),
                      average='macro')

        return [accuracy, auroc, f1]

    def logs(self):
        # TODO: Need to implement logging of metrics and scores
        pass

    @staticmethod
    def average_batch_scores(scores_list: list) -> list:
        reformated_scores = [list(i) for i in zip(*scores_list)]
        avg_scores = [sum(scores) / len(scores) for scores in reformated_scores]

        return avg_scores

    def run(self) -> None:
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

        epoch_num = 0
        for epoch in range(self.run_config.get('epochs') + 1):
            print("-"*20 + f"Epoch number: {epoch_num}" + "-"*20)
            epoch_num += 1

            iteration = 0
            try:
                for (train_batch, train_labels), (test_batch, test_labels) in zip(train_data, test_data):
                    print("-"*15 + f"Iteration number: {iteration}" + "-"*15)
                    iteration += 1

                    train_loss = self.train(train_batch.float().to(self.device),
                                                   train_labels.to(self.device))
                    train_scores = self.score(train_batch.float().to(self.device), train_labels)
                    test_scores = self.score(test_batch.float().to(self.device), test_labels)

                    print(f"Train Loss: {train_loss:.3f}")
                    print(f"Train Accuracy: {train_scores[0]:.3f}, Test Accuracy: {test_scores[0]:.3f}")
                    print(f"Train auroc: {train_scores[1]:.3f} Test auroc: {test_scores[1]:.3f}")
                    print(f"Train F1: {train_scores[2]:.3f} Test F1: {test_scores[2]:.3f}")
            except ValueError:
                print("Ran out of data to continue batched training, continuing run")

            avg_train_scores = self.average_batch_scores(train_scores)
            avg_test_scores = self.average_batch_scores(test_scores)
            print(f"Avg Train Accuracy: {avg_train_scores[0]:.3f}, Avg Test Accuracy: {avg_test_scores[0]:.3f}")
            print(f"Avg Train auroc: {avg_train_scores[1]:.3f}, Avg Test auroc: {avg_test_scores[1]:.3f}")
            print(f"Avg Train F1: {avg_train_scores[2]:.3f}, Avg Test F1: {avg_test_scores[2]:.3f}")

        return None

