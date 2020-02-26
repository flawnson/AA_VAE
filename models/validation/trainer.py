import torch
import datetime
import numpy as np

from torch.nn import functional as f
from sklearn.metrics import roc_auc_score

class TrainLinear:
    def __init__(self, config, targets, model, device):

        self.config = config
        self.targets = targets
        self.model = model
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.config.get('lr'),
                                          weight_decay=self.config.get('wd')).step()

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model()

        imb_wc = torch.bincount(self.targets, minlength=int(self.targets.max())).float().clamp(
            min=1e-10, max=1e10) / self.targets.shape[0]
        weights = (1 / imb_wc) / (sum(1 / imb_wc))

        loss = f.cross_entropy(logits, self.targets, weight=weights)
        loss.backward()

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

