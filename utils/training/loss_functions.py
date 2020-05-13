import torch
from torch.nn import functional as F

from utils.training.common import label_smoothing


class LossFunctions:
    def __init__(self, device, weights, length_stats, vocab_size=21):
        self.length_stats = None if length_stats is None else torch.tensor(length_stats).to(device)
        self.device = device
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device)
        self.vocab_size = vocab_size

    def smoothened_loss(self, pred, actual, epsilon=0.1):
        istarget = actual.le(20)  # (1. - actual.eq(Constants.PAD).float()).contiguous().view(-1)
        istarget = istarget.float()
        target_count = torch.sum(istarget, dim=1)
        target_count[target_count == 0] = 1
        actual_one_hot = torch.zeros(*pred.size(), requires_grad=False).to(self.device)
        actual_one_hot = actual_one_hot.scatter_(1, actual.unsqueeze(1).data, 1)
        # if self.weights is not None:
        #     actual_one_hot = self.weights[actual].view(actual.shape[0], 1, -1) * actual_one_hot

        k = actual.shape[1]
        actual_one_hot = ((1 - epsilon) * actual_one_hot) + (epsilon / k)
        pred_probs = F.log_softmax(pred, dim=1)

        # target_count = torch.sum(istarget)
        loss = -torch.sum(actual_one_hot * pred_probs, dim=1)
        mean_loss = torch.mean(torch.sum(loss * istarget, dim=1) / target_count)
        del loss, istarget, target_count, actual_one_hot
        return mean_loss

    def length_stats_based_averaging(self, predicted, actual, epsilon=0.1):
        istarget = actual.le(20)  # (1. - actual.eq(Constants.PAD).float()).contiguous().view(-1)

        actual_one_hot = torch.zeros(*predicted.size(), requires_grad=True).to(self.device)
        actual_one_hot = actual_one_hot.scatter_(1, actual.unsqueeze(1).data, 1)

        actual_smoothed = label_smoothing(actual_one_hot, epsilon)

        pred_probs = F.log_softmax(predicted, dim=-1)

        weights = self.length_stats[torch.sum(istarget, dim=1)]
        loss = -torch.sum(actual_smoothed * pred_probs, dim=1)
        mean_loss = torch.sum(torch.sum(loss * istarget, dim=1) / weights)

        return mean_loss

    def binary_cross_entropy_wrapper(self, predicted, actual):
        torch.nn.functional.binary_cross_entropy_with_logits(predicted, actual, reduction="mean",
                                                             weight=self.weights)

    def cross_entropy_wrapper(self, predicted, actual):
        """

        :param predicted: The result returned by the model.
        :param actual: The comparison data
        :return: The reconstruction loss
        """
        return torch.nn.functional.cross_entropy(predicted, actual, reduction="mean",
                                                 weight=self.weights)
