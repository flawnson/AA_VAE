import math

import torch
import torch.nn.functional as F

from utils.logger import log

inf = math.inf


def label_smoothing(inputs, epsilon):
    k = inputs.size()[-1]
    return ((1 - epsilon) * inputs) + (epsilon / k)


def calculate_gradient_stats(parameters):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_grad = max(p.grad.data.max() for p in parameters)
    min_grad = min(p.grad.data.min() for p in parameters)
    return max_grad, min_grad


def kl_loss_function(mu, logvar):
    """

    :param mu: Mean of the embedding.
    :param logvar: variance of the embedding.
    :return: KL Loss of the embeddings

     Calculates the representation loss of the embedding.
     see Appendix B from VAE paper:
     Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
     https://arxiv.org/abs/1312.6114
     0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    """
    kld: torch.Tensor = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


class Trainer:
    """
    Class that is used to run the model, runs training and testing.
    Embeddings are nto generated using this method.
    """

    def __init__(self, model, data_length, train_iterator, test_iterator, device, optimizer,
                 train_dataset_length, test_dataset_length, n_epochs, loss_function_name="smoothened",
                 vocab_size=23,
                 patience_count=1000, weights=None, model_name="default", save_best=True, length_stats=None):
        """

        :param model: The pytorch model that needs to be executed
        :param data_length: The length of a single data point
        :param train_iterator: The iterator that is used to access the training data.
        :param test_iterator: Iterator to access test data
        :param device: The device on which to run on, cuda or cpu
        :param optimizer: The optimizer that would run the gradient descent.
        :param train_dataset_length: Length of the train dataset.
        :param test_dataset_length: Length of the test dataset.
        :param n_epochs: number of epochs which the
        :param loss_function_name: The loss function that is to be used
        :param vocab_size: The different unique entities or classes in the dataset.
        :param patience_count: The number of maximum number of iterations that can run without decreasing the loss.
        :param weights: The weight of each class.
        :param model_name: The generic name of the model
        """
        log.info(f"Name: {model_name} Length:{data_length} trainDatasetLength:{train_dataset_length} "
                 f"testDataSetLength:{test_dataset_length} Epochs:{n_epochs}")
        log.info(f"LossFunction:{loss_function_name} VocabSize:{vocab_size} PatienceCount:{patience_count} ")

        loss_functions = {
            "bce": self.cross_entropy_wrapper,
            "nll": torch.nn.functional.nll_loss,
            "smoothened": self.smoothened_loss
        }
        self.model_name = model_name

        self.model = model.to(device)
        self.data_length = data_length
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.device = device
        self.optimizer = optimizer
        self.train_dataset_len = train_dataset_length
        self.test_dataset_len = test_dataset_length

        self.n_epochs = n_epochs
        self.vocab_size = vocab_size
        self.patience_count = patience_count
        self.criterion = loss_functions[loss_function_name]
        self.weights = torch.FloatTensor(weights).to(device)
        self.save_model = save_best
        self.length_stats = length_stats
        self.conf_matrix = torch.zeros([self.vocab_size, self.vocab_size]).to(self.device)

    def smoothened_loss(self, pred, actual, epsilon=0.1):

        istarget = actual.le(20)  # (1. - actual.eq(Constants.PAD).float()).contiguous().view(-1)

        actual_one_hot = torch.zeros(*pred.size(), requires_grad=True).to(self.device)
        actual_one_hot = actual_one_hot.scatter_(1, actual.unsqueeze(1).data, 1)

        actual_smoothed = label_smoothing(actual_one_hot, epsilon)

        pred_probs = F.log_softmax(pred, dim=-1)

        loss = -torch.sum(actual_smoothed * pred_probs, dim=1)
        mean_loss = torch.sum(torch.sum(loss * istarget, dim=1) / torch.sum(istarget, dim=1))

        return mean_loss

    def length_stats_based_averaging(self, predicted, actual, epsilon=0.1):

        istarget = actual.le(20)  # (1. - actual.eq(Constants.PAD).float()).contiguous().view(-1)

        actual_one_hot = torch.zeros(*predicted.size(), requires_grad=True).to(self.device)
        actual_one_hot = actual_one_hot.scatter_(1, actual.unsqueeze(1).data, 1)

        actual_smoothed = label_smoothing(actual_one_hot, epsilon)

        pred_probs = F.log_softmax(predicted, dim=-1)

        loss = -torch.sum(actual_smoothed * pred_probs, dim=1)
        mean_loss = torch.sum(torch.sum(loss * istarget, dim=1) / self.length_stats[torch.sum(istarget, dim=1)])

        return mean_loss

    def binary_cross_entropy_wrapper(self, predicted, actual):
        torch.nn.functional.binary_cross_entropy_with_logits(predicted, actual, reduction="mean",
                                                             weight=self.weights)

    def confusion_matrix(self, predicted, actual, mask):
        actual_sequence = torch.masked_select(actual, mask)
        predicted_sequence = torch.masked_select(predicted.argmax(axis=1), mask)
        self.conf_matrix[actual_sequence, predicted_sequence] += 1

    def cross_entropy_wrapper(self, predicted, actual):
        """

        :param predicted: The result returned by the model.
        :param actual: The comparison data
        :return: The reconstruction loss
        """
        return torch.nn.functional.cross_entropy(predicted, actual, reduction="mean",
                                                 weight=self.weights)

    def reconstruction_accuracy(self, predicted, actual, mask):
        """

        :param predicted: The result returned by the model
        :param actual: The comparison data
        :param mask: The mask that differentiates the actual data points from padding.
        :return: The accuracy of reconstruction

        Computes average sequence identity between input and output sequences
        """
        output_sequences = torch.masked_select(actual, mask)
        input_sequences = torch.masked_select(predicted.argmax(axis=1), mask)

        return (((input_sequences == output_sequences).sum()) / float(len(input_sequences))).item()

    # def __backprop(self):

    def __inner_iteration(self, x, training: bool, i):
        """
        This method runs a single inner iteration or passes a single batch through the model
        :param x:
        :param training:
        :param i:
        :return:
        """

        x = x.long().to(self.device)

        # update the gradients to zero
        if training:
            self.optimizer.zero_grad()
        # forward pass
        predicted, mu, var = self.model(x)
        mask = x.le(20)

        scale = mask.sum()
        recon_loss = self.criterion(predicted, x)

        kl_loss = -0.5 * torch.mean(1 + var - mu.pow(2) - var.exp())

        total_loss = kl_loss + recon_loss

        # reconstruction accuracy
        recon_accuracy = self.reconstruction_accuracy(predicted, x, mask)

        # backward pass
        if training:
            total_loss.backward()
            if (i % 1000) == 0:
                # log.debug(
                #    "KL: {} Recon:{} Total:{} Accuracy:{}".format(kl_loss.item(), recon_loss.item(), total_loss.item(),
                #                                                   recon_accuracy))
                max_grad, min_grad = calculate_gradient_stats(self.model.parameters())
                log.debug(
                    "Log10 Max gradient: {}, Min gradient: {}".format(math.log10(max_grad),
                                                                      math.log10(math.fabs(min_grad))))
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)

            self.optimizer.step()
            self.optimizer.zero_grad()
        else:
            self.confusion_matrix(predicted, x, mask)

        return kl_loss.item(), recon_loss.item(), recon_accuracy

    def train(self):
        """
        Run a training iteration over the entire dataset
        """
        # set the train mode
        self.model.train()

        # Statistics of the epoch
        train_kl_loss = 0
        train_recon_loss = 0
        recon_accuracy = 0
        valid_loop = True

        for i, x in enumerate(self.train_iterator):
            kl_loss, recon_loss, accuracy = self.__inner_iteration(x, True, i)
            total_loss = recon_loss + kl_loss
            if math.isnan(total_loss):
                log.error("Loss was nan, loop is breaking, change parameters")
                valid_loop = False
                break

            train_kl_loss += kl_loss
            train_recon_loss += recon_loss
            recon_accuracy += accuracy
            if (i % 1000) == 0:
                acc = recon_accuracy
                if i != 0:
                    acc = acc / i

                log.debug("KL: {} Recon:{} Accuracy:{}".format(train_kl_loss, train_recon_loss, acc * 100))
        return train_kl_loss, train_recon_loss, recon_accuracy / len(self.train_iterator), valid_loop

    def test(self):
        """
        Run a test or validation iteration
        :return:
        """
        # set the evaluation mode
        self.model.eval()

        # test loss for the data
        test_kl_loss = 0
        test_recon_loss = 0
        test_accuracy = 0.0

        # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
        with torch.no_grad():
            for i, x in enumerate(self.test_iterator):
                # update the gradients to zero

                kl_loss, recon_loss, accuracy = self.__inner_iteration(x, False, i)

                # backward pass
                test_kl_loss += kl_loss
                test_recon_loss += recon_loss
                test_accuracy += accuracy

        return test_kl_loss, test_recon_loss, test_accuracy / len(self.test_iterator)

    def trainer(self):
        """
        The core method in this class, it runs the full training and testing cycle for a given dataset.
        """
        best_training_loss = inf
        best_recon_accuracy = 0
        patience_counter = 0
        train_recon_accuracy = -1
        for e in range(self.n_epochs):

            train_kl_loss, train_recon_loss, train_recon_accuracy, valid = self.train()
            if not valid:
                log.error("Loop breaking as the loss was nan")
                break
            if e % 5 == 0:
                self.conf_matrix = torch.zeros([self.vocab_size, self.vocab_size]).to(self.device)

            train_recon_loss /= self.train_dataset_len
            train_kl_loss /= self.train_dataset_len
            train_loss = train_kl_loss + train_recon_loss

            info_str = f'Epoch {e}, Train Loss: KL,Recon,total: {train_kl_loss:.3f}, {train_recon_loss:.3f}, ' \
                       f'{train_loss:.3f},' \
                       f' Accuracy: {train_recon_accuracy * 100.0:.2f}%'

            if e % 5 == 0:
                test_kl_loss, test_recon_loss, test_recon_accuracy = self.test()
                test_kl_loss /= self.test_dataset_len
                test_recon_loss /= self.test_dataset_len
                info_str += f' Test Loss: KL,Recon: ({test_kl_loss:.3f}, {test_recon_loss:.3f}),' \
                            f' Accuracy: {test_recon_accuracy * 100.0:.2f}%'

            if train_recon_accuracy > 0.99:  # and test_recon_accuracy > 0.97:
                break
            if best_recon_accuracy < train_recon_accuracy:
                if self.save_model:
                    self.save_snapshot(train_recon_accuracy)

            if best_training_loss > train_loss:
                best_training_loss = train_loss
                patience_counter = 1
                if self.save_model:
                    self.save_snapshot(train_recon_accuracy)
            else:
                patience_counter += 1

            info_str += " Patience value: {}".format(patience_counter)
            log.info(info_str)

            if patience_counter > 1000:
                break
            if e % 100 == 99:
                self.save_snapshot(train_recon_accuracy)

        self.save_snapshot(train_recon_accuracy)

    def save_snapshot(self, accuracy):
        """
        Saves a snapshot of the current state of the model.
        :param accuracy:
        """
        from datetime import datetime

        now = datetime.now()

        date_time = now.strftime("%d_%m-%Y_%H_%M_%S")

        log.info(f"Writing model to saved_models/{self.model_name}_{accuracy}_{date_time}")
        torch.save(self.model.state_dict(), f"saved_models/{self.model_name}_{accuracy}_{date_time}")
