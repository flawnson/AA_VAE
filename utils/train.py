import math

import torch

from utils.logger import log

inf = math.inf


def calculate_gradient_stats(parameters):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

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
    KLD: torch.Tensor = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


class Trainer:
    """
    Class that is used to run the model, runs training and testing.
    Embeddings are nto generated using this method.
    """

    def __init__(self, model, data_length, train_iterator, test_iterator, device, optimizer,
                 train_dataset_length, test_dataset_length, n_epochs, loss_function_name="bce",
                 vocab_size=23,
                 patience_count=1000, weights=None, model_name="default", freq=1):
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
        :param freq: The frequency of running backward propagation
        """
        log.debug(f"Name: {model_name} Length:{data_length} trainDatasetLength:{train_dataset_length} "
                  f"testDataSetLength:{test_dataset_length} Epochs:{n_epochs}")
        log.debug(f"LossFunction:{loss_function_name} VocabSize:{vocab_size} PatienceCount:{patience_count} "
                  f"Frequency:{freq}")

        loss_functions = {
            "bce": self.cross_entropy_wrapper,
            "nll": torch.nn.functional.nll_loss
        }
        self.model_name = model_name
        self.backprop_freq = freq

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
        self.total_loss_score: torch.DoubleTensor = torch.DoubleTensor([[0]]).to(device)

    def cross_entropy_wrapper(self, predicted, actual, count):
        """

        :param predicted: The result returned by the model.
        :param actual: The comparison data
        :param count: The number of relevant datapoints in the batch.
        :return: The reconstruction loss
        """
        return torch.nn.functional.cross_entropy(predicted, actual, reduction="none",
                                                 weight=self.weights).sum()

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

        # forward pass
        predicted, mu, var = self.model(x)
        mask = x.le(20)

        scale = mask.sum()
        recon_loss = self.criterion(predicted, x, scale)

        kl_loss = kl_loss_function(mu, var)
        total_loss = kl_loss + recon_loss
        # if training:
        #     self.total_loss_score += total_loss

        # reconstruction accuracy
        recon_accuracy = self.reconstruction_accuracy(predicted, x, mask)
        log.debug("{} {} {}".format(kl_loss.item(), recon_loss.item(), total_loss.item()))
        # backward pass
        if training:
            # if i % self.backprop_freq == 0:
            #     total_loss = self.total_loss_score
                total_loss.backward()
                max_grad, min_grad = calculate_gradient_stats(self.model.parameters())
                log.debug(
                    "Log10 Max gradient: {}, Min gradient: {} Total loss: {}".format(math.log10(max_grad),
                                                                                     math.log10(math.fabs(min_grad)),
                                                                                     total_loss.item()))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 500)

                self.optimizer.step()
                self.optimizer.zero_grad()
                # self.total_loss_score = 0

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
            if total_loss == math.nan:
                log.error("Loss was nan, loop is breaking, change parameters")
                valid_loop = False
                break

            train_kl_loss += kl_loss
            train_recon_loss += recon_loss
            recon_accuracy += accuracy

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
        patience_counter = 0
        train_recon_accuracy = -1
        for e in range(self.n_epochs):

            train_kl_loss, train_recon_loss, train_recon_accuracy, valid = self.train()
            if not valid:
                log.error("Loop breaking as the loss was nan")
                break
            test_kl_loss, test_recon_loss, test_recon_accuracy = self.test()
            train_recon_loss /= self.train_dataset_len
            test_recon_loss /= self.test_dataset_len
            train_kl_loss /= self.train_dataset_len
            test_kl_loss /= self.test_dataset_len

            if train_recon_accuracy > 0.97 and test_recon_accuracy > 0.97:
                break
            train_loss = train_kl_loss + train_recon_loss
            if best_training_loss > train_loss:
                best_training_loss = train_loss
                patience_counter = 1
                self.save_snapshot(train_recon_accuracy)
            else:
                patience_counter += 1

            info_str = f'Epoch {e}, Train Loss: KL,Recon,total: {train_kl_loss:.3f}, {train_recon_loss:.3f}, ' \
                       f'{train_loss:.3f},' \
                       f' Accuracy: {train_recon_accuracy * 100.0:.2f}%'
            info_str += f' Test Loss: KL,Recon: ({test_kl_loss:.3f}, {test_recon_loss:.3f}),' \
                        f' Accuracy: {test_recon_accuracy * 100.0:.2f}%'
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

        date_time = now.strftime("%m_%d-%Y_%H_%M_%S")

        torch.save(self.model.state_dict(), f"saved_models/{self.model_name}_{accuracy}_{date_time}")
