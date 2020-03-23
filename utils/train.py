import torch

from utils.logger import log


def ramp_function(index, length, depth, max_height):
    width = length + depth
    current_epoch = int(max_height / width)
    delta = float(max_height) / float(width)


def kl_loss_function(mu, logvar, scale: float):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    if scale > 1:
        scale = 1
    KLD: torch.Tensor = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD * scale


class Trainer:
    def __init__(self, model, data_length, train_iterator, test_iterator, device, optimizer,
                 train_dataset, test_dataset, n_epochs, loss_function_name="bce",
                 vocab_size=23,
                 patience_count=1000, weights=None, model_name="default", freq=1):

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
        self.train_dataset_len = train_dataset
        self.test_dataset_len = test_dataset

        self.n_epochs = n_epochs
        self.vocab_size = vocab_size
        self.patience_count = patience_count
        self.criterion = loss_functions[loss_function_name]
        self.weights = torch.FloatTensor(weights).to(device)

    def cross_entropy_wrapper(self, predicted, actual, count):
        return torch.nn.functional.cross_entropy(predicted, actual, reduction="none",
                                                 weight=self.weights).sum() / count

    def reconstruction_accuracy(self, predicted, actual, mask):
        """ Computes average sequence identity between input and output sequences
        """
        output_sequences = torch.masked_select(actual, mask)
        input_sequences = torch.masked_select(predicted.argmax(axis=1), mask)

        return (((input_sequences == output_sequences).sum()) / float(len(input_sequences))).item()

    def __inner_iteration(self, x, training: bool, i):
        x = x.long().to(self.device)

        # update the gradients to zero

        # forward pass
        predicted, mu, var = self.model(x)
        mask = x.le(20)

        scale = mask.sum()
        recon_loss = self.criterion(predicted, x, scale)

        kl_loss = kl_loss_function(mu, var, 1)
        total_loss = kl_loss + recon_loss
        # recon_loss = total_loss_function(recon_loss, mu, var, float(scale) / mask.numel())

        # reconstruction accuracy
        # TODO this needs to change once new features are added into the vector
        recon_accuracy = self.reconstruction_accuracy(predicted, x, mask)

        # backward pass
        if training:
            if i % self.backprop_freq == 0:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                self.optimizer.zero_grad()

        return kl_loss.item(), recon_loss.item(), recon_accuracy

    def train(self):
        # set the train mode
        self.model.train()

        # loss of the epoch
        train_kl_loss = 0
        train_recon_loss = 0

        recon_accuracy = 0

        for i, x in enumerate(self.train_iterator):
            # reshape the data into [batch_size, FIXED_PROTEIN_LENGTH*23]
            kl_loss, recon_loss, accuracy = self.__inner_iteration(x, True, i)
            train_kl_loss += kl_loss
            train_recon_loss += recon_loss
            recon_accuracy += accuracy

        return train_kl_loss, train_recon_loss, recon_accuracy / len(self.train_iterator)

    def test(self):
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
        best_training_loss = float('inf')
        patience_counter = 0
        train_recon_accuracy = -1
        for e in range(self.n_epochs):

            train_kl_loss, train_recon_loss, train_recon_accuracy = self.train()
            test_kl_loss, test_recon_loss, test_recon_accuracy = self.test()
            train_recon_loss /= self.train_dataset_len
            test_recon_loss /= self.test_dataset_len
            train_kl_loss /= self.train_dataset_len
            test_kl_loss /= self.test_dataset_len
            info_str = f'Epoch {e}, Train Loss: KL: {train_kl_loss:.8f}, Recon: {train_recon_loss:.8f}' \
                       f', Accuracy: {train_recon_accuracy * 100.0:.2f}% '
            info_str += f'Test Loss: KL: {test_kl_loss:.8f}, Recon: {test_recon_loss:.8f}, ' \
                        f' Accuracy {test_recon_accuracy * 100.0:.2f}%'
            log.info(info_str)

            if train_recon_accuracy > 0.97 and test_recon_accuracy > 0.97:
                break
            train_loss = train_kl_loss + train_recon_loss
            if best_training_loss > train_loss:
                best_training_loss = train_loss
                patience_counter = 1
                self.save_snapshot(train_recon_accuracy)
            else:
                patience_counter += 1

            log.info("Patience value at {}".format(patience_counter))
            if patience_counter > 1000:
                break
            if e % 100 == 99:
                self.save_snapshot(train_recon_accuracy)

        self.save_snapshot(train_recon_accuracy)

    def save_snapshot(self, accuracy):
        from datetime import datetime

        now = datetime.now()

        date_time = now.strftime("%m_%d-%Y_%H_%M_%S")

        torch.save(self.model.state_dict(), f"saved_models/{self.model_name}_{accuracy}_{date_time}")
