import torch


def loss_function(recon_x, mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_x + KLD


class Trainer:
    def __init__(self, model, data_length, train_iterator, test_iterator, input_dim, device, optimizer,
                 train_dataset, test_dataset, n_epochs, loss_function_name="bce",
                 vocab_size=23,
                 patience_count=1000):

        loss_function = {
            "bce": torch.nn.functional.cross_entropy,
            "nll": torch.nn.functional.nll_loss
        }

        self.weights = torch.Tensor([0.6428, 3.3950, 0.7450, 0.9234, 0.7459, 1.2758, 0.8553, 2.2021, 0.8400,
                        2.1896, 0.5179, 1.1491, 1.2546, 1.0246, 0.6837, 0.9261, 0.9258, 4.5166,
                        0.7591, 1.6835, 0, 0]).to(device)

        self.model = model.to(device)
        self.data_length = data_length
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.input_dim = input_dim
        self.device = device
        self.optimizer = optimizer
        self.train_dataset_len = train_dataset
        self.test_dataset_len = test_dataset

        self.n_epochs = n_epochs
        self.vocab_size = vocab_size
        self.patience_count = patience_count
        self.criterion = loss_function[loss_function_name]

    def reconstruction_accuracy(self, predicted, actual):
        """ Computes average sequence identity between input and output sequences
        """
        # if input.shape != output.shape:
        #     raise Exception("Input and output can't have different shapes")
        mask = actual.le(21)
        output_sequences = torch.masked_select(actual, mask)
        input_sequences = torch.masked_select(predicted.argmax(axis=1), mask)

        return (((input_sequences == output_sequences).sum()) / float(len(input_sequences))).item()

    def __inner_iteration(self, x, training: bool, i):
        x = x.long().to(self.device)

        # update the gradients to zero
        if training:
            self.optimizer.zero_grad()

        # forward pass
        predicted, mu, var = self.model(x)

        recon_loss = loss_function(self.criterion(predicted, x, weight=self.weights, ignore_index=22), mu, var)

        loss = recon_loss.item()
        # reconstruction accuracy
        # TODO this needs to change once new features are added into the vector
        recon_accuracy = self.reconstruction_accuracy(predicted, x)

        # backward pass
        if training:
            recon_loss.backward()
            self.optimizer.step()

        return loss, recon_accuracy

    def train(self):
        # set the train mode
        self.model.train()

        # loss of the epoch
        train_loss = 0

        recon_accuracy = 0

        for i, x in enumerate(self.train_iterator):
            # reshape the data into [batch_size, FIXED_PROTEIN_LENGTH*23]
            loss, accuracy = self.__inner_iteration(x, True, i)
            train_loss += loss
            recon_accuracy += accuracy

        return train_loss, recon_accuracy / len(self.train_iterator)

    def test(self):
        # set the evaluation mode
        self.model.eval()

        # test loss for the data
        test_loss = 0

        test_accuracy = 0.0

        # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
        with torch.no_grad():
            for i, x in enumerate(self.test_iterator):
                # update the gradients to zero

                loss, accuracy = self.__inner_iteration(x, False, i)

                # backward pass
                test_loss += loss
                test_accuracy += accuracy

        return test_loss, test_accuracy / len(self.test_iterator)

    def trainer(self):
        best_training_loss = float('inf')
        patience_counter = 0
        for e in range(self.n_epochs):

            train_loss, train_recon_accuracy = self.train()
            test_loss, test_recon_accuracy = self.test()

            train_loss /= self.train_dataset_len
            test_loss /= self.test_dataset_len
            print(f'Epoch {e}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Train accuracy ')
            print(f'{train_recon_accuracy * 100.0:.2f}%, Test accuracy {test_recon_accuracy * 100.0:.2f}%')

            if train_recon_accuracy > 0.97 and test_recon_accuracy > 0.97:
                break

            if best_training_loss > train_loss:
                best_training_loss = train_loss
                patience_counter = 1
            else:
                patience_counter += 1

            print("Patience value at {}".format(patience_counter))
            if patience_counter > 500:
                break
            if e % 100 == 0:
                self.save_snapshot()

        self.save_snapshot()

    def save_snapshot(self):
        from datetime import datetime

        now = datetime.now()

        date_time = now.strftime("%m_%d-%Y_%H_%M_%S")

        torch.save(self.model.state_dict(), f"saved_models/{self.model.name}_{date_time}")
