import torch
import torch.nn.functional as F

from utils.model_factory import create_model


def trainer(config, model_config):
    train, model, optimiser, device = create_model(config, model_config)
    train.trainer()

    SAVE_SNAPSHOT = False

    if SAVE_SNAPSHOT:
        train.save_snapshot()


class Trainer:
    def __init__(self, model, data_length, train_iterator, test_iterator, input_dim, device, optimizer,
                 train_dataset, test_dataset, n_epochs):
        self.model = model.to(device)
        self.data_length = data_length
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.input_dim = input_dim
        self.device = device
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.n_epochs = n_epochs

    def reconstruction_accuracy(self, input, output):
        """ Computes average sequence identity between input and output sequences
        """
        if input.shape != output.shape:
            raise Exception("Input and output can't have different shapes")
        input_sequences = input.transpose(1, 2).view(input.shape[0], self.data_length, -1)[:, :, :23] \
            .argmax(axis=2)
        output_sequences = output.transpose(1, 2).view(output.shape[0], self.data_length, -1)[:, :, :23] \
            .argmax(axis=2)

        return ((input_sequences == output_sequences).sum(axis=1) / float(self.data_length)).mean()

    def __inner_iteration(self, x, training: bool):
        x = x.transpose(1, 2).to(self.device)

        # update the gradients to zero
        if training:
            self.optimizer.zero_grad()

        # forward pass
        predicted = self.model(x)[0]

        # reconstruction loss
        recon_loss = F.binary_cross_entropy(predicted, x, size_average=False)

        loss = recon_loss.item()
        # reconstruction accuracy
        # TODO this needs to change once new features are added into the vector
        recon_accuracy = self.reconstruction_accuracy(predicted, x)

        # backward pass
        if training:
            recon_loss.backward()
            self.optimizer.step()

        # update the weights

        return loss, recon_accuracy

    def train(self):
        # set the train mode
        self.model.train()

        # loss of the epoch
        train_loss = 0

        recon_accuracy = 0

        for i, x in enumerate(self.train_iterator):
            # reshape the data into [batch_size, FIXED_PROTEIN_LENGTH*23]
            loss, accuracy = self.__inner_iteration(x, True)
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
                loss, accuracy = self.__inner_iteration(x, False)
                test_loss += loss
                test_accuracy += accuracy

        return test_loss, test_accuracy / len(self.test_iterator)

    def trainer(self):
        best_training_loss = float('inf')
        patience_counter = 0
        for e in range(self.n_epochs):

            train_loss, train_recon_accuracy = self.train()
            test_loss, test_recon_accuracy = self.test()

            train_loss /= len(self.train_dataset)
            test_loss /= len(self.test_dataset)
            print(
                f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}, Train accuracy {train_recon_accuracy * 100.0:.2f}%, Test accuracy {test_recon_accuracy * 100.0:.2f}%')

            if best_training_loss > train_loss:
                best_training_loss = train_loss
                patience_counter = 1
            else:
                patience_counter += 1

            print("Patience value at {}".format(patience_counter))
            if patience_counter > 100:
                break

    def save_snapshot(self):
        from datetime import datetime

        now = datetime.now()

        date_time = now.strftime("%m_%d-%Y_%H_%M_%S")

        torch.save(self.model.state_dict(), f"saved_models/{self.model.name}_{date_time}")


