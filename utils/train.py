import torch


class Trainer:
    def __init__(self, model, data_length, train_iterator, test_iterator, input_dim, device, optimizer,
                 train_dataset, test_dataset, n_epochs, loss_function_name="bce",
                 vocab_size=23,
                 patience_count=1000):

        loss_function = {
            "bce": torch.nn.functional.cross_entropy
        }

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

    def reconstruction_accuracy(self, predicted, actual, current_index, training: bool):
        """ Computes average sequence identity between input and output sequences
        """
        # if input.shape != output.shape:
        #     raise Exception("Input and output can't have different shapes")
        mask = actual.le(20)
        output_sequences = torch.masked_select(actual, mask)
        input_sequences = torch.masked_select(predicted.argmax(axis=1), mask)

        return ((input_sequences == output_sequences).sum()) / float(len(input_sequences))

    def __inner_iteration(self, x, training: bool, i):
        x = x.long().to(self.device)

        # update the gradients to zero
        if training:
            self.optimizer.zero_grad()

        # forward pass
        predicted = self.model(x)

        recon_loss = self.criterion(predicted, x, ignore_index=22)

        loss = recon_loss.item()
        # reconstruction accuracy
        # TODO this needs to change once new features are added into the vector
        recon_accuracy = self.reconstruction_accuracy(predicted, x, i * x.shape[0], training)

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
            print(
                f'Epoch {e}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Train accuracy {train_recon_accuracy * 100.0:.2f}%, Test accuracy {test_recon_accuracy * 100.0:.2f}%')

            if train_recon_accuracy > 0.87 and test_recon_accuracy > 0.87:
                break

            if best_training_loss > train_loss:
                best_training_loss = train_loss
                patience_counter = 1
            else:
                patience_counter += 1

            print("Patience value at {}".format(patience_counter))
            if patience_counter > 100:
                break

        self.save_snapshot()

    def save_snapshot(self):
        from datetime import datetime

        now = datetime.now()

        date_time = now.strftime("%m_%d-%Y_%H_%M_%S")

        torch.save(self.model.state_dict(), f"saved_models/{self.model.name}_{date_time}")
