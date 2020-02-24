import torch as t
import torch.nn.functional as F


class Trainer:
    def __init__(self, model, FIXED_PROTEIN_LENGTH, train_iterator, test_iterator, INPUT_DIM, device, optimizer,
                 train_dataset, test_dataset, N_EPOCHS):
        self.model = model.to(device)
        self.FIXED_PROTEIN_LENGTH = FIXED_PROTEIN_LENGTH
        self.train_iterator = train_iterator
        self.test_iterator = test_iterator
        self.INPUT_DIM = INPUT_DIM
        self.device = device
        self.optimizer = optimizer
        self.train_dataset_len = train_dataset
        self.test_dataset_len = test_dataset
        self.N_EPOCHS = N_EPOCHS

        self.criterion = t.nn.CrossEntropyLoss()

    def reconstruction_accuracy(self, input, output):
        """ Computes average sequence identity between input and output sequences
        """
        # if input.shape != output.shape:
        #     raise Exception("Input and output can't have different shapes")
        output_sequences = output
        input_sequences = input.argmax(axis=1)

        return ((input_sequences == output_sequences).sum(axis=1) / float(self.FIXED_PROTEIN_LENGTH)).mean()

    def __inner_iteration(self, x, training: bool):
        x = x.long().to(self.device)

        # update the gradients to zero
        if training:
            self.optimizer.zero_grad()

        # forward pass
        predicted = self.model(x)
        # predicted = predicted.view(1,predicted.shape[0], -1)
        # reconstruction loss
        # predicted = F.log_softmax(predicted, 1)
        recon_loss = self.criterion(predicted, x)

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
        with t.no_grad():
            for i, x in enumerate(self.test_iterator):
                loss, accuracy = self.__inner_iteration(x, False)
                test_loss += loss
                test_accuracy += accuracy

        return test_loss, test_accuracy / len(self.test_iterator)

    def trainer(self):
        best_training_loss = float('inf')
        patience_counter = 0
        for e in range(self.N_EPOCHS):

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

        t.save(self.model.state_dict(), f"saved_models/{self.model.name}_{date_time}")
