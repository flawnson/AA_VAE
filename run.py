'''This code is a variation of simple VAE from https://graviraja.github.io/vanillavae/
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import data
import numpy as np

from models.simple_vae import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SAVE_SNAPSHOT = True   # save snapshot of the model
DATASET_LENGTH = "small" # (small|medium|large)
FIXED_PROTEIN_LENGTH = 50 # (50|200)
BATCH_SIZE = 20         # number of data points in each batch
N_EPOCHS = 20           # times to run the model on complete data
INPUT_DIM = FIXED_PROTEIN_LENGTH * data.VOCABULARY_SIZE     # size of each input

lr = 1e-3               # learning rate

train_dataset = data.read_sequences(f"data/train_set_{DATASET_LENGTH}_{FIXED_PROTEIN_LENGTH}.json", fixed_protein_length=FIXED_PROTEIN_LENGTH, add_chemical_features=False)
test_dataset = data.read_sequences(f"data/test_set_{DATASET_LENGTH}_{FIXED_PROTEIN_LENGTH}.json", fixed_protein_length=FIXED_PROTEIN_LENGTH, add_chemical_features=False)

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# vae
model = VAE(INPUT_DIM, 20).to(device) # 20 is number of hidden dimension

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

def reconstruction_accuracy(input, output):
    """ Computes average sequence identity bewteen input and output sequences
    """
    if input.shape != output.shape:
        raise Exception("Input and output can't have different shapes")

    input_sequences = input.view(input.shape[0], FIXED_PROTEIN_LENGTH, -1)[:, :, :data.VOCABULARY_SIZE].argmax(axis=2)
    output_sequences = output.view(output.shape[0], FIXED_PROTEIN_LENGTH, -1)[:, :, :data.VOCABULARY_SIZE].argmax(axis=2)

    return ((input_sequences==output_sequences).sum(axis=1) / float(FIXED_PROTEIN_LENGTH)).mean()

def train():
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, x in enumerate(train_iterator):
        # reshape the data into [batch_size, FIXED_PROTEIN_LENGTH*VOCABULARY_SIZE]
        x = x.view(-1, INPUT_DIM)
        x = x.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        predicted = model(x)

        # reconstruction loss
        recon_loss = F.binary_cross_entropy(predicted, x, size_average=False)

        # reconstruction accuracy
        recon_accuracy = reconstruction_accuracy(predicted, x)

        # backward pass
        recon_loss.backward()
        train_loss += recon_loss.item()

        # update the weights
        optimizer.step()

    return train_loss, recon_accuracy


def test():
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    test_accuracy = 0.0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, x in enumerate(test_iterator):
            # reshape the data
            x = x.view(-1, INPUT_DIM)
            x = x.to(device)

            # forward pass
            predicted = model(x)

            # reconstruction loss
            recon_loss = F.binary_cross_entropy(predicted, x, size_average=False)

            # reconstruction accuracy
            recon_accuracy = reconstruction_accuracy(predicted, x)

            test_loss += recon_loss.item()
            test_accuracy += recon_accuracy

    return test_loss, test_accuracy/len(test_iterator)

best_test_loss = float('inf')

for e in range(N_EPOCHS):

    train_loss, train_recon_accuracy = train()
    test_loss, test_recon_accuracy = test()

    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}, Train accuracy {train_recon_accuracy*100.0:.2f}%, Test accuracy {test_recon_accuracy*100.0:.2f}%')

    if best_test_loss > test_loss:
        best_test_loss = test_loss
        patience_counter = 1
    else:
        patience_counter += 1

    if patience_counter > 3:
        break


if SAVE_SNAPSHOT:
    # save a snapshot of the model
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m_%d-%Y_%H_%M_%S")
    torch.save(model.state_dict(), f"saved_models/{model.name}_{date_time}")
