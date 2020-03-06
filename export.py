""" Script used to export embeddings of proteins
"""
import argparse

import pandas as pd

import utils.model_factory as model_factory

"""
Load the saved model
"""

import collections
import json

import torch
from torch.utils.data import DataLoader

"""
Valid amino acids
X - unknown
U - Selenocysteine
0 - padding for fixed length encoders
"""
amino_acids = "UCSTPAGNDEQHRKMILVFYWX0"
VOCABULARY_SIZE = len(amino_acids)

amino_acids_to_byte_map = {r: amino_acids.index(r) for r in amino_acids}
amino_acids_set = {r for r in amino_acids}


def load_data(_config, max_length=-1):
    data_length = _config["protein_length"]
    batch_size = _config["batch_size"]  # number of data points in each batch
    train_dataset_name = _config["train_dataset_name"]
    test_dataset_name = _config["test_dataset_name"]

    print(f"Loading the sequence for train data: {train_dataset_name} and test data: {test_dataset_name}")
    train_dataset, c, score = read_sequences(train_dataset_name,
                                             fixed_protein_length=data_length, add_chemical_features=True,
                                             sequence_only=True, pad_sequence=True, fill_itself=False,
                                             max_length=max_length)
    print(f"Loading the sequence for test data: {test_dataset_name}")
    test_dataset, ct, scoret = read_sequences(test_dataset_name,
                                              fixed_protein_length=data_length, add_chemical_features=True,
                                              sequence_only=True, pad_sequence=True, fill_itself=False,
                                              max_length=max_length)
    print(f"Loading the iterator for train data: {train_dataset_name} and test data: {test_dataset_name}")
    _train_iterator = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    _test_iterator = DataLoader(test_dataset, batch_size=batch_size)
    return train_dataset, test_dataset, _train_iterator, _test_iterator, c, score


def get_shuffled_sample(data: torch.Tensor, n_samples):
    if n_samples > data.shape[0]:
        n_samples = data.shape[0]
    ids = torch.randperm(data.shape[0])[:n_samples]
    return data[ids]


def load_from_saved_tensor(filename):
    return torch.load(filename)


def save_tensor_to_file(filename, tensor):
    return torch.save(tensor, filename)


def aa_features():
    """ Returns chemical features regarding each amino acid
    """

    prop1 = [[1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
             [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
             [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
             [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
             [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
             [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
             [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
             [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
             [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
             [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
             [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
             [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
             [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
             [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
             [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
             [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
             [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
             [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
             [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
             [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
             [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]
    return torch.Tensor(prop1)


def one_to_number(res_str):
    """
    Convert amino acids to their index in the vocabulary

    """

    return [amino_acids_to_byte_map[r] for r in res_str]


def get_embedding_matrix():
    return seq_to_one_hot(amino_acids, True)


def to_categorical(y, num_classes):
    """ Converts a class vector to binary class matrix. """
    new_y = torch.LongTensor(y)
    n = new_y.size()[0]
    categorical = torch.zeros(n, num_classes)
    arangedTensor = torch.arange(0, n)
    intaranged = arangedTensor.long()
    categorical[intaranged, new_y] = 1
    return categorical


def seq_to_one_hot(res_seq_one, add_chemical_features=False):
    """ Create simple embeddings
    """

    ints = one_to_number(res_seq_one)

    onehot = to_categorical(ints, num_classes=len(amino_acids))

    if add_chemical_features:
        new_ints = torch.LongTensor(ints)
        feats = torch.Tensor(aa_features()[new_ints])
        return torch.cat((onehot, feats), 1)
    else:
        return onehot


def valid_protein(protein_sequence):
    """ Checks if the protein contains only valid amino acid values
    """
    for aa in protein_sequence:
        if aa not in amino_acids_set:
            print(aa)
            return False
    return True


def read_sequences(file, fixed_protein_length, add_chemical_features=False, sequence_only=False, pad_sequence=True,
                   fill_itself=False, max_length=-1):
    """ Reads and converts valid protein sequences"
    """
    proteins = []
    c = collections.Counter()
    sequences = []
    with open(file) as json_file:
        data = json.load(json_file)
        if "sequence" in data:
            sequences = data["sequence"].values()
        else:
            if "protein_sequence" in data:
                sequences = data["protein_sequence"].values()
    print("Size of sequence is {}".format(len(sequences)))
    for protein_sequence in sequences:
        if valid_protein(protein_sequence):
            chunks = len(protein_sequence)
            final_piece = (int(chunks / fixed_protein_length)) * fixed_protein_length
            protein_sequence = [torch.ByteTensor(one_to_number(protein_sequence[i:i + fixed_protein_length])) for i in
                                range(0, final_piece, fixed_protein_length)]
            final_block = protein_sequence[int(final_piece):]
            # pad sequence
            protein_temp = []
            if len(final_block) < fixed_protein_length:
                final_block += "0" * (fixed_protein_length - len(final_block))
            protein_sequence.append(torch.ByteTensor(one_to_number(final_block)))
            proteins.append(protein_sequence)
        else:
            continue

    return proteins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-m", "--modelconfig", help="model config file", type=str)
    parser.add_argument("-x", "--model", help="model to load", type=str)
    args = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config: dict = json.load(open(args.config))
    model_config: dict = json.load(open(args.modelconfig))
    print(f"Creating the model")
    model, _, device = model_factory.create_model(config, model_config)
    FIXED_PROTEIN_LENGTH = config["protein_length"]
    protein_file = "data/human_proteins.json"
    proteins = pd.read_json(protein_file)
    proteins_onehot = read_sequences(protein_file, FIXED_PROTEIN_LENGTH, add_chemical_features=False,
                                     sequence_only=True)
    model_to_load = args.model
    model.load_state_dict(torch.load(model_to_load))
    model.eval()
    embedding_list = []
    for protein in proteins_onehot:
        protein_rep = torch.stack(protein)
        protein_embeddings = model.representation(protein_rep.to(device).long())
        protein_embeddings = protein_embeddings.view(-1)
        data = torch.split(protein_embeddings, 300, 0)
        val = data[0]
        for x in range(1, len(data) - 1):
            val += data[x]
        val = val.to('cpu').detach().numpy()
        embedding_list.append(val)
    proteins['embeddings'] = embedding_list
    proteins.to_json("exports/embeddings.json")
