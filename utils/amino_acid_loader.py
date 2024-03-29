import collections
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from utils.data.common import load_data_from_file, load_from_saved_tensor, save_tensor_to_file
from utils.logger import log

"""
Valid amino acids
X - unknown
U - Selenocysteine
0 - padding for fixed length encoders
"""
amino_acids = "UCSTPAGNDEQHRKMILVFYWX0ZB"
VOCABULARY_SIZE = len(amino_acids)

amino_acids_to_byte_map = {r: amino_acids.index(r) for r in amino_acids}
amino_acids_set = {r for r in amino_acids}


class ProteinIterableDataset(IterableDataset):

    def __init__(self, filename, protein_length):
        self.filename = filename
        self.length = protein_length

    def __process_one_seq(self, sequence):
        return process_sequence(sequence, self.length)

    def __iter__(self):
        file_itr = open(self.filename)
        mapped_itr = map(self.__process_one_seq, file_itr)
        return mapped_itr


def to_categorical(num_classes):
    """ Converts a class vector to binary class matrix. """
    categorical = torch.eye(num_classes)
    unused = [amino_acids_to_byte_map['X'], amino_acids_to_byte_map['0']]
    for x in unused:
        categorical[[x, x]] = 0
    return categorical


def __load_data(pt_file, filetype, dataset_path, max_length, data_length):
    if os.path.exists(pt_file):
        # pass
        return load_from_saved_tensor(pt_file)
    else:
        return process_sequences(
            load_data_from_file(dataset_path, filetype=filetype), max_length, data_length, pad_sequence=True,
            pt_file=pt_file)


def load_data(_config: dict, max_length=-1):
    data_length = _config["protein_length"]
    batch_size = _config["batch_size"]  # number of data points in each batch

    train_dataset_name = os.getcwd() + "/" + _config["train_dataset_name"]
    test_dataset_name = os.getcwd() + "/" + _config["test_dataset_name"]

    log.info(f"Loading the sequence for train data: {train_dataset_name} and test data: {test_dataset_name}")
    pt_file = f"{train_dataset_name}_{data_length}_{True}_{True}_{max_length}.pt"
    train_dataset, c, score, length_scores = __load_data(pt_file, _config.get("train_datatype", "text/json"),
                                                         train_dataset_name, max_length, data_length)

    log.info(f"Loading the sequence for test data: {test_dataset_name}")
    pt_file = f"{test_dataset_name}_{data_length}_{True}_{True}_{max_length}.pt"
    test_dataset, ct, scoret, _ = __load_data(pt_file, _config.get("test_datatype", "text/json"),
                                              test_dataset_name, max_length, data_length)

    log.info(f"Loading the iterator for train data: {train_dataset_name} and test data: {test_dataset_name}")
    _train_iterator = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    _test_iterator = DataLoader(test_dataset, batch_size=batch_size)
    return _train_iterator, _test_iterator, c, score, length_scores


def load_large_data(config: dict):
    data_length = config["protein_length"]
    batch_size = config["batch_size"]  # number of data points in each batch
    train_dataset_name = os.getcwd() + "/" + config["train_dataset_name"]
    test_dataset_name = os.getcwd() + "/" + config["test_dataset_name"]

    plateau_weights = [0.0]
    plateau_weights.extend([1.0] * 20)
    plateau_weights.extend([0.0, 0.0])
    log.info(f"Loading the sequence for train data: {train_dataset_name} and test data: {test_dataset_name}")
    train_dataset_iter = ProteinIterableDataset(train_dataset_name, data_length)
    log.info(f"Loading the sequence for test data: {test_dataset_name}")
    pt_file = f"{test_dataset_name}_{data_length}_{True}_{True}.pt"
    if os.path.exists(pt_file):
        # pass
        test_dataset, ct, scoret, _ = load_from_saved_tensor(pt_file)
    else:
        filetype = config.get("test_datatype", "text/json")
        test_dataset, ct, scoret, _ = process_sequences(load_data_from_file(test_dataset_name, filetype=filetype), -1,
                                                        data_length, pad_sequence=True, pt_file=pt_file)
    log.info(f"Loading the iterator for train data: {train_dataset_name} and test data: {test_dataset_name}")
    _train_iterator = DataLoader(train_dataset_iter, shuffle=False, batch_size=batch_size)
    _test_iterator = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return _train_iterator, _test_iterator, None, torch.FloatTensor(plateau_weights), None


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


def get_embedding_matrix(features: bool = True):
    log.info("Generating embedding matrix, with chemical features: {}".format(features))
    return seq_to_one_hot(amino_acids, features)


def seq_to_one_hot(res_seq_one, add_chemical_features=False):
    """ Create simple embeddings
    """

    ints = one_to_number(res_seq_one)

    onehot = to_categorical(num_classes=len(amino_acids))

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


def process_sequence(protein_sequence, fixed_protein_length=1500, pad_sequence=True):
    protein_sequence = protein_sequence.rstrip("\n")
    if len(protein_sequence) == 0:
        log.error("No data in sequence")
    lengths = []
    if valid_protein(protein_sequence):
        length = len(protein_sequence)
        if length > 200:
            length = 200
        lengths.append(length)
        protein_sequence = protein_sequence[:fixed_protein_length]
        # pad sequence
        if pad_sequence:
            if len(protein_sequence) < fixed_protein_length:
                protein_sequence += "0" * (fixed_protein_length - len(protein_sequence))
        return torch.ByteTensor(one_to_number(protein_sequence))
    # log.error("Invalid sequence found{}".format(protein_sequence))
    return torch.ByteTensor(one_to_number("0" * fixed_protein_length))


def calculate_bucket_cost(length_counter: collections.Counter, fixed_protein_length, window):
    values = [0.0] * (fixed_protein_length + 2)
    for i in range(fixed_protein_length + 1):
        for j in range(-window, window + 1):
            values[i] = values[i] + length_counter.get(i + j, 0)
        if values[i] == 0:
            values[i] = 1
        values[i] = values[i] / (2 * window + 1)

    return values


def process_sequences(sequences, max_length=-1, fixed_protein_length=1500, pad_sequence=True, pt_file=None):
    proteins = []
    c = collections.Counter()
    lengths = []
    i = 0
    lengths = []
    for protein_sequence in sequences:
        if max_length != -1:
            if i > max_length:
                break
        if valid_protein(protein_sequence):
            c.update(protein_sequence)
            length = len(protein_sequence)
            if length > 200:
                length = 200
            lengths.append(length)
        processed = process_sequence(protein_sequence, fixed_protein_length, pad_sequence)
        if processed is not None:
            proteins.append(processed)
        i = i + 1

    log.info("Size of sequence is {}".format(i))
    length_counter = collections.Counter(lengths)
    averaging_window = 10
    window = int(averaging_window / 2)

    scores = []
    buckets = calculate_bucket_cost(length_counter, fixed_protein_length, window)
    length = sum(c.values())
    for k in amino_acids:
        if c[k] > 0 and amino_acids_to_byte_map[k] <= 20:
            rarity = length / (21 * c[k])
            if rarity > 5:
                rarity = 0.05
            rarity = 1
            scores.append(rarity)
        else:
            scores.append(0)

    data = torch.stack(proteins), c, torch.FloatTensor(scores), buckets
    if pt_file is not None:
        save_tensor_to_file(pt_file, data)
    return data
