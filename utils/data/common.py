import json
from itertools import groupby

import torch

from utils.logger import log


def get_shuffled_sample(data: torch.Tensor, n_samples):
    if n_samples > data.shape[0]:
        n_samples = data.shape[0]
    ids = torch.randperm(data.shape[0])[:n_samples]
    return data[ids]


def load_from_saved_tensor(filename):
    return torch.load(filename)


def save_tensor_to_file(filename, tensor):
    return torch.save(tensor, filename)


def fasta_reader(fasta_name):
    """
    modified from Brent Pedersen
    Correct Way To Parse A Fasta File In Python
    given a fasta file.
    """
    sequences = []
    with open(fasta_name) as fh:
        # ditch the boolean (x[0]) and just keep the header or sequence since
        # we know they alternate.
        faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
        for header in faiter:
            # drop the ">"
            headerStr = header.__next__()[1:].strip()
            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())
            sequences.append(seq)
    return sequences


def read_sequences_from_json(file):
    """ Reads and converts valid protein sequences"
    """

    sequences = []

    with open(file) as json_file:
        data = json.load(json_file)
        if "sequence" in data:
            sequences = data["sequence"].values()
        else:
            if "protein_sequence" in data:
                sequences = data["protein_sequence"].values()
    i = 0
    log.info("Size of sequence is {}".format(len(sequences)))
    return sequences


def load_data_from_file(file, filetype="text/json"):
    sequences = []
    if filetype == "text/json":
        sequences = read_sequences_from_json(file)

    if filetype == "text/fasta":
        sequences = load_fasta(file)
    return sequences


def load_fasta(file):
    sequences = fasta_reader(file)
    log.info("Size of sequence is {}".format(len(sequences)))
    return sequences
