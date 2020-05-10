""" Script used to export embeddings of proteins
"""
import argparse
import collections
import json

import pandas as pd
import torch

import utils.model_factory as model_factory
import utils.training.common as common

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


def one_to_number(res_str):
    """
    Convert amino acids to their index in the vocabulary

    """

    return [amino_acids_to_byte_map[r] for r in res_str]


def to_categorical(y, num_classes):
    """ Converts a class vector to binary class matrix. """
    new_y = torch.LongTensor(y)
    n = new_y.size()[0]
    categorical = torch.zeros(n, num_classes)
    arangedTensor = torch.arange(0, n)
    intaranged = arangedTensor.long()
    categorical[intaranged, new_y] = 1
    return categorical


def valid_protein(protein_sequence):
    """ Checks if the protein contains only valid amino acid values
    """
    for aa in protein_sequence:
        if aa not in amino_acids_set:
            print(aa)
            return False
    return True


def read_sequences(file, fixed_protein_length):
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
        if len(protein_sequence) >= 1500:
            continue
        if valid_protein(protein_sequence):
            protein_sequence = protein_sequence[:fixed_protein_length]
            # pad sequence
            if len(protein_sequence) < fixed_protein_length:
                protein_sequence += "0" * (fixed_protein_length - len(protein_sequence))
            proteins.append(torch.ByteTensor(one_to_number(protein_sequence)))
        else:
            continue
    return torch.stack(proteins)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-m", "--modelconfig", help="model config file", type=str)
    parser.add_argument("-x", "--model", help="model to load", type=str)
    parser.add_argument("-g", "--multigpu", help="multigpu mode", action="store_true")
    parser.add_argument("-o", "--outputfile", help="output file to store the embedding", type=str)
    parser.add_argument("-t", "--mimetype", help="mimetype for the output", type=str)
    parser.add_argument("-i", "--input", help="The input file", type=str)

    args = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config: dict = json.load(open(args.config))
    model_config: dict = json.load(open(args.modelconfig))
    print(f"Creating the model")

    # checkpoint = torch.load(args.model)['module.deembed.weight'].detach().cpu().numpy()
    # numpy.save("deembed.npy", checkpoint)
    model, _, device, _ = model_factory.create_model(config, model_config, args.model, args.multigpu)
    FIXED_PROTEIN_LENGTH = config["protein_length"]
    protein_file = "data/human_proteins.json"
    if args.input is not None:
        protein_file = args.input
    proteins = pd.read_json(protein_file)
    proteins = proteins[proteins['protein_sequence'].map(len) < 1500]
    proteins_onehot = read_sequences(protein_file, FIXED_PROTEIN_LENGTH)
    model.eval()
    embedding_list = []
    mu_list = []
    sigma_list = []
    representations = []
    correctness_all = []

    for protein in proteins_onehot:
        protein_rep = protein.view(1, -1).to(device).long()
        if args.multigpu:
            protein_embeddings, mu, var = model.module.representation(protein_rep)
            representation, _, _ = model(protein_rep)
        else:
            protein_embeddings, mu, var = model.representation(protein_rep)
            representation, _, _ = model(protein_rep)
        max_line = representation.argmax(axis=1).view(-1).to('cpu').detach().numpy().tolist()
        sequence = ""
        correctness = common.reconstruction_accuracy(representation, protein_rep, protein_rep.le(20))
        correctness_all.append(correctness)
        for index in max_line:
            sequence = sequence + amino_acids[index]

        # amino_acid_sequence = amino_acids[representation]
        embedding = protein_embeddings.view(-1).to('cpu').detach().numpy()
        embedding_list.append(embedding)
        mu_list.append(mu.view(-1).to('cpu').detach().numpy())
        sigma_list.append(var.view(-1).to('cpu').detach().numpy())
        representations.append(sequence)

    proteins['embeddings'] = embedding_list
    proteins['mu'] = mu_list
    proteins['sigma'] = sigma_list
    proteins['reconstruction'] = representations
    proteins['correctness'] = correctness_all
    print(proteins.describe())
    if args.mimetype == "application/json":
        proteins.to_json(args.outputfile)
    if args.mimetype == "text/csv":
        proteins.to_csv(args.outputfile)
