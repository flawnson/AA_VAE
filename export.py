""" Script used to export embeddings of proteins
"""
import argparse
import collections
import json

import pandas as pd
import torch

import models.model_factory as model_factory

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
        if valid_protein(protein_sequence):
            chunks = len(protein_sequence)
            final_piece = (int(chunks / fixed_protein_length)) * fixed_protein_length
            protein_sequence = [torch.ByteTensor(one_to_number(protein_sequence[i:i + fixed_protein_length])) for i in
                                range(0, final_piece, fixed_protein_length)]
            final_block = protein_sequence[int(final_piece):]
            # pad sequence
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
    parser.add_argument("-g", "--multigpu", help="multigpu mode", action="store_true")
    args = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config: dict = json.load(open(args.config))
    model_config: dict = json.load(open(args.modelconfig))
    print(f"Creating the model")
    model, _, device, _ = model_factory.create_model(config, model_config, args.model, args.multigpu)
    FIXED_PROTEIN_LENGTH = config["protein_length"]
    protein_file = "data/human_proteins.json"
    proteins = pd.read_json(protein_file)
    proteins_onehot = read_sequences(protein_file, FIXED_PROTEIN_LENGTH)
    model.eval()
    embedding_list = []
    for protein in proteins_onehot:
        protein_rep = torch.stack(protein)
        protein_embeddings = model.representation(protein_rep.to(device).long())
        val = protein_embeddings.to('cpu').detach().numpy()
        embedding_list.append(val)
    proteins['embeddings'] = embedding_list
    proteins.to_json("exports/embeddings.json")
