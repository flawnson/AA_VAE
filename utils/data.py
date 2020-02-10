import torch

"""
Valid amino acids
U - unknown
X - missing
0 - padding for fixed length encoders
"""
amino_acids = "CSTPAGNDEQHRKMILVFYWUX0"


def aa_features():
    """ Returns chemical features regarding each amino acid
    """

    prop1 = [[1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
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
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
             [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]
    return torch.Tensor(prop1)


def one_to_number(res_str):
    """
    Convert amino acids to their index in the vocabulary

    """

    return [amino_acids.index(r) for r in res_str]


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
    new_ints = torch.LongTensor(ints)

    feats = torch.Tensor(aa_features()[new_ints])
    onehot = to_categorical(ints, num_classes=len(amino_acids))

    if add_chemical_features:
        return torch.cat((onehot, feats), 1)
    else:
        return onehot


def valid_protein(protein_sequence):
    """ Checks if the protein contains only valid amino acid values
    """
    for aa in protein_sequence:
        if aa not in amino_acids:
            return False
    return True


def read_sequences(file, fixed_protein_length, add_chemical_features=False):
    """ Reads and converts valid protein sequences"
    """

    proteins = []
    with open(file) as f:
        for protein_sequence in f:
            protein_sequence = protein_sequence.strip()
            if valid_protein(protein_sequence):

                # pad sequence
                if len(protein_sequence) < fixed_protein_length:
                    protein_sequence += "0" * (fixed_protein_length - len(protein_sequence))

                proteins.append(seq_to_one_hot(protein_sequence, add_chemical_features=add_chemical_features))
            else:
                print("Protein", protein_sequence, "contains invalid characters")

    return torch.stack(proteins)


if __name__ == "__main__":
    proteins = read_sequences("data/test_set", 50, add_chemical_features=True)
