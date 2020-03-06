""" Script used to export embeddings of proteins
"""
import argparse
import json

import pandas as pd
import torch

import utils.model_factory as model_factory
from utils import data

"""
Load the saved model
"""

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
    proteins_onehot, _, _ = data.read_sequences(protein_file, FIXED_PROTEIN_LENGTH, add_chemical_features=False,
                                                sequence_only=True)
    model_to_load = args.model
    model.load_state_dict(torch.load(model_to_load))
    model.eval()

    protein_embeddings = model.representation(proteins_onehot.to(device).long())
    proteins['embeddings'] = list(protein_embeddings.to('cpu').detach().numpy())
    proteins.to_json("exports/embeddings.json")
