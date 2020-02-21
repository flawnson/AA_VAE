""" Script used to export embeddings of proteins
"""
import torch
import pandas as pd

import argparse
import json
from utils import data

"""
Load the saved model
"""
from models.simple_vae import VAE
from models.convolutional_vae import ConvolutionalVAE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-m", "--modelconfig", help="model config file", type=str)
    parser.add_argument("-x", "--model", help="model to load", type=str)
    args = parser.parse_args()
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config: dict = json.load(open(args.config))
    model_config: dict = json.load(open(args.modelconfig))
    print(f"Creating the model")
    if model_config["model_name"] == "convolutional_vae":
        model = ConvolutionalVAE(model_config["convolutional_parameters"], config["hidden_size"],
                                 config["embedding_size"], config["feature_length"], device,
                                 data.get_embedding_matrix())
    else:
        model = VAE(1500, 20).to(device)  # 20 is number of hidden dimensio

    FIXED_PROTEIN_LENGTH = config["protein_length"]
    protein_file = "data/human_proteins.json"
    proteins = pd.read_json(protein_file)
    proteins_onehot = data.read_sequences(protein_file, FIXED_PROTEIN_LENGTH, add_chemical_features=False,
                        sequence_only = True)
    model_to_load = args.model
    model.load_state_dict(torch.load(f"saved_models/{model_to_load}"))
    model.eval()

    protein_embeddings = model.representation(proteins_onehot)
    proteins['embeddings'] = list(protein_embeddings.detach().numpy())
    proteins.to_json("exports/embeddings.json")
