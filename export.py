""" Script used to export embeddings of proteins
"""
import torch
import data
import pandas as pd
FIXED_PROTEIN_LENGTH = 50


""" Load the proteins
"""

protein_file = "data/human_proteins.json"
proteins = pd.read_json(protein_file)
proteins_onehot = data.read_sequences(protein_file, FIXED_PROTEIN_LENGTH, add_chemical_features=False)

"""
Load the saved model
"""
from models.simple_vae import VAE
EMBEDDING_LENGTH = 20
INPUT_DIM = FIXED_PROTEIN_LENGTH * data.VOCABULARY_SIZE
model_to_load = "simple_vae_02_05_2020_15_53_09"
model = VAE(INPUT_DIM, EMBEDDING_LENGTH)
model.load_state_dict(torch.load(f"saved_models/{model_to_load}"))
model.eval()

"""
Generate embeddings
"""
protein_embeddings = model.encode(proteins_onehot.view(-1, INPUT_DIM))
proteins['embeddings'] = list(protein_embeddings.detach().numpy())
proteins.to_json("exports/embeddings.json")
