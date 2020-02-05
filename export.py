""" Script used to export embeddings of proteins
"""
import torch
import data

""" Load the proteins
"""

protein_file = "data/train_set_small_50"
proteins = data.read_sequences(protein_file, 50, add_chemical_features=False)

"""
Load the saved model
"""
from models.simple_vae import VAE
FIXED_PROTEIN_LENGTH = 50
EMBEDDING_LENGTH = 20
INPUT_DIM = FIXED_PROTEIN_LENGTH * data.VOCABULARY_SIZE
model_to_load = "simple_vae_02-05-2020_10:25:51"
model = VAE(INPUT_DIM, EMBEDDING_LENGTH)
model.load_state_dict(torch.load(f"saved_models/{model_to_load}"))
model.eval()

"""
Generate embeddings
"""
embeddings = model.encode(proteins.view(-1, INPUT_DIM))
