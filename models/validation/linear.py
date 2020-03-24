import json
import torch
import argparse
import numpy as np
import os.path as osp

from torch.nn import functional as f
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from data_processing import *
from model import LinearModel
from trainer import TrainLinear

if __name__ == "__main__":
    path = osp.join('simple-vae', 'configs')  # Implicitly used to get config file?
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-f", "--config", help="json config file", type=str)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_file = osp.join(osp.dirname(osp.dirname(osp.dirname(__file__))), "exports", "embeddings.json")
    json_embed = open(embed_file)
    json_data = json.load(json_embed)

    json_file = open(args.config)
    json_config = json.load(json_file)

    data_config = json_config.get('data_config')
    if data_config['task'] == "binary":
        dataset = BinaryLabels(json_data, data_config.get("onehot"))
    elif data_config['task'] == "quaternary":
        dataset = QuaternaryLabels(json_data, data_config.get("onehot"))
    elif data_config['task'] == "quinary":
        dataset = QuinaryLabels(json_data, data_config.get("onehot"))
    elif data_config['task'] == "protein":
        dataset = ProteinLabels(json_data, data_config.get("onehot"))
    else:
        raise NotImplementedError("Task described is not implemented")

    model_config = json_config.get('model_config')
    model = LinearModel(len(np.squeeze(list(dataset.x["embeddings"].values())[0])),  # Embedding size
                        len(dataset.y[0]),  # Number of classes
                        model_config.get('layer_sizes'),  # List of layer sizes
                        model_config.get('dropout')).to(device)  # Boolean value

    run_config = json_config.get('run_config')
    TrainLinear(run_config, data_config, dataset, model, device).run()
