import json
import torch
import argparse
import numpy as np
import os.path as osp

from torch.nn import functional as f
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from models.validation.data_processing import *
from models.validation.model import LinearModel
from models.validation.trainer import TrainLinear

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
        dataset = BinaryLabels(json_data)
    elif data_config['task'] == "quaternary":
        dataset = QuaternaryLabels(json_data)
    elif data_config['task'] == "quinary":
        dataset = QuinaryLabels(json_data)
    elif data_config['task'] == "protein":
        dataset = ProteinLabels(json_data)
    else:
        raise NotImplementedError("Task described is not implemented")

    model_config = json_config.get('model_config')
    model = LinearModel(model_config.get('in_size'),
                        model_config.get('out_size'),
                        model_config.get('layer_sizes'),
                        model_config.get('dropout')).to(device)

    train_config = json_config.get('train_config')
    TrainLinear(train_config, data_config, dataset, model, device).run()
