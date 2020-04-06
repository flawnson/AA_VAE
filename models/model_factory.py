import torch
import torch.optim as optim

import utils.radam as radam
from models.convolutional_base_vae import ConvolutionalBaseVAE
from models.convolutional_linear import Convolutional_Linear_VAE
from models.convolutional_vae import ConvolutionalVAE
from models.gated_cnn import GatedCNN
from models.linear_vae import LinearVAE
from models.lstm_vae import LSTMVae
from models.transformer_convolutional_vae import TransformerConvVAEModel
from models.transformer_vae import TransformerModel
from utils import data_load
from utils.optimiser import ScheduledOptim, StepOptim


def get_optimizer(optimizer_config: dict, model):
    optimisers = {
        "Adam": optim.Adam,
        "RAdam": radam.RAdam
    }
    learning_rate_schedulers = {
        "Transformer": ScheduledOptim,
        "Ramp": StepOptim
    }
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]
    optimizer = optimisers.get(optimizer_config.get("optimizer", "Adam"))(model.parameters(), lr=lr,
                                                                          weight_decay=weight_decay)
    wrapped = optimizer_config.get("LearningRateScheduler", "False")
    if wrapped != "False":
        return learning_rate_schedulers[wrapped](optimizer, lr=lr,
                                                 n_warmup_steps=optimizer_config.get("sched_freq", 4000))
    else:
        return optimizer


def create_model(config, model_config, pretrained_model=None, multigpu=False):
    models = {"convolutional_vae": ConvolutionalVAE,
              "lstm_vae": LSTMVae,
              "linear_vae": LinearVAE,
              "convolutional_linear": Convolutional_Linear_VAE,
              "convolutional_basic": ConvolutionalBaseVAE,
              "gated_cnn": GatedCNN,
              "transformer": TransformerModel,
              "transformer_convolutional": TransformerConvVAEModel
              }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.get(model_config["model_name"])(
        model_config, config["hidden_size"],
        config["embedding_size"], config["protein_length"], device,
        data_load.get_embedding_matrix(config["chem_features"] == "True"), model_config["embedding_gradient"] == "True") \
        .to(device)
    model_name = model.name

    if multigpu:
        model = torch.nn.DataParallel(model)
    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model))
    # optimizer
    return model, get_optimizer(model_config, model), device, model_name
