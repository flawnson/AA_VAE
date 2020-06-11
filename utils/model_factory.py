import torch
import torch.optim as optim

import utils.optimisers.radam as radam
from models import *
from models.convolutional_linear import Convolutional_Linear_VAE
from models.convolutional_vae import ConvolutionalVAE
from models.gated_cnn import GatedCNN
from utils.optimisers.learning_rate_optimiser import ScheduledOptim, StepOptim, LearningRateOptim
from utils.optimisers.rangerlars import RangerLars


def get_optimizer(optimizer_config: dict, model):
    optimisers = {
        "Adam": optim.Adam,
        "RAdam": radam.RAdam,
        "Ranger": RangerLars
    }
    learning_rate_schedulers = {
        "Transformer": ScheduledOptim,
        "Ramp": StepOptim,
        "Cosine": optim.lr_scheduler.CosineAnnealingLR,
        "CosineWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts
    }
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]
    optimizer = optimisers.get(optimizer_config.get("optimizer", "Adam"))(model.parameters(), lr=lr,
                                                                          weight_decay=weight_decay)
    scheduler = optimizer_config.get("LearningRateScheduler", "False")
    if scheduler != "False":
        if scheduler == "Cosine":
            epoch_max = optimizer_config.get("sched_freq", 4000)
            min_lr = lr * 0.01
            return LearningRateOptim(optimizer, optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                                     T_max=epoch_max, eta_min=min_lr))
        if scheduler == "CosineWarmRestarts":
            epoch_max = optimizer_config.get("sched_freq", 4000)
            min_lr = lr * 0.01
            return LearningRateOptim(optimizer, optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                                               T_0=epoch_max,
                                                                                               eta_min=min_lr,
                                                                                               T_mult=1))
        else:
            return learning_rate_schedulers[scheduler](optimizer, lr=lr,
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
              "transformer": TransformerVAEModel,
              "transformer_convolutional": TransformerConvVAEModel,
              "global_context_vae": GlobalContextVAEModel,
              "lstm_convolutional": LSTMConvolutionalVae
              }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.get(model_config["model_name"])(
        model_config,
        config["embedding_size"], config["protein_length"], device,
        [25, 128]).to(device)
    model_name = model.name

    if multigpu:
        model = torch.nn.DataParallel(model)
    optimiser = get_optimizer(model_config, model)
    if pretrained_model is not None:
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint)
    return model, optimiser, device, model_name
