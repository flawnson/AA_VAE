"""
This code is a variation of simple VAE from https://graviraja.github.io/vanillavae/
"""
import argparse
import json

from utils.train import trainer
from utils.tuning import tuner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Config file parser")
    parser.add_argument("-c", "--config", help="common config file", type=str)
    parser.add_argument("-m", "--model", help="model config file", type=str)
    parser.add_argument("-b", "--benchmarking", help="benchmarking run config", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--training', dest='training', action='store_true')
    feature_parser.add_argument('--tuning', dest='training', action='store_false')
    parser.set_defaults(training=True)
    args = parser.parse_args()

    config_: dict = json.load(open(args.config))
    model_config_: dict = json.load(open(args.model))

    if args.training:
        trainer(config_, model_config_)
    else:
        tuner(True, config_, model_config_)
