import argparse
import sys
from logging import getLogger, DEBUG, StreamHandler, Formatter, INFO

import torch

from api_client import TestApiClient
from trainer import ReinforcementTrainer, CancelToken
from trainer import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_url', type=str, default='http://140.83.48.233/train-manager')
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--cache_all', action='store_true')
    args = parser.parse_args()

    logger = getLogger("Trainer")
    lvl = DEBUG if args.log_level == 'DEBUG' else INFO
    ch = StreamHandler(stream=sys.stdout)
    ch.setLevel(lvl)
    ch.setFormatter(Formatter('[%(asctime)s %(levelname)s] %(message)s'))
    logger.setLevel(lvl)
    logger.addHandler(ch)

    task = {
        "type": "REINFORCEMENT",
        "baseModelId": None,
    }
    bin = ReinforcementTrainer(logger, TestApiClient(), torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), CancelToken()).start(task)
    with open("model.pth", "wb") as f:
        f.write(bin)
    exit(0)

    train(api_url=args.api_url, logger=logger, cache_all=args.cache_all)