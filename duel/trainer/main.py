import argparse
import sys
from logging import getLogger, DEBUG, StreamHandler, Formatter, INFO

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

    train(api_url=args.api_url, logger=logger, cache_all=args.cache_all)