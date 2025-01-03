#04
#python 3.12.7
#
#--dependencies--
#Flask==2.3.2
#requests
#torch
#deap
#numpy

import os

import argparse
import requests
import torch

from duel_evaluator import Evaluator
from duel_player import AIPlayer

def info(args):
    author = "sneak"
    if args.model:
        author = args.model
    return {
        "apiversion": "1",
        "author": author,  # TODO: Your Battlesnake Username
        "color": "#ff0000",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }

# Start server when `python main.py` is run
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='./evaluator.pth')
    parser.add_argument('--zipped', action='store_true')
    parser.add_argument('-m', '--model', type=str)
    parser.add_argument('--safe', action='store_true')
    args = parser.parse_args()
    from shared.server import run_server
    evaluator = Evaluator()
    if args.model:
        if not os.path.exists("./cache"):
            os.makedirs("./cache")
        if os.path.exists(f"./cache/{args.model}"):
            with open(f"./cache/{args.model}", "rb") as f:
                data = f.read()
        else:
            data = requests.get(f'http://140.83.48.233/train-manager/static/models/{args.model}')
            data.raise_for_status()
            data = data.content
        with open(f"./cache/{args.model}", "wb") as f:
            f.write(data)
        args.zipped = True
        args.input = f"./cache/{args.model}"
    if args.zipped:
        import lzma
        with lzma.open(args.input, "rb") as f:
            evaluator.model.load_state_dict(torch.load(f, weights_only=True)["model"])
    else:
        evaluator.model.load_state_dict(torch.load(args.input, weights_only=True))
    player = AIPlayer(evaluator, safe=args.safe)
    run_server({"info":lambda :info(args), "start": player.on_start, "move": lambda game_state: {"move":player.on_move(game_state).value}, "end": player.on_end})
