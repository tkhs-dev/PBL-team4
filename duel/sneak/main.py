#04
#python 3.12.7
#
#--dependencies--
#Flask==2.3.2
#requests
#torch
#deap
#numpy

import sys

from duel_evaluator import Evaluator
from duel_player import AIPlayer

# Start server when `python main.py` is run
if __name__ == "__main__":
    from shared.server import run_server
    if len(sys.argv) > 1:
        path = sys.argv[1]
        evaluator = Evaluator.load(path)
    else:
        evaluator = Evaluator.load("./evaluator.pth")
    player = AIPlayer(evaluator)
    run_server({"start": player.on_start, "move": lambda game_state: {"move":player.on_move(game_state).value}, "end": player.on_end})
