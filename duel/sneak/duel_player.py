import os
import sys

from duel.sneak.duel_evaluator import Evaluator
from shared.rule import Direction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.player import IPlayer

class AIPlayer(IPlayer):
    def __init__(self, evaluator:Evaluator):
        self.evaluator = evaluator

    def on_start(self, game_state):
        print("GAME START")

    def on_end(self, game_state):
        print("GAME OVER\n")

    def on_move(self, game_state):
        q = self.evaluator.evaluate(game_state)
        direction = max(zip(Direction,q), key = lambda x:x[1])[0]
        return direction