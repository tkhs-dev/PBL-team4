import os
import sys

from duel.sneak.duel_evaluator import Evaluator
from shared.rule import Direction, move, TurnResult

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.player import IPlayer

class AIPlayer(IPlayer):
    def __init__(self, evaluator:Evaluator, safe:bool = False):
        self.evaluator = evaluator
        self.safe = safe

    def on_start(self, game_state):
        return 0

    def on_end(self, game_state):
        return 0

    def on_move(self, game_state):
        q = self.evaluator.evaluate(game_state)
        if not self.safe:
            print(game_state['turn'],list(map(lambda x:'{:.2f}'.format(x), q)))
            direction = max(zip(Direction,q), key = lambda x:x[1])[0]
            return direction
        safes = list(filter(lambda x:move(game_state,x[0])[0]!=TurnResult.LOSE, zip(Direction,q)))
        if len(safes) == 0:
            return max(zip(Direction,q), key = lambda x:x[1])[0]
        direction = max(safes, key = lambda x:x[1])[0]
        return direction