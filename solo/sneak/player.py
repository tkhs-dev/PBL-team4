import abc
import os
import sys
from abc import ABCMeta

from solo.sneak.evaluator import Evaluator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared import rule
from shared.rule import TurnResult, Direction


class IPlayer(metaclass=ABCMeta):
    @abc.abstractmethod
    def on_start(self, game_state):
        raise NotImplementedError

    @abc.abstractmethod
    def on_end(self, game_state):
        raise NotImplementedError

    @abc.abstractmethod
    def on_move(self, game_state) -> Direction:
        raise NotImplementedError

class AIPlayer(IPlayer):
    def __init__(self, evaluator:Evaluator):
        self.evaluator = evaluator

    def on_start(self, game_state):
        print("GAME START")

    def on_end(self, game_state):
        print("GAME OVER\n")

    def on_move(self, game_state):
        safe_moves = list(filter(
            lambda x: x[1][0] is TurnResult.CONTINUE,
            map(
                lambda x:  (x,rule.move(game_state, x)),
                Direction
            )
        ))
        if len(safe_moves) == 0:
            return Direction.UP
        ev = list(map(
            lambda x: (x[0], self.evaluator.evaluate(x[1][1])),
            safe_moves
        ))
        # print(ev)
        choice = max(ev,
                     key=lambda x: x[1]
                     )
        return choice[0]