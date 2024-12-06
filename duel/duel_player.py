import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from solo.sneak.evaluator import Evaluator
from shared import rule
from shared.rule import TurnResult, Direction
from shared.player import IPlayer

class AIPlayer(IPlayer):
    def __init__(self, evaluator:Evaluator):
        self.evaluator = evaluator

    def on_start(self, game_state):
        print("GAME START")

    def on_end(self, game_state):
        print("GAME OVER\n")

    def on_move(self, game_state):
        survival_moves = list(filter(
            lambda x: x[1][0] is TurnResult.CONTINUE,
            map(
                lambda x:  (x,rule.move(game_state, x)),
                Direction
            )
        ))
        safe_moves = list(filter(
            lambda x: any(
                map(
                    lambda y: rule.move(x[1][1], y)[0] is TurnResult.CONTINUE,
                    Direction
                )
            ),
            survival_moves
        ))

        if len(safe_moves) == 0:
            safe_moves = survival_moves
        if len(safe_moves) == 0:
            return Direction.UP
        ev = list(map(
            lambda x: (x[0], self.evaluator.evaluate(x[1][1], game_state)),
            safe_moves
        ))
        # print(ev)
        choice = max(ev,
                     key=lambda x: x[1]
                     )
        return choice[0]