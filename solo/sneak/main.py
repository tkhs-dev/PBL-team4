import os
import typing
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared import rule
from shared.rule import TurnResult, Direction
from solo.sneak.evaluator import Evaluator

evaluator = Evaluator.load("solo/sneak/evaluator.pth")

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "",  # TODO: Your Battlesnake Username
        "color": "#888888",  # TODO: Choose color
        "head": "default",  # TODO: Choose head
        "tail": "default",  # TODO: Choose tail
    }

# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")

# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")

# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:
    safe_moves = list(filter(
        lambda x: x[1][0] is TurnResult.CONTINUE,
        map(
            lambda x:  (x,rule.move(game_state, x)),
            Direction
        )
    ))
    if len(safe_moves) == 0:
        return {"move": "up"}
    ev = list(map(
        lambda x: (x[0], evaluator.evaluate(x[1][1])),
        safe_moves
    ))
    print(ev)
    choice = max(ev,
        key=lambda x: x[1]
    )
    return {"move": choice[0].value}


# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
