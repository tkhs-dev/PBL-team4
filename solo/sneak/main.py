import sys
import typing

from evaluator import Evaluator
from player import AIPlayer

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

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server
    if len(sys.argv) > 1:
        path = sys.argv[1]
        evaluator = Evaluator.load(path)
    else:
        evaluator = Evaluator()
    player = AIPlayer(evaluator)
    run_server({"info": info, "start": player.on_start, "move": lambda game_state: {"move":player.on_move(game_state).value}, "end": player.on_end})
