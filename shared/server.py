import logging
import os
import time
import typing

from flask import Flask
from flask import request


def run_server(handlers: typing.Dict):
    app = Flask("Battlesnake")

    @app.get("/")
    def on_info():
        if "info" in handlers:
            return handlers["info"]()
        return {
            "apiversion": "1",
            "author": "",  # TODO: Your Battlesnake Username
            "color": "#ff0000",  # TODO: Choose color
            "head": "default",  # TODO: Choose head
            "tail": "default",  # TODO: Choose tail
        }

    @app.post("/start")
    def on_start():
        game_state = request.get_json()
        handlers["start"](game_state)
        return "ok"

    @app.post("/move")
    def on_move():
        start = time.time()
        game_state = request.get_json()
        result = handlers["move"](game_state)
        end = time.time()
        print(f"Elapsed time: {(end - start)*1000}ms")
        return result

    @app.post("/end")
    def on_end():
        game_state = request.get_json()
        handlers["end"](game_state)
        return "ok"

    @app.after_request
    def identify_server(response):
        response.headers.set(
            "server", "battlesnake/github/starter-snake-python"
        )
        return response

    host = "0.0.0.0"
    port = int(os.environ.get("PORT", "8000"))

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"\nRunning Battlesnake at http://{host}:{port}")
    app.run(host=host, port=port)
