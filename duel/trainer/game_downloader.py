import websocket
import json

from shared.rule import Direction

class GameDownloader:
    def __init__(self):
        self.player_id = None
        self.url = None

    def _lowercase_keys(self, obj):
        """
        JSONパース時に辞書のキーを小文字に変換する関数
        """
        if isinstance(obj, dict):
            # キーを小文字に変換
            return {key.lower(): value for key, value in obj.items()}
        return obj

    def _on_open(self, ws):
        self.result = []

    def _on_message(self, ws, message):
        data = json.loads(message, object_hook=self._lowercase_keys)
        if data["type"] != "frame":
            return

        turn_result = {
            "game_state": {
                "turn": data["data"]["turn"],
                "board": {
                    "height": 11,
                    "width": 11,
                    "food": data["data"]["food"],
                    "snakes": data["data"]["snakes"],
                },
                "you": {},
            }
        }
        snakes = turn_result["game_state"]["board"]["snakes"]
        if snakes:
            for sn in snakes:
                snake = sn
                head = snake['body'][0]
                snake['head'] = head
                snake['length'] = len(snake['body'])
                if snake['author'] == self.player_id:
                    turn_result["game_state"]['you'] = snake
                    if turn_result["game_state"]["turn"] > 0:
                        prev_head = snake['body'][1]
                        action = Direction.UP
                        if head['x'] > prev_head['x']:
                            action = Direction.RIGHT
                        elif head['x'] < prev_head['x']:
                            action = Direction.LEFT
                        elif head['y'] < prev_head['y']:
                            action = Direction.DOWN
                        self.result[-1]['action'] = action



        self.result.append(turn_result)

    def _on_error(self, ws, error):
        print("WebSocket error:", error)

    def _on_close(self, ws, close_status_code, close_msg):
        pass

    def _run_websocket(self):
        ws = websocket.WebSocketApp(self.url,
                                    on_open=self._on_open,
                                    on_message=self._on_message,
                                    on_error=self._on_error,
                                    on_close=self._on_close)
        ws.run_forever()

    def download_data(self, game_id, player_id):
        self.player_id = player_id
        self.url = "wss://engine.battlesnake.com/games/{}/events".format(game_id)
        self._run_websocket()
        return self.result

#           --USAGE--
if __name__ == "__main__":
    downloader = GameDownloader()
    data = downloader.download_data("c70b1355-9a9d-4e5a-8a79-063d9ce4d2ff", "yannikm")
    print(data)

