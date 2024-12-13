import websocket
import json

class GameDownloader:
    def _on_open(self, ws):
        print("WebSocket connection opened")
        self.result = []

    def _on_message(self, ws, message):
        data = json.loads(message)
        data['Data']['width'] = 11
        data['Data']['Height'] = 11
        snakes = data.get('Data', {}).get('Snakes', [])
        if snakes:
            for sn in snakes:
                snake = sn
                head = snake['Body'][0]
                snake['Head'] = head

        self.result.append(data)

    def _on_error(self, ws, error):
        print("WebSocket error:", error)

    def _on_close(self, ws, close_status_code, close_msg):
        print("WebSocket connection closed")

    def _run_websocket(self):
        ws = websocket.WebSocketApp(self.url,
                                    on_open=self._on_open,
                                    on_message=self._on_message,
                                    on_error=self._on_error,
                                    on_close=self._on_close)
        ws.run_forever()

    def download_data(self, game_id):
        self.url = "wss://engine.battlesnake.com/games/{}/events".format(game_id)
        self._run_websocket()
        return self.result

#           --USAGE--
# if __name__ == "__main__":
#     downloader = GameDownloader()
#     data = downloader.download_data("c70b1355-9a9d-4e5a-8a79-063d9ce4d2ff")
#     print(data)

