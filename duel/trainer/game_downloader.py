import websocket
import json

class GameDownloader:
    def _lowercase_keys(self, obj):
        """
        JSONパース時に辞書のキーを小文字に変換する関数
        """
        if isinstance(obj, dict):
            # キーを小文字に変換
            return {key.lower(): value for key, value in obj.items()}
        return obj

    def _on_open(self, ws):
        print("WebSocket connection opened")
        self.result = []

    def _on_message(self, ws, message):
        data = json.loads(message, object_hook=self._lowercase_keys)
        data['data']['width'] = 11
        data['data']['height'] = 11
        snakes = data.get('data', {}).get('snakes', [])
        if snakes:
            for sn in snakes:
                snake = sn
                head = snake['body'][0]
                snake['head'] = head

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
if __name__ == "__main__":
    downloader = GameDownloader()
    data = downloader.download_data("c70b1355-9a9d-4e5a-8a79-063d9ce4d2ff")
    print(data)

