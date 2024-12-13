import websocket
import json

def on_open(ws):
    print("WebSocket connection opened")

def on_message(ws, message):
    data = json.loads(message)
    data['Data']['width'] = 11
    data['Data']['Height'] = 11
    snakes = data.get('Data', {}).get('Snakes', [])
    if snakes:
        for sn in snakes:
            snake = sn 
            head = snake['Body'][0]
            snake['Head'] = head

    return json.dumps(data['Data'], indent=2)

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

def run_websocket(url):
    ws = websocket.WebSocketApp(url, 
                                on_open=on_open, 
                                on_message=on_message, 
                                on_error=on_error, 
                                on_close=on_close)
    ws.run_forever()
