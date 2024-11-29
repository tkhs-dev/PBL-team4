import websocket
import json

url = "wss://engine.battlesnake.com/games/529c4b28-bcf9-4d09-ab32-ab36fe75869b/events"

def on_open(ws):
    print("WebSocket connection opened")

def on_message(ws, message):
    data = json.loads(message)  
    print("Received message:", json.dumps(data, indent=2))  

def on_error(ws, error):
    print("WebSocket error:", error)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed")

ws = websocket.WebSocketApp(url, 
                            on_open=on_open, 
                            on_message=on_message, 
                            on_error=on_error, 
                            on_close=on_close)

ws.run_forever()
