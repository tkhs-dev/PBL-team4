import json
import os
import sys

import ctypes

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

lib = ctypes.CDLL("../../embedded-rules/build/rules.dll")
lib.StartSoloGame.restype = ctypes.c_char_p
CALLBACKFUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p)

class Client(ctypes.Structure):
    _fields_ = [
        ("on_start", CALLBACKFUNC),
        ("on_move", CALLBACKFUNC),
        ("on_end", CALLBACKFUNC)
    ]

class GameSettings(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_long),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("food_spawn_chance", ctypes.c_int),
        ("minimum_food", ctypes.c_int)
    ]

def start_solo_game(callback_funcs : Client, settings : GameSettings) -> dict:
    result_text = lib.StartSoloGame(callback_funcs, settings)
    result = json.loads(result_text)
    return result
