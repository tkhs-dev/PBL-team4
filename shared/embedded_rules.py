import ctypes
import json
import os
import sys

from shared.rule import Direction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

lib = ctypes.CDLL("../../embedded-rules/build/rules.dll")
lib.StartSoloGame.restype = ctypes.POINTER(ctypes.c_char)
lib.StartDuelGame.restype = ctypes.POINTER(ctypes.c_char)
CALLBACKFUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_char))

class ClientCStruct(ctypes.Structure):
    _fields_ = [
        ("on_start", CALLBACKFUNC),
        ("on_move", CALLBACKFUNC),
        ("on_end", CALLBACKFUNC)
    ]

class Client:

    def __init__(self):
        self.on_start = lambda state: 0
        self.on_move = lambda state: 0
        self.on_end = lambda state: 0

    @staticmethod
    def direction_to_int(direction):
        if direction == Direction.UP:
            return 0
        elif direction == Direction.DOWN:
            return 1
        elif direction == Direction.LEFT:
            return 2
        elif direction == Direction.RIGHT:
            return 3
        else :
            return 0

    @staticmethod
    def call_func(func, state, is_json = True):
        json_str = ctypes.string_at(state)
        if is_json:
            game_state = json.loads(json_str)
        else:
            game_state = {"":""}
        ctypes.cdll.msvcrt.free(state)
        return func(game_state)

    def to_c_struct(self) -> ClientCStruct:
        client = ClientCStruct()
        client.on_start = CALLBACKFUNC(lambda str_ptr: self.call_func(self.on_start, str_ptr, False))
        client.on_move = CALLBACKFUNC(lambda str_ptr: self.direction_to_int(self.call_func(self.on_move, str_ptr)))
        client.on_end = CALLBACKFUNC(lambda str_ptr: self.call_func(self.on_end, str_ptr))
        return client

class GameSettingCStruct(ctypes.Structure):
    _fields_ = [
        ("seed", ctypes.c_long),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("food_spawn_chance", ctypes.c_int),
        ("minimum_food", ctypes.c_int)
    ]

class GameSettings:
    def __init__(self):
        self.seed = 0
        self.width = 0
        self.height = 0
        self.food_spawn_chance = 0
        self.minimum_food = 0

    def to_c_struct(self) -> GameSettingCStruct:
        setting = GameSettingCStruct()
        setting.seed = self.seed
        setting.width = self.width
        setting.height = self.height
        setting.food_spawn_chance = self.food_spawn_chance
        setting.minimum_food = self.minimum_food
        return setting

def start_solo_game(callback_funcs : Client, settings : GameSettings) -> dict:
    result_text = lib.StartSoloGame(callback_funcs.to_c_struct(), settings.to_c_struct())
    result = json.loads(ctypes.string_at(result_text))
    ctypes.cdll.msvcrt.free(result_text)
    return result

def start_duel_game(callback_funcs1 : Client, callback_funcs2 : Client, settings : GameSettings) -> dict:
    result_text = lib.StartDuelGame(callback_funcs1.to_c_struct(), callback_funcs2.to_c_struct(), settings.to_c_struct())
    result = json.loads(ctypes.string_at(result_text))
    ctypes.cdll.msvcrt.free(result_text)
    return result
