import copy
from enum import Enum

class TurnResult(Enum):
    CONTINUE = 1
    WIN = 2
    LOSE = 3
    DRAW = 4

class Direction(Enum):
    UP = (0,1,"up")
    DOWN = (0,-1,"down")
    LEFT = (1,0,"left")
    RIGHT = (0,1,"right")

def move(game_state:dict, direction:Direction) -> (TurnResult, dict):
    #------ IMPORTANT ------
    #pythonのdictは参照渡しであるため、game_stateを変更すると元のgame_stateも変更される
    #そのため,game_stateの値を変更することは禁止
    #game_stateは,元の盤面の状態を取得したいときに使用する
    #次の状態に変更するなどの場合は,以下のnext_stateを変更し,戻り値として返すこと
    next_state = copy.deepcopy(game_state)
    next_head = copy.deepcopy(next_state["you"]["head"])
    next_head["x"] += direction.value[0]
    next_head["y"] += direction.value[1]

    #何らかの衝突があるかどうかを判定し,それに応じて処理、結果を返す
    if _is_head_out_of_bounds(game_state, next_head):
        return TurnResult.LOSE, None
    elif (snake := _is_head_colliding_with_other_snake(game_state, next_head)) is not None:
        #snakeには衝突したsnakeのdictが入る
        #もし接触したのが他の蛇の頭の場合,蛇の長さに応じて勝敗が変わる
        #体に接触した場合はこちらの負け
        #TODO
        pass
    elif _is_head_colliding_with_self(game_state, next_head):
        return TurnResult.LOSE, None


    if _is_head_colliding_with_food(game_state, next_head):
        #餌に接触した場合の処理
        #TODO
        pass
    else:
        #何も接触しなかった場合の処理
        #TODO
        pass

    return TurnResult.CONTINUE, next_state

def _is_head_out_of_bounds(game_state:dict, next_head:(int,int)) -> bool:
    return next_head["x"] < 0 or next_head["x"] >= game_state["board"]["width"] or next_head["y"] < 0 or next_head["y"] >= game_state["board"]["height"]

def _is_head_colliding_with_self(game_state:dict, next_head:(int,int)) -> bool:
    return next_head in game_state["you"]["body"][:game_state["you"]["length"]-1]

def _is_head_colliding_with_other_snake(game_state:dict, next_head:(int,int)) -> dict | None: #return snake dict or None
    for snake in game_state["board"]["snakes"]:
        if next_head in snake["body"]:
            return snake
    return None

def _is_head_colliding_with_food(game_state:dict, next_head:(int,int)) -> bool:
    #TODO
    return False
