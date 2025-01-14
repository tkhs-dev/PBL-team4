import copy
from enum import Enum


class TurnResult(Enum):
    CONTINUE = 0
    WIN = 1
    LOSE = 2
    DRAW = 3

class Direction(str,Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    def get_direction_pair(self) -> (int,int):
        if self == Direction.UP:
            return 0,1
        elif self == Direction.DOWN:
            return 0,-1
        elif self == Direction.LEFT:
            return -1,0
        elif self == Direction.RIGHT:
            return 1,0

    @staticmethod
    def index(direction) -> int:
        if direction == Direction.DOWN:
            return 1
        elif direction == Direction.LEFT:
            return 2
        elif direction == Direction.RIGHT:
            return 3
        else:
            return 0

    @staticmethod
    def by_index(index):
        if index == 1:
            return Direction.DOWN
        elif index == 2:
            return Direction.LEFT
        elif index == 3:
            return  Direction.RIGHT
        else:
            return Direction.UP

def move(game_state:dict, direction:Direction) -> (TurnResult, dict):
    #------ IMPORTANT ------
    #pythonのdictは参照渡しであるため、game_stateを変更すると元のgame_stateも変更される
    #そのため,game_stateの値を変更することは禁止
    #game_stateは,元の盤面の状態を取得したいときに使用する
    #次の状態に変更するなどの場合は,以下のnext_stateを変更し,戻り値として返すこと
    next_state = copy.deepcopy(game_state)
    next_head = _get_next_head(game_state, direction)

    next_state["turn"] += 1

    if _is_head_colliding_with_food(game_state, next_head):
        #餌に接触した場合の処理
        next_state["you"]["health"] = 100
        next_state["you"]["length"] += 1
        next_state["you"]["head"] = next_head ##蛇の頭を更新
        next_state["you"]["body"].insert(0, next_head) #蛇の頭を更新
        next_state["board"]["food"].remove(next_head)
    else:
        #何も接触しなかった場合の処理
        next_state["you"]["health"] -= 1
        next_state["you"]["head"] = next_head #蛇の頭を更新
        next_state["you"]["body"].pop()   #長さは変化しない、蛇が前に進む
        next_state["you"]["body"].insert(0, next_head) #蛇の頭を更新
        if next_state["you"]["health"] == 0:
            return TurnResult.LOSE, None

    #他の蛇の情報を更新
    for snake in next_state["board"]["snakes"]:
        if snake["id"] == next_state["you"]["id"]:
            continue
        if snake["body"][-1] == snake["head"]:
            snake["body"].pop()
        else:
            snake["body"].pop()
            snake["body"].insert(0,snake["head"])
        snake["head"] = snake["body"][0]
        snake["health"] -= 1
        if snake["health"] == 0:
            next_state["board"]["snakes"].remove(snake)

    #何らかの衝突があるかどうかを判定し,それに応じて処理、結果を返す
    if _is_head_out_of_bounds(game_state, next_head):
        return TurnResult.LOSE, None
    elif _is_head_colliding_with_self(game_state, next_head):
        return TurnResult.LOSE, None
    elif (snake := _is_head_colliding_with_other_snake(game_state, next_head)) is not None:
        #snakeには衝突したsnakeのdictが入る
        #もし接触したのが他の蛇の頭の場合,蛇の長さに応じて勝敗が変わる
        #体に接触した場合はこちらの負け
        if next_head == snake["head"]:
            if game_state["you"]["length"] > game_state["snake"]["length"]:
                return TurnResult.WIN,None
            elif game_state["you"]["length"] == game_state["snake"]["length"]:
                return TurnResult.DRAW,None
            elif game_state["you"]["length"] < game_state["snake"]["length"]:
                return TurnResult.LOSE,None
        elif next_head in snake["body"]:
            return TurnResult.LOSE,None

    return TurnResult.CONTINUE, next_state

def is_move_maybe_safe(game_state:dict, direction:Direction) -> bool:
    next_head = _get_next_head(game_state, direction)

    if _is_head_out_of_bounds(game_state, next_head):
        return False
    elif _is_head_colliding_with_food(game_state, next_head):
        return True
    elif _is_head_colliding_with_self(game_state, next_head):
        return False
    elif _is_head_colliding_with_other_snake(game_state, next_head, True) is not None:
        return False

def _get_next_head(game_state:dict, direction:Direction) -> dict:
    head = copy.deepcopy(game_state["you"]["head"])
    dx,dy = direction.get_direction_pair()
    head["x"] += dx
    head["y"] += dy
    return head
def _is_head_out_of_bounds(game_state:dict, next_head) -> bool:
    return next_head["x"] < 0 or next_head["x"] >= game_state["board"]["width"] or next_head["y"] < 0 or next_head["y"] >= game_state["board"]["height"]

def _is_head_colliding_with_self(game_state:dict, next_head) -> bool:
    return next_head in game_state["you"]["body"][:-1]

def _is_head_colliding_with_other_snake(game_state:dict, next_head, loose=False) -> dict | None: #return snake dict or None
    if loose:
        for snake in game_state["board"]["snakes"]:
            if snake["body"]:
                snake["body"].pop()
    for snake in game_state["board"]["snakes"]:
        if next_head in snake["body"] and snake["id"] != game_state["you"]["id"]:
            return snake
    return None

def _is_head_colliding_with_food(game_state:dict, next_head) -> bool:
    return next_head in game_state["board"]["food"]
