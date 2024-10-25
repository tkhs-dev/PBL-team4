#　必要なパラメーターを取得する関数を定義する
# example:
#  def get_snake_length(game_state: dict) -> int:
#      return game_state["you"]["length"]

def get_up_body(game_state: dict) -> int:
    r =0
    head = game_state["you"]["head"]
    body = ["data"]["body"]
    check_point1 = {"x": head["x"], "y": head["y"] - 1}
    check_point2 = {"x": head["x"], "y": head["y"] - 2}

    if check_point1 in body:
        r += 1
    if check_point2 in body:
        r += 2
    return r

def get_down_body(game_state: dict) -> int:
    r =0
    head = game_state["you"]["head"]
    body = ["data"]["body"]
    check_point1 = {"x": head["x"], "y": head["y"] + 1}
    check_point2 = {"x": head["x"], "y": head["y"] + 2}

    if check_point1 in body:
        r += 1
    if check_point2 in body:
        r += 2
    return r

def get_right_body(game_state: dict) -> int:
    r =0
    head = game_state["you"]["head"]
    body = ["data"]["body"]
    check_point1 = {"x": head["x"] + 1 , "y": head["y"]}
    check_point2 = {"x": head["x"] + 2 , "y": head["y"]}

    if check_point1 in body:
        r += 1
    if check_point2 in body:
        r += 2
    return r

def get_left_body(game_state: dict) -> int:
    r =0
    head = game_state["you"]["head"]
    body = ["data"]["body"]
    check_point1 = {"x": head["x"] - 1 , "y": head["y"]}
    check_point2 = {"x": head["x"] - 2 , "y": head["y"]}

    if check_point1 in body:
        r += 1
    if check_point2 in body:
        r += 2
    return r