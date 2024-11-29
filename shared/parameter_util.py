#　必要なパラメーターを取得する関数を定義
# example:
#  def get_snake_length(game_state: dict) -> int:
#      return game_state["you"]["length"]
import copy


def get_front_body(game_state: dict) -> int:
    r = 0
    head = game_state["you"]["head"]
    body = game_state["you"]["body"]
    neck = game_state["you"]["body"][1]
    check_point1 = {}
    check_point2 = {}
    if neck == {"x": head["x"] - 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] + 1 , "y": head["y"]}
        check_point2 = {"x": head["x"] + 2 , "y": head["y"]}
    elif neck == {"x": head["x"] + 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] - 1 , "y": head["y"]}
        check_point2 = {"x": head["x"] - 2 , "y": head["y"]}
    elif neck == {"x": head["x"] , "y": head["y"] - 1}:
        check_point1 = {"x": head["x"] , "y": head["y"] + 1}
        check_point2 = {"x": head["x"] , "y": head["y"] + 2}
    elif neck == {"x": head["x"] , "y": head["y"] + 1}:
        check_point1 = {"x": head["x"] , "y": head["y"] - 1}
        check_point2 = {"x": head["x"] , "y": head["y"] - 2}

    if check_point1 in body:
        r += 2
    if check_point2 in body:
        r += 1
    return r

def get_right_body(game_state: dict) -> int:
    r = 0
    head = game_state["you"]["head"]
    body = game_state["you"]["body"]
    neck = game_state["you"]["body"][1]
    check_point1 = {}
    check_point2 = {}
    if neck == {"x": head["x"] - 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] , "y": head["y"] - 1}
        check_point2 = {"x": head["x"] , "y": head["y"] - 2}
    elif neck == {"x": head["x"] + 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] , "y": head["y"] + 1}
        check_point2 = {"x": head["x"] , "y": head["y"] + 2}
    elif neck == {"x": head["x"] , "y": head["y"] - 1}:
        check_point1 = {"x": head["x"] + 1, "y": head["y"]}
        check_point2 = {"x": head["x"] + 2, "y": head["y"]}
    elif neck == {"x": head["x"] , "y": head["y"] + 1}:
        check_point1 = {"x": head["x"] - 1, "y": head["y"]}
        check_point2 = {"x": head["x"] - 2, "y": head["y"]}

    if check_point1 in body:
        r += 2
    if check_point2 in body:
        r += 1
    return r

def get_left_body(game_state: dict) -> int:
    r = 0
    head = game_state["you"]["head"]
    body = game_state["you"]["body"]
    neck = game_state["you"]["body"][1]
    check_point1 = {}
    check_point2 = {}
    if neck == {"x": head["x"] - 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] , "y": head["y"] + 1}
        check_point2 = {"x": head["x"] , "y": head["y"] + 2}
    elif neck == {"x": head["x"] + 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] , "y": head["y"] - 1}
        check_point2 = {"x": head["x"] , "y": head["y"] - 2}
    elif neck == {"x": head["x"] , "y": head["y"] - 1}:
        check_point1 = {"x": head["x"] - 1, "y": head["y"]}
        check_point2 = {"x": head["x"] - 2, "y": head["y"]}
    elif neck == {"x": head["x"] , "y": head["y"] + 1}:
        check_point1 = {"x": head["x"] + 1, "y": head["y"]}
        check_point2 = {"x": head["x"] + 2, "y": head["y"]}

    if check_point1 in body:
        r += 2
    if check_point2 in body:
        r += 1
    return r

def get_leftd_body(game_state: dict) -> int:
    r = 0
    head = game_state["you"]["head"]
    body = game_state["you"]["body"]
    neck = game_state["you"]["body"][1]
    check_point1 = {}
    check_point2 = {}
    if neck == {"x": head["x"] - 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] + 1 , "y": head["y"] + 1}
        check_point2 = {"x": head["x"] + 2 , "y": head["y"] + 1}
    elif neck == {"x": head["x"] + 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] - 1 , "y": head["y"] - 1}
        check_point2 = {"x": head["x"] - 2 , "y": head["y"] - 1}
    elif neck == {"x": head["x"] , "y": head["y"] - 1}:
        check_point1 = {"x": head["x"] - 1, "y": head["y"] + 1}
        check_point2 = {"x": head["x"] - 1, "y": head["y"] + 2}
    elif neck == {"x": head["x"] , "y": head["y"] + 1}:
        check_point1 = {"x": head["x"] + 1, "y": head["y"] - 1}
        check_point2 = {"x": head["x"] + 1, "y": head["y"] - 2}

    if check_point1 in body:
        r += 2
    if check_point2 in body:
        r += 1
    return r

def get_rightd_body(game_state: dict) -> int:
    r = 0
    head = game_state["you"]["head"]
    body = game_state["you"]["body"]
    neck = game_state["you"]["body"][1]
    check_point1 = {}
    check_point2 = {}
    if neck == {"x": head["x"] - 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] + 1 , "y": head["y"] - 1}
        check_point2 = {"x": head["x"] + 2 , "y": head["y"] - 1}
    elif neck == {"x": head["x"] + 1 , "y": head["y"]}:
        check_point1 = {"x": head["x"] - 1 , "y": head["y"] + 1}
        check_point2 = {"x": head["x"] - 2 , "y": head["y"] + 1}
    elif neck == {"x": head["x"] , "y": head["y"] - 1}:
        check_point1 = {"x": head["x"] + 1, "y": head["y"] + 1}
        check_point2 = {"x": head["x"] + 1, "y": head["y"] + 2}
    elif neck == {"x": head["x"] , "y": head["y"] + 1}:
        check_point1 = {"x": head["x"] - 1, "y": head["y"] - 1}
        check_point2 = {"x": head["x"] - 1, "y": head["y"] - 2}

    if check_point1 in body:
        r += 2
    if check_point2 in body:
        r += 1
    return r

#餌との最短距離
def get_snake_foods(game_state: dict) -> int:
    distance = game_state["board"]["width"] + game_state["board"]["height"]
    head = game_state["you"]["head"]
    flg = False
    for food in game_state["board"]["food"]:
        body = copy.deepcopy(game_state["you"]["body"])
        dx = food["x"] - head["x"]
        dy = food["y"] - head["y"]
        for x in range(0, dx-1, (1 > dx-1)*-2+1):
            if { "x": head["x"] + x, "y": head["y"] } == head:
                continue
            if len(body) == 1:
                break
            body.pop()
            if { "x": head["x"] + x, "y": head["y"] } in body:
                flg = True
                break
        if not flg:
            for y in range(0, dy-1, (1 > dy-1)*-2+1):
                if { "x": head["x"] + dx, "y": head["y"] + y } == head:
                    continue
                if len(body) == 1:
                    break
                body.pop()
                if { "x": head["x"] + dx, "y": head["y"] + y } in body:
                    flg = True
                    break
        body = copy.deepcopy(game_state["you"]["body"])
        if not flg:
            distance = min(distance, abs(dx) + abs(dy))
        else:
            flg = False
            for y in range(0, dy-1, (1 > dy-1)*-2+1):
                if { "x": head["x"], "y": head["y"] + y } == head:
                    continue
                if len(body) == 1:
                    break
                body.pop()
                if { "x": head["x"], "y": head["y"] + y } in body:
                    flg = True
                    break
            if not flg:
                for x in range(0, dx-1, (1 > dx-1)*-2+1):
                    if { "x": head["x"] + x, "y": head["y"] + dy } == head:
                        continue
                    if len(body) == 1:
                        break
                    body.pop()
                    if { "x": head["x"] + x, "y": head["y"] +dy } in body:
                        flg = True
                        break
            if not flg:
                distance = min(distance, abs(dx) + abs(dy))

    return distance




#壁との距離
def get_snake_distance(game_state: dict) -> int:
    coordinate_x = game_state["you"]["head"]["x"]
    coordinate_y = game_state["you"]["head"]["y"]
    x_distance = game_state["board"]["width"] + 1 - coordinate_x
    y_distance = game_state["board"]["height"] + 1 - coordinate_y
    min_distance = min(coordinate_x, coordinate_y, x_distance, y_distance)
    return min_distance



#蛇の長さ
def get_snake_length(game_state: dict) -> int:
    length = game_state["you"]["length"]

    return length

#ヘルス
def get_snake_health(game_state: dict) -> int:
    health = game_state["you"]["health"]

    return health

def get_free_space(game_state: dict) -> int:
    free_space = 0
    head = game_state["you"]["head"]
    front = {"x": head["x"], "y": head["y"] - 1}
    right = {"x": head["x"] + 1, "y": head["y"]}
    left = {"x": head["x"] - 1, "y": head["y"]}
    back = {"x": head["x"], "y": head["y"] + 1}
    if front["y"] >= 0 and front not in game_state["you"]["body"] and front not in game_state["board"]["food"]:
        free_space += 1

    if right["x"] < game_state["board"]["width"] and right not in game_state["you"]["body"] and right not in game_state["board"]["food"]:
        free_space += 1

    if left["x"] >= 0 and left not in game_state["you"]["body"] and left not in game_state["board"]["food"]:
        free_space += 1

    if back["y"] < game_state["board"]["height"] and back not in game_state["you"]["body"] and back not in game_state["board"]["food"]:
        free_space += 1

    return free_space
