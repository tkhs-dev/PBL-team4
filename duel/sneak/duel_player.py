import os
import random
import sys
from copy import deepcopy

from duel.sneak.duel_evaluator import Evaluator
from shared.rule import Direction, move, TurnResult, get_reachable_cells, is_move_maybe_safe, dict_coord_to_tuple, \
    get_distance_to_opponent, get_opponent_snake, get_nearest_foods, is_move_do_nothing, _get_next_head, \
    is_move_dangerous, _get_opponent_next_head

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from shared.player import IPlayer

class AIPlayer(IPlayer):
    def __init__(self, evaluator:Evaluator, safe:bool = False):
        self.evaluator = evaluator
        self.safe = safe

    def trap(self, game_state) -> Direction:
        you = game_state['you']
        opponent = list(filter(lambda x:x['id']!=you['id'], game_state['board']['snakes']))[0]
        if opponent['head']['x'] == 0:
            if you['head']['x'] == 1:
                if {"x":1,"y":opponent['head']['y']} in you['body']:
                    if you['head']['y'] > opponent['head']['y']:
                        return Direction.UP
                    else:
                        return Direction.DOWN
                elif you['length'] > opponent['length']:
                    if {"x":1,"y":opponent['head']['y']} in you['body']:
                        if you['head']['y'] > opponent['head']['y']:
                            return Direction.UP
                        else:
                            return Direction.DOWN

        elif opponent['head']['x'] == 10:
            if you['head']['x'] == 9:
                if {"x":9,"y":opponent['head']['y']} in you['body']:
                    if you['head']['y'] > opponent['head']['y']:
                        return Direction.UP
                    else:
                        return Direction.DOWN
        elif opponent['head']['y'] == 0:
            if you['head']['y'] == 1:
                if {"x":opponent['head']['x'],"y":1} in you['body']:
                    if you['head']['x'] > opponent['head']['x']:
                        return Direction.RIGHT
                    else:
                        return Direction.LEFT
        elif opponent['head']['y'] == 10:
            if you['head']['y'] == 9:
                if {"x":opponent['head']['x'],"y":9} in you['body']:
                    if you['head']['x'] > opponent['head']['x']:
                        return Direction.RIGHT
                    else:
                        return Direction.LEFT



    def on_start(self, game_state):
        return 0

    def on_end(self, game_state):
        return 0

    def on_move(self, game_state):
        if len(game_state['board']['snakes']) != 2:
            return Direction.UP
        q = self.evaluator.evaluate(game_state)

        # game_stateの内容がどこかで変わってしまうため,最後で使うために元のものを保存 修正が必要
        original_game_state = deepcopy(game_state)

        # 早期の引き分けを防止
        if game_state['turn'] < 20:
            opponent = get_opponent_snake(game_state)
            if game_state['you']['length'] <= opponent['length']:
                if get_distance_to_opponent(game_state) <= 3:
                    your_nearest_foods = get_nearest_foods(game_state, dict_coord_to_tuple(game_state['you']['head']), 1)
                    opponent_nearest_foods = get_nearest_foods(game_state, dict_coord_to_tuple(opponent['head']), 1)
                    if your_nearest_foods and opponent_nearest_foods:
                        compete_foods = set(your_nearest_foods) & set(opponent_nearest_foods)
                        print('compete_foods',compete_foods)
                        if len(compete_foods) > 0:
                            # 競合するエサを食べる以外の行動を取る
                            print("餌かぶってる")
                            suggest = list(filter(lambda x: is_move_do_nothing(game_state, x), Direction))
                            if len(suggest) > 0:
                                print(game_state['turn'],"回避")
                                safe_moves = list(filter(lambda x: not is_move_dangerous(game_state, x), suggest))
                                if len(safe_moves) > 0:
                                    return random.choice(suggest)
                                return random.choice(suggest)

        nexts = list(zip(Direction,map(lambda x:move(game_state,x), Direction)))
        killable = []
        trap = self.trap(game_state)
        if trap:
            if is_move_maybe_safe(game_state, trap):
                print('TRAP ACTION',trap)
                return trap
        for nx in nexts:
            if nx[1][0] == TurnResult.WIN:
                killable.append(nx[0])
                continue
            if nx[1][1] is None:
                continue
            if len(nx[1][1]['board']['snakes']) != 2:
                continue
            opponent = list(filter(lambda x:x['id']!=game_state['you']['id'], nx[1][1]['board']['snakes']))[0]
            if len(get_reachable_cells(nx[1][1], dict_coord_to_tuple(opponent['head']), 7))<= 12:
                print(game_state['turn'],'KILL ACTION')
                # 殺される可能性がある場合は回避
                if game_state['you']['length'] < opponent['length']:
                    next_head = _get_next_head(game_state, nx[0])
                    if any(map(lambda x: _get_opponent_next_head(game_state, x) == next_head,Direction)):
                        print('殺される可能性ありなので断念')
                        continue
                killable.append(nx)

        if len(killable) > 0:
            return max(killable, key = lambda x:len(get_reachable_cells(x[1][1], dict_coord_to_tuple(x[1][1]['you']['head']), 7)))[0]

        if not self.safe:
            print(game_state['turn'],list(map(lambda x:'{:.2f}'.format(x), q)))
            direction = max(zip(Direction,q), key = lambda x:x[1])[0]
            return direction
        candidate = list(filter(lambda x: not is_move_dangerous(game_state, x[0]), zip(Direction,q)))
        game_state = original_game_state
        if len(candidate) == 0:
            safes = list(filter(lambda x: x[0][1][1] is not None and len(get_reachable_cells(x[0][1][1], dict_coord_to_tuple(x[0][1][1]['you']['head']), 7)) >= 7, zip(nexts, q)))

            if len(safes) == 0:
                safes = list(filter(lambda x: x[0][1][0] == TurnResult.CONTINUE, zip(nexts, q)))
                if len(safes) == 0:
                    return max(zip(Direction,q), key = lambda x:x[1])[0]

            direction = max(safes, key = lambda x:x[1])[0][0]
        else:
            nx = list(map(lambda x:(move(game_state, x[0]), x[1], x[0]), candidate))
            safes = list(filter(lambda x: x[0][1] is not None and len(get_reachable_cells(x[0][1], dict_coord_to_tuple(x[0][1]['you']['head']), 7)) >= 7, nx))

            if len(safes) == 0:
                return max(candidate, key = lambda x:x[1])[0]

            direction = max(safes, key = lambda x:x[1])[2]
        return direction