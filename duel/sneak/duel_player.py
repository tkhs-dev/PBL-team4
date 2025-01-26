import os
import sys

from duel.sneak.duel_evaluator import Evaluator
from shared.rule import Direction, move, TurnResult, get_reachable_cells, is_move_maybe_safe, dict_coord_to_tuple

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
            if len(get_reachable_cells(nx[1][1], dict_coord_to_tuple(opponent['head']), 7)) <= 12:
                print(game_state['turn'],'KILL ACTION')
                killable.append(nx[0])

        if len(killable) > 0:
            max(killable, key = lambda x:len(get_reachable_cells(game_state, dict_coord_to_tuple(game_state['you']['head']), 7)))

        if not self.safe:
            print(game_state['turn'],list(map(lambda x:'{:.2f}'.format(x), q)))
            direction = max(zip(Direction,q), key = lambda x:x[1])[0]
            return direction
        safes = list(filter(lambda x: x[0][1][1] is not None and len(get_reachable_cells(x[0][1][1], dict_coord_to_tuple(x[0][1][1]['you']['head']), 7)) >= 7, zip(nexts, q)))

        if len(safes) == 0:
            safes = list(filter(lambda x: x[0][1][0] == TurnResult.CONTINUE, zip(nexts, q)))
            if len(safes) == 0:
                return max(zip(Direction,q), key = lambda x:x[1])[0]
        direction = max(safes, key = lambda x:x[1])[0][0]
        return direction