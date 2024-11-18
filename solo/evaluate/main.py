import copy
import multiprocessing
import os
import random
import sys
from multiprocessing import Pool
from statistics import mean

import torch
from deap import creator, base
from deap import tools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from solo.sneak.evaluator import EvaluatorModel, Evaluator
from solo.sneak.player import AIPlayer
from solo.trainer.rules import start_solo_game, Client, GameSettings
from shared.rule import Direction

def move_callback(game_state, player)->Direction:
    move = player.on_move(game_state)
    return move

def evaluate(seed_model):
    evaluator, seed = seed_model
    player = AIPlayer(evaluator)
    client = Client()
    client.on_move = player.on_move

    setting = GameSettings()
    setting.seed = seed
    # 盤面を狭めて学習させてみるのも良いかもしれない
    setting.width = 6
    setting.height = 6
    setting.food_spawn_chance = 0
    setting.minimum_food = 3

    result = start_solo_game(client, setting)
    return result


if __name__ == "__main__":
    pool = Pool(multiprocessing.cpu_count())

    #カレントの.pthすべてについて
    for path in os.listdir("."):
        if path.endswith(".pth"):
            evaluator = Evaluator.load(path=path)
            seed_set = random.sample(range(1000000000), 500)
            results = pool.map(evaluate, [(evaluator, seed) for seed in seed_set])
            turns = map(lambda x: x["turn"], results)
            length = map(lambda x: x["you"]["length"], results)

            print("--------------------------------")
            print("Path: ", path)
            print("Result Length Avg: ", mean(length))
            print("Result Turns Avg: ", mean(copy.deepcopy(turns)))
            print("Result Turns Min: ", min(copy.deepcopy(turns)))
            print("Result Turns Max: ", max(copy.deepcopy(turns)))




