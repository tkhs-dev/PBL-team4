import os
import random
import sys
import time
from multiprocessing import Pool

import torch
from deap import creator, base
from deap import tools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from solo.sneak.evaluator import EvaluatorModel, Evaluator
from solo.sneak.player import AIPlayer
from solo.trainer.rules import start_solo_game, Client, GameSettings, CALLBACKFUNC


def init_weights():
    model = EvaluatorModel()
    # 個体のリストに重みを格納
    weights = []
    for param in model.parameters():
        weights.extend(param.data.numpy().flatten())
    return weights

def individual_to_model(individual):
    model = EvaluatorModel()
    start = 0
    for param in model.parameters():
        param_shape = param.data.shape
        param_size = param.data.numel()  # 要素数
        param_data = individual[start:start + param_size]
        param.data.copy_(torch.tensor(param_data).view(param_shape))
        start += param_size
    return model

def evaluate_single(individual):
    model = EvaluatorModel()
    # 重みをモデルにセット
    start = 0
    for param in model.parameters():
        param_shape = param.data.shape
        param_size = param.data.numel()  # 要素数
        param_data = individual[start:start + param_size]

        param.data.copy_(torch.tensor(param_data).view(param_shape))
        start += param_size

    evaluator = Evaluator()
    evaluator.model = model

    # ゲーム開始
    player = AIPlayer(evaluator)
    client = Client()
    client.on_move = lambda state: player.on_move(state)

    setting = GameSettings()
    setting.seed = random.randint(0, 100000)
    setting.width = 6
    setting.height = 6
    setting.food_spawn_chance = 0
    setting.minimum_food = 3

    result = start_solo_game(client, setting)
    score = result["turn"]
    if result["you"]["health"] <= 0:
        score /= 2
    if -10<=(result["turn"]+200) - result["you"]["length"]*100 < 0 and result["you"]["length"] > 5:
        score *= 2
    return score

pool = None

def evaluate(individual):
    results = pool.map(evaluate_single, [individual] * 5)
    return sum(results) / len(results),

def train():
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, init_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    population = toolbox.population(n=10)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 100
    print("Start of evolution")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(population))

    for g in range(NGEN):
        print("-- Generation %i --" % g)
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        population[:] = offspring
        fits = [ind.fitness.values[0] for ind in population]
        length = len(population)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    model = individual_to_model(best_ind)
    model.save("../evaluator.pth")

if __name__ == "__main__":
    train()