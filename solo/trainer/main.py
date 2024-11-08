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
from solo.trainer.rules import start_solo_game, Client, GameSettings
from shared import rule
from shared.rule import Direction


def init_weights():
    model = EvaluatorModel()
    return get_weights(model)

def get_weights(model):
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

score = 0
def move_callback(game_state, player)->Direction:
    move = player.on_move(game_state)
    next_state = rule.move(game_state, Direction.UP)[1]
    if next_state is None:
        return move
    global score
    # ==必要に応じてターンごとにスコアを加算する==
    if next_state["you"]["health"] == 100 and game_state["you"]["health"] == 1:
        score += 50 * next_state["you"]["length"] # エサを体力1で食べた場合に長さx50のスコアを加算(ぎりぎりでエサを食べてほしいから)
    return move

def evaluate_single(individual):
    global score
    score = 0
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
    client.on_move = lambda state: move_callback(state, player)

    setting = GameSettings()
    setting.seed = random.randint(0, 100000)
    # 盤面を狭めて学習させてみるのも良いかもしれない
    setting.width = 6
    setting.height = 6
    setting.food_spawn_chance = 0
    setting.minimum_food = 3

    result = start_solo_game(client, setting)

    # 以下で最終結果に応じてスコアの加算を行う
    score += result["turn"] # ターン数をスコアに加算
    return score

pool = None

def evaluate(individual):
    # 1つの個体に対してN回評価を行い、その平均値を返す.大きくすると学習が安定するが時間がかかる.
    N = 15

    results = pool.map(evaluate_single, [individual] * N)
    return sum(results) / len(results),

path = None

def train(init_weights = init_weights):
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, init_weights)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    # 以下の３種類(交叉,突然変異,選択)のメソッドを変更してみると学習がうまくいくかも(@see https://deap.readthedocs.io/en/master/api/tools.html)
    # 学習の進行度合いに応じてメソッドを変更すると学習を適切に進められるらしい(by ChatGPT)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # １つの世代の個体数 50ぐらいが良い？ 学習にかかる時間に関わるので小さすぎると悪い
    population = toolbox.population(n=10)

    CXPB, MUTPB, NGEN = 0.5, 0.2, 50 # 交叉、突然変異、世代数 (CXとMUTは上で設定した値と同じにしておくべき,世代数は50ぐらい？ population x generationが1500~3000程度が良いらしい by ChatGPT)
    print("Start of evolution")
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(population))

    last_best = 0
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
        if last_best < max(fits):
            last_best = max(fits)
            best_ind = tools.selBest(population, 1)[0]
            model = individual_to_model(best_ind)
            model.save(path+"evaluator-%d_%d.pth" % (g, last_best))

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
    if not os.path.exists("../pth/"):
        os.makedirs("../pth/")
    os.mkdir("../pth/"+str(int(time.time())))
    path = "../pth/"+str(int(time.time()))+"/"
    pool = Pool(4)
    if len(sys.argv) > 1:
        parent_path = sys.argv[1]
        train(init_weights=lambda: get_weights(Evaluator.load(path=parent_path).model))
    else:
        train()