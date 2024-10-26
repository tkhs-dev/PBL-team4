from deap import creator, base


def train():
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

if __name__ == "__main__":
    train()