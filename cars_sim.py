import numpy as np
from organism import NEATOrganism
import pandas as pd


def data_generator(car_data_file, batch_size):
    car_data = pd.read_csv(car_data_file, chunksize=batch_size)
    for chunk in car_data:
        goal = chunk["sellingprice"].astype(float)
        chunk = chunk.drop(columns=["sellingprice"])
        X = np.array(chunk)
        y = goal.values
        yield X, y

def simulate_and_evaluate(organism_1, organism_2=None, print_game=False, trials=1, test=False, batch_size=1000):
    print("Starting car price simulation")

    organism_1.score = 0
    if organism_2 is not None:
        organism_2.score = 0

    if test:
        car_data_file = "data/car_data_test.csv"
    else:
        car_data_file = "data/car_data_train.csv"

    data_gen = data_generator(car_data_file, batch_size)
    for i in range(trials):
        X, y = next(data_gen)
        organism_1.score += -1 * np.sum(np.abs(y - organism_1.predict(X)))

        if organism_2 is not None:
            organism_2.score += -1 * np.sum(np.abs(y - organism_2.predict(X)))

    if not test:
        organism_1.score = organism_1.score / trials
        organism_2.score = organism_2.score / trials
        print("Organisms evaluated")
    else:
        print("Organism tested")
        return organism_1.score

    return [organism_1.score, organism_2.score]

def parallel_simulate_and_evaluate(organism_1_pkl, organism_2_pkl, num_sims=10, print_game=False):
    # print("Running parallel simulations with print_game = ", print_game)
    # print("Organism 1: ", organism_1_pkl)
    # print("Organism 2: ", organism_2_pkl)
    organism_1 = NEATOrganism.load(organism_1_pkl)
    organism_2 = NEATOrganism.load(organism_2_pkl)

    # print("organisms loaded")
    
    return simulate_and_evaluate(organism_1, organism_2, print_game=print_game, trials=num_sims)
