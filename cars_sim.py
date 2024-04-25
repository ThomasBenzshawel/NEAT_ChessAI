import numpy as np
from organism import NEATOrganism
import pandas as pd


# Input shape must be 1269

def simulate_and_evaluate(organism_1, organism_2, print_game=False, trials=1):
    print("Starting car price simulation")

    #load the data
    car_data = pd.read_csv('./data/car_prices_processed.csv')
    #only use half of the data
    car_data = car_data.sample(frac=0.2)

    organism_1.score = 0
    organism_2.score = 0

    # print("Data loaded")
    goal = car_data["sellingprice"].astype(float)
    # print("Goal loaded")
    car_data = car_data.drop(columns=["sellingprice"])

    X = np.array(car_data)
    # print("Data converted to numpy")

    y = goal.values

    # print("Data split")

    #evaluate the organisms in estimating car prices

    #trials is actually the number of batches

    #split the data into batches of size len(X)/trials

    batch_size = len(X)//trials

    for i in range(trials):
        start = i*batch_size
        end = (i+1)*batch_size

        organism_1.score += -1 * np.sum(np.abs(y[start:end] - organism_1.predict(X[start:end])))
        organism_2.score += -1 * np.sum(np.abs(y[start:end] - organism_2.predict(X[start:end])))

    print("Organisms evaluated")
    return [organism_1.score, organism_2.score]

def parallel_simulate_and_evaluate(organism_1_pkl, organism_2_pkl, num_sims=10, print_game=False):
    # print("Running parallel simulations with print_game = ", print_game)
    # print("Organism 1: ", organism_1_pkl)
    # print("Organism 2: ", organism_2_pkl)
    organism_1 = NEATOrganism.load(organism_1_pkl)
    organism_2 = NEATOrganism.load(organism_2_pkl)

    # print("organisms loaded")
    
    return simulate_and_evaluate(organism_1, organism_2, print_game=print_game, trials=num_sims)
