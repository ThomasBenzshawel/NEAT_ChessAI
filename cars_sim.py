import chess
import numpy as np
from organism import NEATOrganism
import pandas as pd

def simulate_and_evaluate(organism_1, organism_2, print_game=False, trials=1):
    print("Starting simulation with ", trials, " trials")

    #load the data
    car_data = pd.read_csv('/data/car_prices.csv')

    #initialize the scores
    organism_1.score = 0
    organism_2.score = 0

    goal = car_data["sellingprice"].astype(float)
    car_data = car_data.drop(columns=["sellingprice", "car_ID", "vin", "seller"])
    #evaluate the organisms in estimating car prices
    for i in range(trials):
        predicted_price_1 = organism_1.predict(car_data)

    return [organism_1.score / trials, organism_2.score / trials]

def parallel_simulate_and_evaluate(organism_1_pkl, organism_2_pkl, num_sims=10, print_game=False):
    print("Running parallel simulations inside chess_sim.py, with print_game = ", print_game)
    print("Organism 1: ", organism_1_pkl)
    print("Organism 2: ", organism_2_pkl)
    organism_1 = NEATOrganism.load(organism_1_pkl)
    organism_2 = NEATOrganism.load(organism_2_pkl)

    print("organisms loaded")
    
    return simulate_and_evaluate(organism_1, organism_2, print_game=print_game, trials=num_sims)
