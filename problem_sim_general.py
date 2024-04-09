import updated_chess_sim
from organism import Organism
import cars_sim

# The goal of this script is to provide a general framework for comparing the performance of organisms in a simulation

simulation_type = None  # 'chess', 'cars'

# This function should run the simulation and compare the perfomance of the organism to the baseline or other organism
def simulate_and_evaluate_organism(organism_1, organism_2=None, num_sims=10):
    raise NotImplementedError


# Set up prediction function
def predict(X):
    return np.random.rand(X.shape[0], 1)