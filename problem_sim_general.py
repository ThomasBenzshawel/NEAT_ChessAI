import updated_chess_sim as chess_sim
from organism import Organism
import cars_sim

# The goal of this script is to provide a general framework for comparing the performance of organisms in a simulation

simulation_type = "chess"  # 'chess', 'cars'

# This function should run the simulation and compare the perfomance of the organism to the baseline or other organism
def simulate_and_evaluate_organism(organism_1, organism_2=None, num_sims=10):
    if simulation_type == 'chess':
        return chess_sim.simulate_and_evaluate(organism_1, organism_2, num_sims)
    elif simulation_type == 'cars':
        return cars_sim.simulate_and_evaluate(organism_1, organism_2, num_sims)