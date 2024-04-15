import updated_chess_sim as chess_sim
import cars_sim

# The goal of this script is to provide a general framework for comparing the performance of organisms in a simulation

simulation_type = "chess"  # 'chess', 'cars'

# This function should run the simulation and compare the perfomance of the organism to the baseline or other organism
def simulate_and_evaluate_organism(organism_1, organism_2=None, num_sims=10, objective_function=None):
    if simulation_type == 'chess':
        eval =  chess_sim.simulate_and_evaluate(organism_1, organism_2, num_sims)
        return objective_function(eval)
    elif simulation_type == 'cars':
        eval = cars_sim.simulate_and_evaluate(organism_1, organism_2, num_sims)
        return objective_function(eval)