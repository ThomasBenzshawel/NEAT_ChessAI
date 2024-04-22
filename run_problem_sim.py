import numpy as np
import problem_sim_general as sim
import matplotlib.pyplot as plt
from organism import NEATOrganism
import numpy as np
import dill
from functools import partial
from multiprocessing import Pool
from organism import NEATOrganism

# Ecosystem and GA work

def pairwise(iterable):
    # s -> (s0, s1), (s2, s3), (s4, s5), ...
    a = iter(iterable)
    return zip(a, a)

class Ecosystem():
    def __init__(self, orginism_creator, scoring_function, population_size=100, holdout='sqrt', mating=True):
        """
        origanism_creator must be a function to produce Organisms, used for the original population
        scoring_function must be a function which accepts an Organism as input and returns a float
        """
        self.population_size = population_size

        self.population = [organism_creator() for _ in range(population_size)]
        self.mating = mating

        self.rewards = []

        self.scoring_function = scoring_function
        if holdout == 'sqrt':
            self.holdout = max(1, int(np.sqrt(population_size)))
        elif holdout == 'log':
            self.holdout = max(1, int(np.log(population_size)))
        elif holdout > 0 and holdout < 1:
            self.holdout = max(1, int(holdout * population_size))
        else:
            self.holdout = max(1, int(holdout))

    def generation(self, repeats=1, keep_best=True):
        self.rewards = [self.scoring_function(x, y) for x, y in pairwise(self.population)]
        self.rewards = [item for sublist in self.rewards for item in sublist]

        self.population = [self.population[x] for x in np.argsort(self.rewards)[::-1]]
        self.population_size = len(self.population)

        new_population = []
        for i in range(self.population_size):
            parent_1_idx = i % self.holdout

            if self.mating:
                parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
            else:
                parent_2_idx = parent_1_idx

            offspring = self.population[parent_1_idx].mate(self.population[parent_2_idx])
            new_population.append(offspring)

        if keep_best:
            new_population[-1] = self.population[0]  # Ensure best organism survives
        self.population = new_population


    
    def parallel_generation(self, repeats=1, keep_best=True):
        model_files = []
        for i, organism in enumerate(self.population):
            model_file = f"model_{i}.pkl"
            organism.save(model_file)
            model_files.append(model_file)

        with Pool() as pool:
            self.rewards = pool.starmap(self.scoring_function, pairwise(range(len(self.population))))
        self.rewards = [item for sublist in self.rewards for item in sublist]

        self.population = [self.population[x] for x in np.argsort(self.rewards)[::-1]]
        self.population_size = len(self.population)

        new_population = []
        for i in range(self.population_size):
            parent_1_idx = i % self.holdout

            if self.mating:
                parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
            else:
                parent_2_idx = parent_1_idx

            offspring = self.population[parent_1_idx].mate(self.population[parent_2_idx])
            new_population.append(offspring)

        if keep_best:
            new_population[-1] = self.population[0]  # Ensure best organism survives
        self.population = new_population

    def get_best_organism(self, include_reward=False):
        # rewards = [np.mean(self.scoring_function(x)) for _ in range(repeats) for x in self.population]
        if include_reward:
            best = np.argsort(self.rewards)[-1]
            return self.population[best], self.rewards[best]
        else:
            return self.population[np.argsort(self.rewards)[-1]]

def make_organism_generator(in_shape, out_shape):
    return lambda: NEATOrganism(in_shape, out_shape)

def run_generations(ecosystem, generations, parallel=False):
    print("Starting simulations")
    
    best_ai_list = []
    best_ai_models = []
    
    for i in range(generations):
        print("Starting generation ", i, " out of ", generations)
        print("Population size is: ", ecosystem.population_size)
        
        if parallel:
            ecosystem.parallel_generation()
        else:
            ecosystem.generation()
        
        best_ai = ecosystem.get_best_organism(include_reward=True)
        best_ai_models.append(best_ai[0])
        best_ai_list.append(best_ai[1])
        print("Best AI = ", best_ai[1])
        
        ecosystem.get_best_organism().save("model_new.pkl")
        
        fig, ax = plt.subplots()
        
        # Creating data
        x = [i for i in range(len(best_ai_list))]
        y = best_ai_list
        
        # Plotting barchart
        plt.plot(x, y)
        ax.set(xlabel='Generation', ylabel='Total Points Collected',
               title='Points Collected vs generations')
        ax.grid()
        
        # Saving the figure.
        plt.savefig("output_gen_" + str(i) + ".jpg")

if __name__ == '__main__':
    TF_ENABLE_ONEDNN_OPTS=0
    parrallel = True
    #Change this depending on the type of simulation
    organism_creator = make_organism_generator((384,), 1)

    def scoring_function_parallel(organism_1, organism_2):
        return sim.parallel_simulate_and_evaluate_organism(organism_1, organism_2, num_sims=1)

    if parrallel:
        scoring_function = scoring_function_parallel
    else:
        scoring_function = lambda organism_1, organism_2 : sim.simulate_and_evaluate_organism(organism_1, organism_2, num_sims=1)
    ecosystem = Ecosystem(organism_creator, scoring_function, population_size=4, holdout=0.1, mating=False)

    generations = 2
    run_generations(ecosystem, generations, parallel=True)