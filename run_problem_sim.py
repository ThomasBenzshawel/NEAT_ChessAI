import numpy as np
import problem_sim_general as sim
import matplotlib.pyplot as plt
from organism import NEATOrganism
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np

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

    def mp_generation(_self_, _repeats_=1, _keep_best_=True):
        # make the population and score stuff array
        population = np.array(_self_.population)  # parameters to send to simulate_and_evaluate_

        n = population.shape[0]
        num_processes = 4  # number of processes to use

        # create a pool of worker processes
        pool = Pool(processes=num_processes)

        # split the population into chunks for each process
        chunk_size = n // num_processes
        population_chunks = [population[i:i + chunk_size] for i in range(0, n, chunk_size)]

        # run the scoring function for each chunk of the population in parallel
        results = pool.starmap(_self_.scoring_function, [(x, y) for chunk in population_chunks for x, y in pairwise(chunk)])

        # flatten the results list
        final_results = [item for sublist in results for item in sublist]
        print (final_results)

        _self_.rewards = [x.score for x in _self_.population]
        _self_.population = [_self_.population[x] for x in np.argsort(_self_.rewards)[::-1]]
        _self_.population_size = len(_self_.population)

        new_population = []

        for i in range(_self_.population_size):
            parent_1_idx = i % _self_.holdout
            # print(parent_1_idx)
            if _self_.mating:
                parent_2_idx = min(_self_.population_size - 1, int(np.random.exponential(_self_.holdout)))
            else:
                parent_2_idx = parent_1_idx

            offspring = _self_.population[parent_1_idx].mate(_self_.population[parent_2_idx])
            new_population.append(offspring)

        if _keep_best_:
            new_population[-1] = _self_.population[0]  # Ensure best organism survives

        _self_.population = new_population

        # close the pool of worker processes
        pool.close()
        pool.join()

    def get_best_organism(self, include_reward=False):
        # rewards = [np.mean(self.scoring_function(x)) for _ in range(repeats) for x in self.population]
        if include_reward:
            best = np.argsort(self.rewards)[-1]
            return self.population[best], self.rewards[best]
        else:
            return self.population[np.argsort(self.rewards)[-1]]

def make_organism_generator(in_shape, out_shape):
    return lambda: NEATOrganism(in_shape, out_shape)


def run_generations(ecosystem, generations):
    print("Starting simulations")
    
    best_ai_list = []
    best_ai_models = []
    
    for i in range(generations):
        print("Starting generation ", i, " out of ", generations)
        print("Population size is: ", ecosystem.population_size)
        
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
    
    #Change this depending on the type of simulation
    organism_creator = make_organism_generator((384,), (1,))

    scoring_function = lambda organism_1, organism_2 : sim.simulate_and_evaluate_organism(organism_1, organism_2, num_sims=10, objective_function = lambda x: x)
    ecosystem = Ecosystem(organism_creator, scoring_function, population_size=40, holdout=0.1, mating=True)

    generations = 15
    run_generations(ecosystem, generations)