import copy
import numpy as np
import pickle
from mpi4py import MPI
import simulate_and_evaluate as sim
import matplotlib.pyplot as plt
from simulate_and_evaluate import Organism

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
        # print("Before flatten, ", self.rewards)
        self.rewards = [item for sublist in self.rewards for item in sublist]
        # print("After flatten, ", self.rewards)

        self.population = [self.population[x] for x in np.argsort(self.rewards)[::-1]]
        self.population_size = len(self.population)

        #         self.population_new = [pop for pop in self.population if pop.winner]

        #         if(len(self.population_new) > 2):
        #             self.population = self.population_new
        #             self.population_size = len(self.population)

        new_population = []
        for i in range(self.population_size):
            parent_1_idx = i % self.holdout
            # print(parent_1_idx)

            if self.mating:
                parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
            else:
                parent_2_idx = parent_1_idx
            offspring = self.population[parent_1_idx].mate(self.population[parent_2_idx])
            new_population.append(offspring)
        if keep_best:
            new_population[-1] = self.population[0]  # Ensure best organism survives
        self.population = new_population
        # return -1 * max(rewards)

    def mpi_generation(self, repeats=1, keep_best=True):

        # make the population and score stuff array
        population = np.array(self.population)  # parameters to send to simulate_and_evaluate
        n = population.shape[0]


        count = n // size  # number of catchments for each process to analyze
        remainder = n % size  # extra catchments if n is not a multiple of size

        if rank < remainder:  # processes with rank < remainder analyze one extra catchment
            start = rank * (count + 1)  # index of first catchment to analyze
            stop = start + count + 1  # index of last catchment to analyze
        else:
            start = rank * count + remainder
            stop = start + count

        local_pop = population[start:stop]  # get the portion of the array to be analyzed by each rank
        # run the function for each parameter set and rank
        local_results = [self.scoring_function(x, y) for x, y in pairwise(local_pop)]
        local_results = [item for sublist in local_results for item in sublist]

        # Testing stuff
        # print("Len of local pop: ", len(local_pop))
        # print("Local results: ", local_results)

        if rank > 0:
            comm.isend(local_results, dest=0, tag=14)  # send results to process 0
        else:
            final_results = np.copy(local_results)  # initialize final results with results from process 0
            for i in range(1, size):  # determine the size of the array to be received from each process

                tmp = np.empty(len(final_results))  # create empty array to receive results
                comm.irecv(tmp, source=i, tag=14)  # receive results from the process
                final_results = np.hstack((final_results, tmp))  # add the received results to the final results

                # print("results")
                # print(final_results)

                # More testing
                # Saving the value pairings in self.rewards
                # self.rewards = [item for sublist in final_results for item in sublist]
                # print("#################")
                # print(self.rewards, len(self.rewards))
                # print("results based on internal score")

                self.rewards = [x.score for x in self.population]

                # print(self.rewards) testing some stuff again
                #todo this might run into issues with lining up the organism with its actual score with mpi

                self.population = [self.population[x] for x in np.argsort(self.rewards)[::-1]]
                self.population_size = len(self.population)

                # If we are murdering the ones that don't win, use this
                #         self.population_new = [pop for pop in self.population if pop.winner]

                #         if(len(self.population_new) > 2):
                #             self.population = self.population_new
                #             self.population_size = len(self.population)

                new_population = []
                for i in range(self.population_size):
                    parent_1_idx = i % self.holdout
                    # print(parent_1_idx)

                    if self.mating:
                        parent_2_idx = min(self.population_size - 1, int(np.random.exponential(self.holdout)))
                    else:
                        parent_2_idx = parent_1_idx
                    offspring = self.population[parent_1_idx].mate(self.population[parent_2_idx])
                    new_population.append(offspring)
                if keep_best:
                    new_population[-1] = self.population[0]  # Ensure best organism survives
                self.population = new_population
                # return -1 * max(rewards)

    def get_best_organism(self, repeats=1, include_reward=False):
        # rewards = [np.mean(self.scoring_function(x)) for _ in range(repeats) for x in self.population]
        if include_reward:
            best = np.argsort(self.rewards)[-1]
            return self.population[best], self.rewards[best]
        else:
            return self.population[np.argsort(self.rewards)[-1]]


organism_creator = lambda: Organism([7, 32, 32, 8, 1], output='relu')

scoring_function = lambda organism_1, organism_2 : sim.simulate_and_evaluate(organism_1, organism_2, print_game=False, trials=1)
ecosystem = Ecosystem(organism_creator, scoring_function, population_size=40, holdout=0.1, mating=True)

generations = 15
best_ai_list = []
best_ai_models = []

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print("Starting simulations")

for i in range(generations):
    if rank == 0:
        print("Rank", rank, "Starting generation ", i, " out of ", generations)
        print("Population size is: ", ecosystem.population_size)

    ecosystem.mpi_generation()

    if rank == 0:
        best_ai = ecosystem.get_best_organism(repeats=1, include_reward=True)
        best_ai_models.append(best_ai[0])
        best_ai_list.append(best_ai[1])
        print("Best AI = ", best_ai[1])
        ecosystem.get_best_organism().save("changed_rooks_and_depth3_model.pkl")

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