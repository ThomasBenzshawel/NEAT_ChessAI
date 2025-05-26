from concurrent.futures import ThreadPoolExecutor
import threading
from organisms import NEATOrganism, RandomOrganism, Organism

from copy import copy, deepcopy
import numpy as np
from random import randint

_orgs_list = [NEATOrganism, RandomOrganism]

class Ecosystem():
    """
    Defines an object that manages a population of organisms
    in an evolutionary sim environment across generations.
    """
    def __init__(
            self,
            input_shape: tuple | int,
            out_size: int,
            org_types: list[str] | None=None,
            population_size: int = 100,
            breeding_threshold: str | int | float = 'log',
            mating: bool = False,
            test_eval: bool = False,
            use_elitism: bool = True,
            n_workers=5,
            **organism_constraints
        ):
        self.n_workers=n_workers
        self.is_parallel = n_workers > 0
        # store list of allowable organism types (currently limited to random and NEAT)
        if org_types == None:
            self.org_types = copy(_orgs_list)
        else:
            self.org_types = []
            for org in org_types:
                match (org):
                    case "NEAT" | "neat":
                        self.org_types.append(NEATOrganism)
                    case "random":
                        self.org_types.append(RandomOrganism)
        
        # initialize initial population
        n_allowed_orgs = len(self.org_types)
        self.population = list()
        self.pop_size = population_size
        for _ in range(population_size):
            o_type: Organism.__class__ = self.org_types[randint(0, n_allowed_orgs-1)]
            if o_type == NEATOrganism and "NEAT" in organism_constraints.keys():
                constraints = organism_constraints["NEAT"]
                new_org = NEATOrganism(
                    input_shape,
                    out_size,
                    **constraints
                )
            else: # default parameters are used
                new_org = o_type(input_shape, out_size)
            self.population.append(new_org)
        
        # define breeding threshold
        if breeding_threshold == 'log':
            self._breed_thresh = max(1, int(np.log(population_size)))
        elif 0 < breeding_threshold < 1:
            self._breed_thresh = max(1, int(breeding_threshold * population_size))
        elif type(breeding_threshold) is int:
            if breeding_threshold > population_size or breeding_threshold <= 0:
                raise ValueError(f"Cannot select breeding threshold that is negative or larger than population size (was {breeding_threshold}).")
            self._breed_thresh = breeding_threshold
        else:
            raise ValueError("Invalid breeding threshold value.")
        
        if self._breed_thresh > self.pop_size:
            raise ValueError("Cannot have breeding threshold greater than population size")
        
        self.mating = mating
        self.elite = use_elitism
        self.test_eval = test_eval
        self._poll_idx = 0

        def mutate_agent(agent):
            agent.mutate()

        # begin with a diverse population
        if self.is_parallel:
            with ThreadPoolExecutor(max_workers=n_workers) as exec:
                for i, agent in enumerate(self.population):
                    exec.submit(mutate_agent, agent)
        else:
            for i, agent in enumerate(self.population):
                mutate_agent(agent)

    
    def poll_agent(self):
        return self.poll_agents(1)
    
    def poll_agents(self, n_agents: int | float) -> tuple[np.ndarray[int], np.ndarray[Organism]]:
        """Pulls the next `n_agents` from the population for use by the environment.
        It is the responsibility of the caller to maintain the order of the polled agents
        for passing scores into `repopulate`.

        Args:
            n_agents (int | float): number or percentage of agents to pull from the population
        
        Returns:
            indices (np.ndarray[int]): If there are agents left to poll,
                                   the indices for the polled agents, otherwise empty array
            agents (np.ndarray[Organism]): If there are agents left to poll,
                                  the "next" `n_agents` in the population, otherwise `None`
        """
        if self._poll_idx > self.pop_size:
            return np.array([]), np.array([])
        if type(n_agents) == int:
            idx = np.arange(self._poll_idx, min(self.pop_size-1, self._poll_idx+n_agents))
        elif type(n_agents) == float and 0 < n_agents < 1:
            n_agents = int(len(self.population) * n_agents)
            idx = np.arange(self._poll_idx, min(self.pop_size-1, self._poll_idx+n_agents))
        else:
            raise ValueError("Unrecognized value or type for number of agents to pull.")
        
        self._poll_idx += n_agents
        return idx, np.array(self.population)[idx]

    def order_population(self, scores):
        sorted_pop = np.vstack((self.population, scores)).T
        sorted_pop = sorted_pop[sorted_pop[:,1].argsort()][::-1]
        return sorted_pop[:,0]

    def repopulate(self, scores):
        self._poll_idx = 0 # new population means polling should start back from beginning

        sorted_pop = self.order_population(scores)
        
        # if using elitism, start by placing un-changed parents into new population
        breeding_pool = sorted_pop[:self._breed_thresh].tolist()
        if self.elite:
            new_pop = [a for a in breeding_pool]
            to_delete = self.population[self._breed_thresh:]
        else:
            new_pop = []
            to_delete = self.population
        
        # until size condition of new population is met, add children to new population
        # NOTE: mating is not currently supported
        if self.is_parallel:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                parent_idx = 0
                while len(new_pop) < self.pop_size:
                    child: Organism = copy(breeding_pool[parent_idx])
                    executor.submit(child.mutate)
                    new_pop.append(child)
                    parent_idx = (parent_idx + 1) % len(breeding_pool)
                executor.shutdown(wait=True)
        else:
            parent_idx = 0
            while len(new_pop) < self.pop_size:
                child: Organism = copy(breeding_pool[parent_idx])
                child.mutate()
                new_pop.append(child)
                parent_idx = (parent_idx+1) % len(breeding_pool)
        
        # To fully kill off unused agents and to make sure it is garbage-collected,
        # explicitly delete them
        for killed in to_delete:
            del killed
        
        self.population = new_pop

if __name__  == "__main__":
    def score_agent(agent: NEATOrganism, h: float=6, k: float=10):
        """This is a parabolic scoring function that defines its vertex at (h,k), k != 0
        and has a root at (0,0). The derivation of this function is trivial and left
        as an exercise for the reader.

        y = a(x-h)^2 + k

        Args:
            agent (NEATOrganism): The agent to score
            h (float): the number of layers that the agent should have for maximum score
            k (float): the maximum score

        Returns:
            score (float): The score of the agent.
        """
        a = -k / (h**2)
        l = len(agent._layers)
        return a*(l-h)**2 + k

    score_agent_v = np.vectorize(score_agent)

    pop_size = 10000
    eco = Ecosystem(10, 10, use_elitism=False, population_size=pop_size, org_types=['NEAT'], NEAT={})
    GENERATIONS = 10
    BATCH_SIZE = 10
    for gen in range(GENERATIONS):
        scores = np.zeros(pop_size)
        batch_idx, batch = eco.poll_agents(BATCH_SIZE)
        while batch.size != 0:
            scores[batch_idx] = score_agent_v(batch)
            batch_idx, batch = eco.poll_agents(BATCH_SIZE)
        print(f"Avg score for gen {gen}: {scores.mean()}")
        eco.repopulate(scores)
