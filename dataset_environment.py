from concurrent.futures import ThreadPoolExecutor
import threading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

from organisms import Organism
from ecosystem import Ecosystem

def torch_accuracy(real: torch.Tensor, predicted: torch.Tensor):
    assert real.flatten().shape == predicted.flatten().shape
    n_correct = (real.flatten() == predicted.flatten()).int().sum().item()
    return n_correct / real.shape[0]

class TabularEnvironment:
    def __init__(
            self,
            dataset_path: str,
            x_cols: list[str],
            y_col: str,
            val=None,
            score='accuracy',
            is_classification=False,
            positive_thresh=.5,
            ecosystem_config: dict={}
    ):
        self.is_classification = is_classification
        self.positive_thresh = positive_thresh
        data = pd.read_csv(dataset_path)
        self.bootstrap = 'NEAT' in ecosystem_config.keys() and 'supervised' in ecosystem_config['NEAT'].keys() and ecosystem_config['NEAT']['supervised']
        X = data.filter(items=x_cols)
        y = data[y_col]
        if val != None and val > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val)
        else:
            # if val is None or 0, then use full train set for validation
            X_train, X_val = X, X
            y_train, y_val = y, y
        
        if self.bootstrap:
            self.X_train = torch.from_numpy(X_train.to_numpy()).float()
            self.y_train = torch.from_numpy(y_train.to_numpy()).float()
            self.X_val = torch.from_numpy(X_val.to_numpy()).float()
            self.y_val = torch.from_numpy(y_val.to_numpy()).float()
            if torch.cuda.is_available():
                self.X_train = self.X_train.cuda()
                self.y_train = self.y_train.cuda()
                self.X_val = self.X_val.cuda()
                self.y_val = self.y_val.cuda()
        else:
            self.X_train = X_train.to_numpy()
            self.y_train = y_train.to_numpy()
            self.X_val = X_val.to_numpy()
            self.y_val = y_val.to_numpy()
        self.ecosystem = Ecosystem(X.shape[1], 1, **ecosystem_config)
        self.n_agents = self.ecosystem.pop_size
        match(score):
            case 'accuracy':
                self.score_func = accuracy_score if not self.bootstrap else torch_accuracy
            case 'f1':
                self.score_func = f1_score
            case 'precision':
                self.score_func = precision_score
            case _:
                if callable(score):
                    self.score_func = score
                else:
                    raise ValueError(f"Did not recognize score function: {score}")
    
    def run(self, iterations=1, batch_size=10):
        def run_iteration():
            scores = np.zeros(self.n_agents)
            scores_lock = threading.Lock()
            def run_agent(agent, i):
                y_pred = agent.predict(self.X_val)
                # if type(y_pred) == torch.Tensor:
                #     y_pred = y_pred.detach().cpu().numpy()
                if self.is_classification:
                    if self.bootstrap:
                        y_pred = (y_pred >= self.positive_thresh).int()
                    else:
                        y_pred = (y_pred >= self.positive_thresh).astype(int)
                agent_score = self.score_func(self.y_val, y_pred)
                with scores_lock:
                    scores[i] = agent_score

            batch_idx, batch = self.ecosystem.poll_agents(batch_size)
            while batch.size != 0:
                with ThreadPoolExecutor(max_workers=5) as executor:
                    for i, agent in zip(batch_idx, batch):
                        executor.submit(run_agent, agent, i)
                batch_idx, batch = self.ecosystem.poll_agents(batch_size)
            return scores
        
        max_scores = np.zeros(iterations+1)
        avg_scores = np.zeros(iterations+1)
        for gen in range(iterations-1):
            print(f"Starting Generation {gen}...")
            print("\tEvaluating Agents...")
            scores = run_iteration()
            print("\tRepopulating Ecosystem...")
            self.ecosystem.repopulate(scores)
            max_scores[gen] = scores.max()
            avg_scores[gen] = scores.mean()
            print(f"\tMax score for generation {gen}:", max_scores[gen])
            print(f"\tAverage score for generation {gen}:", avg_scores[gen])
        final_scores = run_iteration()
        max_scores[-1] = final_scores.max()
        avg_scores[-1] = final_scores.mean()
        print(f"Max score for final generation:", final_scores.max())
        return max_scores, avg_scores, self.ecosystem.order_population(final_scores)

if __name__ == "__main__":
    import time

    TRAIN_SIZE = 100_000
    def non_linear_data_maker(n_features=10, n_samples=10000, noise=0.1, random_seed=None):
        """
        Generate a synthetic dataset where a greedy feature selection algorithm performs poorly.
    
        Parameters:
        - n_samples (int): Number of data points to generate.
        - n_features (int): Total number of features.
        - noise (float): Standard deviation of Gaussian noise to add to the data.
        - random_seed (int): Seed for reproducibility.
    
        Returns:
        - X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        - y (numpy.ndarray): Binary labels of shape (n_samples,).
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
        # Generate random features
        X = np.random.uniform(-5, 5, (n_samples, n_features))
    
        # Create deceptive labels: labels depend on combinations of features, not individual ones
        y = (
            (np.sin(X[:, 0]) + np.cos(X[:, 1]) > 0) &
            (X[:, 2] * X[:, 3] > 3)
        ).astype(int)
    
        # Add noise to make the problem more realistic
        noise_term = np.random.normal(0, noise, n_samples)
        y = np.where(noise_term > 0.5, 1 - y, y)
    
        return X, y
    if not os.path.exists("./test_montecarlo_set.csv"):
        n_features = 10
        X_gen, y_gen = non_linear_data_maker(n_features=n_features, n_samples=TRAIN_SIZE)
        print((y_gen == 1).sum()/y_gen.shape[0])
        train_set = np.hstack([X_gen, y_gen.reshape((TRAIN_SIZE, 1))])
        feature_cols = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(train_set, columns=[*feature_cols, 'y'])
        print(df.head())
        print(df['y'].value_counts())
        df.to_csv("./test_montecarlo_set.csv")
    
    df = pd.read_csv("test_montecarlo_set.csv", index_col=0)
    feature_cols = df.drop(columns=['y']).columns

    dense_layer_config = {
        'noise': 1.5,
        'n_nodes': (2,128),
        'activations': ['relu', 'sigmoid']
    }
    neat_config = {
        'learning_rate': .05,
        'add_rate': .1,
        'del_rate': .1,
        'supervised': True,
        'X': df.drop(columns='y').to_numpy(),
        'y': df['y'].to_numpy(),
        'loss': 'binary-cross-entropy',
        'dense': dense_layer_config,
        'options': ['dense', 'conv'],
        'output_activation': 'sigmoid',
    }
    POP_SIZE = 500
    # Inter-Generational Surviving Population
    igsp_sizes = [.05, *[i / 10 for i in range(1,6)]]

    elapsed_times = []
    pop_max_scores = []
    for igsp in igsp_sizes:
        eco_config = {
            'population_size': POP_SIZE,
            'NEAT': neat_config,
            'breeding_threshold': igsp,
            'org_types': ['NEAT']
        }
        env = TabularEnvironment(
            "./test_montecarlo_set.csv",
            feature_cols,
            'y',
            val=.2,
            is_classification=True,
            positive_thresh=.5,
            ecosystem_config=eco_config
        )
        ITERATIONS= 10
        start = time.process_time()
        max_scores, avg_scores, final_pop = env.run(iterations=ITERATIONS, batch_size=100)
        end = time.process_time()

        elapsed = end - start
        elapsed_times.append(elapsed)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        pop_max_scores.append(max_scores)

        plt.plot(np.arange(len(max_scores)), max_scores, label="Max Score")
        plt.plot(np.arange(len(avg_scores)), avg_scores, label="Avg Score")
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.suptitle("Montecarlo Dataset Model Progression")
        plt.title(f"Inter-generational Surviving Population of {igsp}")
        plt.legend()
        plt.savefig(f'montecarlo_results_{igsp}.png')
        plt.clf()

        print(f"Elapsed time taken for {ITERATIONS} iterations of population size {POP_SIZE} ({igsp * POP_SIZE} of which reproduce) (HH:MM:SS): {hours:02g}:{minutes:02g}:{seconds}")
        print("Breeding pool of final population:")
        print("\n".join([str(a) for a in final_pop[:env.ecosystem._breed_thresh]]))
        # free up RAM for next iteration (it will need it)
        del env
    
    for igsp, scores in zip(igsp_sizes, pop_max_scores):
        plt.plot(np.arange(len(scores)), scores, label=f"$N_{{IGSP}}={igsp}$")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.ylim((0,1))
    plt.legend()
    plt.title("Montecarlo Dataset Accuracy Across Generations")
    plt.savefig("montecarlo_scores_IGSPs.png")
    plt.clf()
    
    plt.plot(igsp_sizes, elapsed_times)
    plt.xlabel("Population Size (log)")
    plt.xscale('log')
    plt.ylabel("Time taken (s)")
    plt.title("Process Time as Population Increases")
    plt.savefig(f"montecarlo_time_IGSP.png")
    plt.show()