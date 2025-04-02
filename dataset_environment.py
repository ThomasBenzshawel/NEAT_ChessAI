import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

from organisms import Organism
from ecosystem import Ecosystem

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
            ecosystem_config={}
        ):
        self.is_classification = is_classification
        self.positive_thresh = positive_thresh
        data = pd.read_csv(dataset_path)
        X = data.filter(items=x_cols)
        y = data[y_col]
        if val != None and val > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val)
        else:
            # if val is None or 0, then use full train set for validation
            X_train, X_val = X, X
            y_train, y_val = y, y
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()
        self.X_val = X_val.to_numpy()
        self.y_val = y_val.to_numpy()
        self.ecosystem = Ecosystem(X.shape[1], 1, **ecosystem_config)
        self.n_agents = self.ecosystem.pop_size
        match(score):
            case 'accuracy':
                self.score_func = accuracy_score
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
            batch_idx, batch = self.ecosystem.poll_agents(batch_size)
            while batch.size != 0:
                batch_scores = np.zeros(batch_size)
                for i, agent in enumerate(batch):
                    y_pred = agent.predict(self.X_val)
                    if type(y_pred) == torch.Tensor:
                        y_pred = y_pred.detach().numpy()
                    if self.is_classification:
                        y_pred = (y_pred >= self.positive_thresh).astype(int)
                    agent_score = self.score_func(self.y_val, y_pred)
                    batch_scores[i] = agent_score
                scores[batch_idx] = batch_scores
                batch_idx, batch = self.ecosystem.poll_agents(batch_size)
                return scores
        max_scores = np.zeros(iterations)
        avg_scores = np.zeros(iterations)
        for gen in range(iterations-1):
            scores = run_iteration()
            self.ecosystem.repopulate(scores)
            max_scores[gen] = scores.max()
            avg_scores[gen] = scores.mean()
            print(f"Max score for generation {gen}:", scores.max())
            # print(f"Average score for generation {gen}:", scores.mean())
        final_scores = run_iteration()
        max_scores[-1] = final_scores.max()
        avg_scores[-1] = final_scores.mean()
        print(f"Max score for final generation:", final_scores.max())
        return max_scores, avg_scores, self.ecosystem.order_population(final_scores)

if __name__ == "__main__":
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
        'options': ['dense',],
        'output_activation': 'sigmoid',
    }
    eco_config = {
        'population_size': 500,
        'NEAT': neat_config,
        'breeding_threshold': .05,
        'org_types': ['NEAT']
    }
    POP_SIZE = 50_000
    env = TabularEnvironment(
        "./test_montecarlo_set.csv",
        feature_cols,
        'y',
        val=.2,
        is_classification=True,
        positive_thresh=.5,
        ecosystem_config=eco_config
    )
    ITERATIONS= 75
    try:
        max_scores, avg_scores, final_pop = env.run(iterations=ITERATIONS, batch_size=100)
    finally:
        print("Breeding pool of final population:")
        print("\n".join([str(a) for a in final_pop[:env.ecosystem._breed_thresh]]))
        plt.plot(np.arange(ITERATIONS), max_scores, label="Max Score")
        plt.plot(np.arange(ITERATIONS), avg_scores, label="Avg Score")
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.title("XOR Model Progression")
        plt.legend()
        plt.savefig('xor_results.png')
        plt.show()