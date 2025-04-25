from dataset_environment import TabularEnvironment
import numpy as np
import pandas as pd
import os

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

TRAIN_CSV = "./montecarlo_train.csv"

if not os.path.exists(TRAIN_CSV):
    n_features = 10
    TRAIN_SIZE = 750_000
    X_gen, y_gen = non_linear_data_maker(n_features=n_features, n_samples=TRAIN_SIZE)
    print((y_gen == 1).sum()/y_gen.shape[0])
    train_set = np.hstack([X_gen, y_gen.reshape((TRAIN_SIZE, 1))])
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(train_set, columns=[*feature_cols, 'y'])
    print(df.head())
    print(df['y'].value_counts())
    df.to_csv(TRAIN_CSV)