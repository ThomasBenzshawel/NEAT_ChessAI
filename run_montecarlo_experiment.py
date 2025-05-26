from dataset_environment import TabularEnvironment
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.classification import BinaryAveragePrecision
import os
import csv
import random
import time
from genealogy_visualizer import generate_genealogy
import uuid
from filelock import FileLock

EXPERIMENT_ID = uuid.uuid4().hex
OUTPUT_CSV = "./montecarlo_results.csv"
FILELOCK_PATH = "./montecarlo_results.csv.lock"
output_lock = FileLock(FILELOCK_PATH)

def generate_montecarlo_set(n_samples, noise=True):
    """
    Binary classification decision function with a complex nonlinear decision boundary.

    Parameters:
        X: np.ndarray of shape (n_samples, 10)
        
    Returns:
        np.ndarray of shape (n_samples,) with binary labels (0 or 1)
    """
    X = np.random.uniform(-5, 5, size=(n_samples, 10))
    
    # Compute a complex nonlinear transformation
    z = (
        np.sin(X[:, 0] * X[:, 1]) +
        np.cos(X[:, 2] + X[:, 3]**2) -
        np.tanh(X[:, 4] * X[:, 5]) +
        np.exp(-X[:, 6]**2) +
        np.log1p(np.abs(X[:, 7])) * np.sign(X[:, 7]) +
        np.sin(X[:, 8] * np.pi) * np.cos(X[:, 9] * np.pi)
    )
    
    # Add some thresholding and noise for complexity
    if noise:
        z += 0.1 * np.random.randn(*z.shape)  # add Gaussian noise

    # Apply a threshold to generate binary labels
    return X, (z > 0).astype(int)

TRAIN_CSV = "./montecarlo_train.csv"
TEST_CSV = "./montecarlo_test.csv"
n_features = 10
feature_cols = [f"feature_{i}" for i in range(n_features)]

if not os.path.exists(TRAIN_CSV):
    TRAIN_SIZE = 750_000
    X_gen, y_gen = generate_montecarlo_set(TRAIN_SIZE)
    print((y_gen == 1).sum()/y_gen.shape[0])
    train_set = np.hstack([X_gen, y_gen.reshape((TRAIN_SIZE, 1))])
    df = pd.DataFrame(train_set, columns=[*feature_cols, 'y'])
    print(df.head())
    print(df['y'].value_counts())
    df.to_csv(TRAIN_CSV)

if not os.path.exists(TEST_CSV):
    TEST_SIZE = 100_000
    X_gen, y_gen = generate_montecarlo_set(TEST_SIZE, noise=False)
    test_set = np.hstack([X_gen, y_gen.reshape((TEST_SIZE, 1))])
    df = pd.DataFrame(train_set, columns=[*feature_cols, 'y'])
    df.to_csv(TEST_CSV)

pop_size = random.randint(10, 100)
diff_rates = random.random() > .5
if diff_rates:
    add_rate = random.random()
    del_rate = random.random()
else:
    add_rate = random.random()
    del_rate = add_rate
inter_generational_surviving_percentage = random.random()
fine_change_prob = random.random()

dense_layer_config = {
    'n_nodes': (1,256),
    'activations': ['relu', 'sigmoid']
}
conv_layer_config = {
    'n_nodes': (1, 256),
    'activations': ['relu', 'sigmoid', 'linear']
}
neat_config = {
    'learning_rate': .05,
    'add_rate': add_rate,
    'del_rate': del_rate,
    'supervised': True,
    'loss': 'binary-cross-entropy',
    'fine_change_prob': fine_change_prob,
    'dense': dense_layer_config,
    'conv': conv_layer_config,
    'options': ['dense', 'conv'],
    'output_activation': 'sigmoid',
}
eco_config = {
    'population_size': pop_size,
    'NEAT': neat_config,
    'breeding_threshold': inter_generational_surviving_percentage,
    'org_types': ['NEAT']
}
ITERATIONS = 10
# because a round of mutations happens in the constructor, include construction time
# in performance time
start = time.perf_counter()
env = TabularEnvironment(TRAIN_CSV, feature_cols, 'y', val=.2, ecosystem_config=eco_config, is_classification=True, n_workers=5)
max_scores, avg_scores, final_pop = env.run(ITERATIONS, batch_size=25)
end = time.perf_counter()

# get wall clock elapsed time
elapsed = end - start



# evaluate best model against test set
test_df = pd.read_csv(TEST_CSV)
X_test = torch.from_numpy(test_df[feature_cols].to_numpy()).float()
y_test = torch.from_numpy(test_df['y'].to_numpy()).float()

dataset = TensorDataset(X_test, y_test)
dataloader = DataLoader(dataset, batch_size=10000, shuffle=False)

best_model = final_pop[0]

pred = []
n_correct = 0
for bX, by in dataloader:
    bX = bX.cuda()
    by = by.cuda()
    batch_pred = best_model.predict(bX)
    pred.append(batch_pred)
    n_correct += ((batch_pred > .5).int() == by).int().sum().item()
pred = torch.cat(pred, dim=0)

# n_correct = ((pred > .5).int() == y_test.cuda()).sum()
accuracy = n_correct / y_test.shape[0]

average_precision = BinaryAveragePrecision()
ap_score = average_precision(pred.cpu().flatten(), y_test.int()).item()

data = [
    EXPERIMENT_ID,
    pop_size,
    add_rate,
    del_rate,
    inter_generational_surviving_percentage,
    elapsed,
    accuracy,
    ap_score,
    str(best_model)
]

success = False
attempts = 0
while attempts < 3 and not success:
    attempts += 1
    try:
        with output_lock.acquire(timeout=10):
            with open(OUTPUT_CSV, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data)
            success = True
    except TimeoutError:

        print(f"Another process is currently writing to the file. Attempt #{attempts}")

# generate genealogy of top 10 agents
# generate_genealogy(
#     f"montecarlo_genealogy_{EXPERIMENT_ID}",
#     final_pop[:10],
#     graph_attr={
#         'label': f"Genealogy of Top {env.ecosystem._breed_thresh} Agents",
#         'labelloc': 'top'
#     }
# )