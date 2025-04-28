from mnist_environment import MNISTEnvironment, accuracy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torchmetrics.classification import MulticlassAveragePrecision
import csv
import random
import time
from genealogy_visualizer import generate_genealogy
import uuid
from filelock import FileLock

EXPERIMENT_ID = uuid.uuid4().hex
OUTPUT_CSV = "./mnist_results.csv"
FILELOCK_PATH = "./mnist_results.csv.lock"
output_lock = FileLock(FILELOCK_PATH)

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
    'n_nodes': (2, 256),
    'activations': ['relu', 'sigmoid']
}
conv_layer_config = {
    'n_nodes': (1, 512),
    'activations': ['relu', 'sigmoid', 'linear']
}
neat_config = {
    'learning_rate': .005,
    'add_rate': add_rate,
    'del_rate': del_rate,
    'supervised': True,
    'fine_change_prob': fine_change_prob,
    'burn_in': 5,
    'loss': 'cross-entropy',
    'dense': dense_layer_config,
    'conv': conv_layer_config,
    'options': ['dense', 'conv'],
    'output_activation': 'softmax',
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
env = MNISTEnvironment(val=.2, ecosystem_config=eco_config, score=accuracy)
max_scores, avg_scores, final_pop = env.run(ITERATIONS, batch_size=25)
end = time.perf_counter()

# get wall clock elapsed time
elapsed = end - start



# evaluate best model against test set
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # taken from https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/6
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
dataset = MNIST("./data", train=False, transform=transform)
dataloader = DataLoader(dataset, batch_size=10000, shuffle=False)

best_model = final_pop[0]

pred = []
y_test = []
n_correct = 0
for bX, by in dataloader:
    bX = bX.cuda()
    by = by.cuda()
    batch_pred: torch.Tensor = best_model.predict(bX)
    pred.append(batch_pred.cpu())
    y_test.append(by.cpu())
    n_correct += (batch_pred.argmax(dim=1) == by).int().sum().item()
pred = torch.cat(pred, dim=0)
y_test = torch.cat(y_test, dim=0).flatten()

# n_correct = ((pred > .5).int() == y_test.cuda()).sum()
accuracy = n_correct / y_test.shape[0]

average_precision = MulticlassAveragePrecision(num_classes=10)
ap_score = average_precision(pred, y_test.int()).item()

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
generate_genealogy(
    f"montecarlo_genealogy_{EXPERIMENT_ID}",
    final_pop[:10],
    graph_attr={
        'label': f"Genealogy of Top 10 Agents (MNIST Environment)",
        'labelloc': 'top'
    }
)