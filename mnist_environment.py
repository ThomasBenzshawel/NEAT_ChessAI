from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from typing import Any, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

from ecosystem import Ecosystem

def accuracy(real: torch.Tensor, predicted: torch.Tensor):
    if real.shape == predicted.shape:
        n_correct = (real == predicted).int().sum().item()
    else:
        n_correct = (real == predicted.argmax(dim=1)).int().sum().item()
    return n_correct / real.shape[0]

class MNISTDataset(MNIST):
    def __init__(
            self,
            root: str | Path,
            train: bool=True,
            transform=None,
            target_transform=None,
            download: bool=False
    ):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

class MNISTEnvironment:
    def __init__(
            self,
            val=None,
            score='accuracy',
            ecosystem_config: dict = {}
    ):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # taken from https://discuss.pytorch.org/t/normalization-in-the-mnist-example/457/6
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.val = val if val != None else 0
        mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        # if torch.cuda.is_available():
        #     dev = torch.device('cuda')
        #     mnist_dataset.data = mnist_dataset.data.to(dev)
        #     mnist_dataset.targets = mnist_dataset.targets.to(dev)
        self.is_cuda = torch.cuda.is_available()
        if val != None and val > 0:
            train_dataset, val_dataset = random_split(mnist_dataset, [1-val, val])
        else:
            train_dataset, val_dataset = mnist_dataset, mnist_dataset
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

        if 'NEAT' in ecosystem_config.keys():
            ecosystem_config['NEAT']['X'] = self.train_dataloader
        else:
            ecosystem_config['NEAT'] = { 'X': self.train_dataloader }

        self.ecosystem = Ecosystem((1, 28, 28), 10, **ecosystem_config)
        self.n_agents = self.ecosystem.pop_size
        match(score):
            case 'accuracy':
                self.score_func = accuracy_score
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
                full_pred = []
                full_y = []
                for X, y in self.val_dataloader:
                    if self.is_cuda:
                        X = X.cuda()
                        y = y.cuda()
                    y_pred = agent.predict(X)
                    full_y.append(y)
                    full_pred.append(y_pred)
                pred = torch.cat(full_pred, dim=0).argmax(dim=1)
                y = torch.cat(full_y, dim=0)
                agent_score = self.score_func(y, pred)
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
        for gen in range(iterations):
            print(f"Starting Generation {gen}...")
            print("\tEvaluating Agents...")            
            scores = run_iteration()
            print(f"\tMax score for generation {gen}:", scores.max())
            print(f"\tRepopulating Ecosystem...")
            self.ecosystem.repopulate(scores)
            max_scores[gen] = scores.max()
            avg_scores[gen] = scores.mean()
        print("Evaluating final generation...")
        final_scores = run_iteration()
        max_scores[-1] = final_scores.max()
        avg_scores[-1] = final_scores.mean()
        print(f"Max score for final generation:", final_scores.max())
        return max_scores, avg_scores, self.ecosystem.order_population(final_scores)


if __name__ == "__main__":
    import time
    from genealogy_visualizer import generate_genealogy

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
        'add_rate': .2,
        'del_rate': .2,
        'supervised': True,
        'fine_change_prob': .8,
        'burn_in': 5,
        'loss': 'cross-entropy',
        'dense': dense_layer_config,
        'conv': conv_layer_config,
        'options': ['dense', 'conv'],
        'output_activation': 'softmax',
    }

    POP_SIZE = 25
    eco_config = {
        'population_size': POP_SIZE,
        'NEAT': neat_config,
        'breeding_threshold': .2,
        'org_types': ['NEAT']
    }
    env = MNISTEnvironment(
        val=.2,
        ecosystem_config=eco_config,
        score=accuracy
    )

    ITERATIONS = 10
    start = time.process_time()
    max_scores, avg_scores, final_pop = env.run(iterations=ITERATIONS, batch_size=10)
    end = time.process_time()

    plt.plot(np.arange(len(max_scores)), max_scores, label="Max Score")
    plt.plot(np.arange(len(avg_scores)), avg_scores, label="Avg Score")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.title("MNIST Dataset Model Progression")
    plt.legend()
    plt.savefig("mnist_results.png")

    elapsed = end - start
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    seconds = elapsed % 60

    print(f"Elapsed time taken for {ITERATIONS} iterations of population size {POP_SIZE} (HH:MM:SS): {hours:02g}:{minutes:02g}:{seconds}")
    print("Inter-generational surviving members of final population:")
    print("\n".join([str(a) for a in final_pop[:env.ecosystem._breed_thresh]]))

    generate_genealogy(
        f"mnist_top{env.ecosystem._breed_thresh}_gen",
        final_pop[:env.ecosystem._breed_thresh],
        graph_attr={
            'label': f"Genealogy of Top {env.ecosystem._breed_thresh} Agents",
            'labelloc': 'top'
        }
    )
