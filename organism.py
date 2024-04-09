import numpy as np
import dill
import copy

class Organism:
    def __init__(self, dimensions, output='softmax'):
        self.score = 0

        self.winner = False

        self.layers = []
        self.biases = []
        self.output = self._activation(output)
        self.dimensions = dimensions

        for i in range(len(dimensions) - 1):
            shape = (dimensions[i], dimensions[i + 1])
            std = np.sqrt(2 / sum(shape))
            layer = np.random.normal(0, std, shape)
            bias = np.random.normal(0, std, (1, dimensions[i + 1]))
            self.layers.append(layer)
            self.biases.append(bias)

    def _activation(self, output):
        if output == 'softmax':
            return lambda X: np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)
        if output == 'sigmoid':
            return lambda X: (1 / (1 + np.exp(-X)))
        if output == 'linear':
            return lambda X: X
        if output == 'relu':
            return lambda X: max(0, X)

    def predict(self, X):
        #TODO: Change to pytorch style
        if not X.ndim == 2:
            raise ValueError(f'Input has {X.ndim} dimensions, expected 2')
        if not X.shape[1] == self.layers[0].shape[0]:
            raise ValueError(f'Input has {X.shape[1]} features, expected {self.layers[0].shape[0]}')
        for index, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            # Update to be pytorch style
            X = X @ layer + np.ones((X.shape[0], 1)) @ bias
            if index == len(self.layers) - 1:
                X = self.output(X)  # output activation
            else:
                X = np.clip(X, 0, np.inf)  # ReLU

        return X

    def predict_choice(self, X, deterministic=True):
        #TODO: Change to pytorch style
        probabilities = self.predict(X)
        if deterministic:
            return np.argmax(probabilities, axis=1).reshape((-1, 1))
        if any(np.sum(probabilities, axis=1) != 1):
            raise ValueError(f'Output values must sum to 1 to use deterministic=False')
        if any(probabilities < 0):
            raise ValueError(f'Output values cannot be negative to use deterministic=False')
        choices = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            U = np.random.rand(X.shape[0])
            c = 0
            while U > probabilities[i, c]:
                U -= probabilities[i, c]
                c += 1
            else:
                choices[i] = c
        return choices.reshape((-1, 1))

    def mutate(self, stdev=0.03):
        #TODO: Change to pytorch style
        for i in range(len(self.layers)):
            self.layers[i] += np.random.normal(0, stdev, self.layers[i].shape)
            self.biases[i] += np.random.normal(0, stdev, self.biases[i].shape)

    def mate(self, other, mutate=True):
        #TODO: Change to pytorch style 
        if not len(self.layers) == len(other.layers):
            raise ValueError('Both parents must have same number of layers')
        if not all(self.layers[x].shape == other.layers[x].shape for x in range(len(self.layers))):
            raise ValueError('Both parents must have same shape')

        child = copy.deepcopy(self)
        for i in range(len(child.layers)):
            pass_on = np.random.rand(1, child.layers[i].shape[1]) < 0.5
            child.layers[i] = pass_on * self.layers[i] + ~pass_on * other.layers[i]
            child.biases[i] = pass_on * self.biases[i] + ~pass_on * other.biases[i]
        if mutate:
            child.mutate()
        return child

    def save_human_readable(self, filepath):
        file = open(filepath, 'w')
        file.write('----------NEW MODEL----------\n')
        file.write('DIMENSIONS\n')
        for dimension in self.dimensions:
            file.write(str(dimension) + ',')
        file.write('\nWEIGHTS\n')
        for layer in self.layers:
            file.write('NEW LAYER\n')
            for node in layer:
                for weight in node:
                    file.write(str(weight) + ',')
                file.write('\n')
            file.write('\n')
            file.write('BIASES:\n')
        for layer in self.biases:
            file.write('\nNEW LAYER\n')
            for connection in layer:
                file.write(str(connection) + ',')
        file.close()

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as file:
            organism = dill.load(file)
        return organism