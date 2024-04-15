import numpy as np
import dill
import layers
from layers import (
    AbstractLayer,
    Input,
    Dense,
    Conv,
    Attn,
    SkipConn,
    BatchNorm,
)
import copy
from abc import abstractmethod
import random

class Organism:
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, state):
        pass

    @abstractmethod
    def mutate(self):
        pass

    @abstractmethod
    def mate(self, other):
        pass

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            dill.dump(self, file)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as file:
            organism = dill.load(file)
        return organism


class NEATOrganism(Organism):
    _learning_rate: float
    _layers: list[AbstractLayer]
    _layer_constraints: dict[str:dict]
    _layer_options: list
    _n_inputs: int
    _n_outputs: int
    _add_rate: float
    _del_rate: float
    _layer_str_map = {
        Dense: 'dense',
        Conv: 'conv',
        Attn: 'attn',
        SkipConn: 'skipconn',
        BatchNorm: 'batchnorm',
    }

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 learning_rate=.1,
                 add_rate=.1,
                 del_rate=.1,
                 **constraints):
        """
        Initialize an organism for a Neuro-evolution by Augmented Topologies (NEAT) genetic algorithm.

        Params:
        n_inputs: int
            The feature space for the models to be generated.
        n_outputs: int
            The number of outputs for the model to generate.
        learning_rate: float
            A scalar value used to scale the deltas when updating parameters
            Default: 0.1
        add_rate: float
            When replicating, the probability of adding a new layer to the child organism
            Default: 0.1
        del_rate: float
            When replicating, the probability of removing a layer in the child organism
            Default: 0.1
        **constraints: a set of optional arguments that set constraints for the layers of the model behind the organism
            options: list[str]
                The list of layer types that the organism is allowed to have.
                Current valid options:
                    - 'dense': Dense layer
                    - 'conv': Convolutional layer
                    - 'attn': Attention module
                    - 'skipconn': Skip-connection module
                    - 'batchnorm': Batch normalization module
            Specific constraints can be set for each of the layer types:
            dense: dict[str:any]
                'n_nodes': tuple
                    the minimum and maximum number, respectively, of nodes
                    permitted in the dense layer
            conv: dict[str:any]
                'n_nodes': tuple
                    the minimum and maximum number, respectively, of nodes
                    permitted in the convolutional layer
            attn: dict[str:any]
                'n_nodes': tuple
                    the minimum and maximum number, respectively, of output
                    nodes permitted in the attention module.
            skipconn: dict[str:any]
                'n_nodes': tuple
                    the minimum and maximum number, respectively, of output
                    nodes permitted in the attention module.
            batchnorm: dict[str:any]
                There are currently no contraints supported for this layer type.
        """
        self._layers = []
        self._layer_options = []
        self._learning_rate = learning_rate
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._layer_constraints = constraints
        self._add_rate = add_rate
        self._del_rate = del_rate

        if "options" in constraints.keys():
            options = constraints['options']
            if self._layer_str_map[Dense] in options:
                self._layer_options.append(Dense)
            if self._layer_str_map[Conv] in options:
                self._layer_options.append(Conv)
            if self._layer_str_map[Attn] in options:
                self._layer_options.append(Attn)
            if self._layer_str_map[SkipConn] in options:
                self._layer_options.append(SkipConn)
            if self._layer_str_map[BatchNorm] in options:
                self._layer_options.append(BatchNorm)
            # As layer options are added, update this logic
        else:
            self._layer_options = [
                Dense,
                Conv,
                Attn,
                SkipConn,
                BatchNorm
                # Similarly, as layer options are added, update this list
            ]
        
        default_n_node_range = (2,32)
        
        if not self._layer_str_map[Dense] in constraints.keys():
            self._layer_constraints[self._layer_str_map[Dense]] = {'n_nodes': default_n_node_range}
        if not self._layer_str_map[Conv] in constraints.keys():
            self._layer_constraints[self._layer_str_map[Conv]] = {'n_nodes': default_n_node_range}
        if not self._layer_str_map[Attn] in constraints.keys():
            self._layer_constraints[self._layer_str_map[Attn]] = {'n_nodes': default_n_node_range}
        if not self._layer_str_map[SkipConn] in constraints.keys():
            self._layer_constraints[self._layer_str_map[SkipConn]] = {'n_nodes': default_n_node_range} # TODO - consider adding different methods for skip
        if not self._layer_str_map[BatchNorm] in constraints.keys():
            self._layer_constraints[self._layer_str_map[BatchNorm]] = {}

        if not "activations" in constraints.keys():
            self._layer_constraints['activations'] = [layers.xelu, 'sigmoid']

        m_input = Input(n_inputs)
        self._layers.append(m_input)
        self._layers.append(Dense(m_input, n_outputs, learning_rate))
    
    def predict(self, state):
        """
        Given a simulation state, feed-forward and return the model's predictions.

        Params:
        state: Numpy-style array of shape n_inputs
            The board state on which to run the model.

        Returns:
            The prediction of shape n_outputs of the model based on the given state.
        """
        return self._layers[-1](state)
    
    def mutate(self):
        """
        Generate and return a child of this organism with some modifications.
        """
        cop = self._copy()
        layers = []
        for l in cop._layers:
            layers.append(l.update())
        cop._layers = layers

        # deletion step
        roll = random.random()
        if roll < self._del_rate:
            loc = random.randrange(1,len(self._layers)-1)
            cop._layers[loc+1].prior = cop._layers[loc].prior
            cop._layers.pop(loc)
        
        # insertion step
        roll = random.random()
        if roll < self._add_rate:
            loc = random.randrange(1,len(cop._layers)-1)
            layer = random.choice(cop._layer_options)
            constraints = cop._layer_constraints[self._layer_str_map[layer]]
            if not layer == SkipConn:
                n_node_min, n_node_max = constraints['n_nodes'] if 'n_nodes' in constraints.keys() else (0,0)
                insert = layer(
                    prior=cop._layers[loc-1],
                    out_features=random.randint(n_node_min, n_node_max),
                    learning_rate=cop._learning_rate,
                    kwargs=constraints[self._layer_str_map[layer]]
                )
            else: # selected layer type is a skip connection
                skip_from = random.choice(cop._layers[:loc])
                insert = SkipConn(
                    prior=cop._layers[loc-1],
                    out_features=random.randing(n_node_min, n_node_max),
                    learning_rate=cop._learning_rate,
                    skip_from=skip_from,
                    kwargs=cop._layer_constraints[self._layer_str_map[SkipConn]]
                )
            
            cop._layers.insert(loc, insert)
            cop._layers[loc+1].prior = insert
            
        return cop
    
    def _copy(self):
        c = NEATOrganism(self._n_inputs,
                         self._n_outputs,
                         learning_rate=self._learning_rate,
                         options=self._layer_options
                        )
        c._layers = [l.copy() for l in self._layers]
        c._layer_constraints = self._layer_constraints
        c._add_rate = self._add_rate
        return c

    def mate(self, other):
        raise NotImplementedError
    
class FixedOrganism(Organism):
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