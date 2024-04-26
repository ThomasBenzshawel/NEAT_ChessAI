import numpy as np
import layers
import random
import pickle


from layers import (
    AbstractLayer,
    Input,
    Dense,
    Conv,
    Attn,
    SkipConn,
    BatchNorm,
)
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
        state = {
            '_learning_rate': self._learning_rate,
            '_layers': self._layers,
            '_layer_constraints': self._layer_constraints,
            '_layer_options': self._layer_options,
            '_n_inputs': self._n_inputs,
            '_n_outputs': self._n_outputs,
            '_add_rate': self._add_rate,
            '_del_rate': self._del_rate
        }
        with open(filepath, 'wb') as file:
            pickle.dump(state, file)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as file:
            state = pickle.load(file)
        organism = NEATOrganism(state['_n_inputs'], state['_n_outputs'])
        organism._learning_rate = state['_learning_rate']
        organism._layers = state['_layers']
        organism._layer_constraints = state['_layer_constraints']
        organism._layer_options = state['_layer_options']
        organism._add_rate = state['_add_rate']
        organism._del_rate = state['_del_rate']
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
                 model_file=None,
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
        if model_file is not None:
            self.load(model_file)
            return
        
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
            self._layer_constraints['activations'] = ['xelu', 'sigmoid']

        m_input = Input(n_inputs)
        self._layers.append(m_input)
        self._layers.append(Dense(m_input, learning_rate, n_outputs))
    
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
                    out_features=random.randint(n_node_min, n_node_max),
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
        c._layers = layers.copy_model(self._layers[-1])
        c._layer_constraints = self._layer_constraints
        c._add_rate = self._add_rate
        return c

    def mate(self, other):
        raise NotImplementedError

# Testing pickling
if __name__ == "__main__":
    organism = NEATOrganism(10, 2)
    organism.save('organism.pkl')
    organism = NEATOrganism.load('organism.pkl')
    test = np.zeros(10).reshape(1,-1)
    print(organism.predict(test))
