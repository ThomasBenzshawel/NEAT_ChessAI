import numpy as np
import random
import torch
from torch import nn
import pickle

from abc import abstractmethod
from collections.abc import Iterable
from math import prod

def copy_model(last_layer):
    return pickle.loads(pickle.dumps(last_layer))

def _update_weights(w, r, sig=1):
    w += r*np.random.normal(scale=sig, size=w.shape)

def softmax(xs):
    xs = np.clip(xs, -10, 10)
    xs = np.exp(xs)
    return xs / xs.sum(axis=-1, keepdims=True)

class SupervisedLayer(nn.Module):
    def __init__(self, input_shape: tuple | int, output_shape: tuple | int):
        super().__init__()
        self._input_shape = input_shape
        self.output_shape = output_shape
    
    @property
    def is_output_flat(self):
        return not (isinstance(self.output_shape, Iterable) and len(self.output_shape) > 1)
    
    @property
    def input_shape(self):
        return self._input_shape
    
    
    @input_shape.setter
    @abstractmethod
    def input_shape(self, val):
        pass
    
    @abstractmethod
    def forward(self, X):
        pass

class SupervisedConvLayer(SupervisedLayer):
    def __init__(
            self,
            input_shape,
            kernel_size,
            n_kernels,
            padding=0,
            stride=1,
            activation="relu",
        ):
        self.kernel_size = kernel_size
        self.n_kernels = n_kernels
        self.padding = padding
        self.stride = stride
        # assumed that the input shape is either square or cube
        # and that last value in input_shape corresponds to n_kernels in previous conv layer
        # (or 1 if first conv layer)
        multi_dim_input = isinstance(input_shape, Iterable)
        W = input_shape[0] if multi_dim_input else input_shape
        # based on stackoverflow answer: https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
        out_size = ((W-kernel_size-(2*padding))/stride) + 1
        n_dims = len(input_shape) - 1 if multi_dim_input else 1
        output_shape = (*[out_size for _ in range(n_dims)], n_kernels)
        super().__init__(input_shape, output_shape)
        match n_dims:
            case 1:
                l = nn.Conv1d(
                    input_shape[0] if multi_dim_input else input_shape,
                    n_kernels,
                    kernel_size,
                    stride=stride,
                    padding=padding
                )
            case 2:
                l = nn.Conv2d(
                    input_shape[-1],
                    n_kernels,
                    kernel_size,
                    stride=stride,
                    padding=padding
                )
            case 3:
                l = nn.Conv3d(
                    input_shape[-1],
                    n_kernels,
                    kernel_size,
                    stride=stride,
                    padding=padding
                )
            case _:
                raise ValueError(f"{n_dims}-dimensional convolutional layer not supported.")
        layers = [l]
        match activation:
            case "relu":
                layers.append(nn.ReLU())
            case "sigmoid":
                layers.append(nn.Sigmoid())
            case "linear":
                pass
            case _:
                raise ValueError(f"Did not recognize {activation} as a supported activation function.")
        self._layers = layers
        self._layer_sequence = nn.Sequential(*layers)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    @SupervisedLayer.input_shape.setter
    def input_shape(self, val):
        super()._input_shape = val
        self._layers[0] = type(self._layers[0])(
            self.input_shape[-1],
            self.output_shape[-1],
            self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )
        self._layer_sequence = nn.Sequential(*self._layers)
    
    def forward(self, X):
        return self._layer_sequence(X)

class SupervisedDenseLayer(SupervisedLayer):
    def __init__(self, input_shape, nodes_out, activation='relu'):
        super().__init__(input_shape, nodes_out)
        self.flatten = isinstance(input_shape, Iterable) and len(input_shape) > 1
        match activation:
            case 'relu':
                activation_layer = nn.ReLU()
            case 'sigmoid':
                activation_layer = nn.Sigmoid()
            case 'softmax':
                activation_layer = nn.Softmax(dim=1)
            case _:
                raise ValueError(f"{activation} not recognized as supported activation type")
        self._layers = [
            nn.Linear(
                prod(input_shape) if isinstance(input_shape, Iterable) else input_shape,
                nodes_out
            ),
            activation_layer
        ]
        self._sequential = nn.Sequential(*self._layers)
        self.flatten_layer = nn.Flatten()
    
    @SupervisedLayer.input_shape.setter
    def input_shape(self, val):
        self._input_shape = val
        self._layers[0] = nn.Linear(
            prod(self.input_shape) if isinstance(self.input_shape, Iterable) else self.input_shape,
            self.output_shape
        )
        self._sequential = nn.Sequential(*self._layers)
    
    def forward(self, X):
        if self.flatten:
            X = self.flatten_layer(X)
        y = self._sequential(X)
        return y

class AbstractLayer:
    def __init__(self, prior:"AbstractLayer", learning_rate:float, out_features:int, allowed_activations:list[str]=None, **kwargs):
        assert type(out_features) == int # No longer doing multidim input for now

        self.prior = prior
        self.out_features = out_features
        self.learning_rate = learning_rate

        if allowed_activations is None:
            allowed_activations = ['xelu', 'sigmoid']
        self.activation = random.choice(allowed_activations)

    def update(self):
        pass

    def __call__(self, x):
        return self.prior(x)

    def iter(self, ret=None):
        if ret is None:
            ret = [self]
        return self.prior.iter() + ret
    
    def _activation(self):
        return {
            'xelu': lambda x: np.clip(x, -10, 10)/(1+np.exp(-np.clip(x, -10, 10))),
            'sigmoid': lambda x: 1/(1+np.exp(-np.clip(x, -10, 10)))
        }[self.activation]
    
    def __str__(self):
        return f"{self.__class__.__name__}[{self.out_features}]"

class Input(AbstractLayer):
    def __init__(self, out_features:int, **kwargs):
        super().__init__(None, 0, out_features, kwargs=kwargs)

    def __call__(self, x):
        x = np.array(x, dtype=float)
        if len(x.shape) != 2 or x.shape[-1] != self.out_features:
            raise ValueError(f'Expected input shape (batch_size, {self.out_features}), but got shape: {x.shape}')
        return x

    def iter(self):
        return [self]

class Dense(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float, out_features:int, allowed_activations=None, **kwargs):
        super().__init__(prior, learning_rate, out_features, allowed_activations, kwargs=kwargs)
        self.weights = np.random.uniform(-10, 10, size=(prior.out_features,out_features))
        self.noise = kwargs['noise'] if 'noise' in kwargs.keys() else 1

    def update(self):
        _update_weights(self.weights, self.learning_rate, sig=self.noise)

    def __call__(self, x):
        return self._activation()(super().__call__(x) @ self.weights)

class Conv(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float, out_features:int, allowed_activations=None, **kwargs):
        super().__init__(prior, learning_rate, out_features, allowed_activations, kwargs=kwargs)

        assert out_features <= self.prior.out_features
        kernel_size = self.prior.out_features - out_features + 1
        self.weights = np.random.normal(size=kernel_size)

    def update(self):
        _update_weights(self.weights, self.learning_rate)

    def __call__(self, x):
        return self._activation()(np.array([
            np.convolve(x_i, self.weights, 'valid')
            for x_i in super().__call__(x)
        ]))

class Attn(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float, out_features:int, allowed_activations=None, **kwargs):
        super().__init__(prior, learning_rate, out_features, allowed_activations, kwargs=kwargs)
        self.W_q = np.random.normal(size=(prior.out_features, out_features))
        self.W_k = np.random.normal(size=(prior.out_features, out_features))
        self.W_v = np.random.normal(size=(prior.out_features, out_features))

    def update(self):
        _update_weights(self.W_q, self.learning_rate)
        _update_weights(self.W_k, self.learning_rate)
        _update_weights(self.W_v, self.learning_rate)

    def __call__(self, x):
        x = super().__call__(x)

        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        return self._activation()(softmax(q*k)*v)

class BatchNorm(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float, **kwargs):
        super().__init__(prior, learning_rate, prior.out_features, kwargs=kwargs)
        self.gamma = np.log(np.e - 1)
        self.beta = 0

    def update(self):
        self.beta += self.learning_rate*np.random.normal()
        self.gamma += self.learning_rate*np.random.normal()

    def __call__(self, x):
        x = super().__call__(x)

        scale = np.log(np.exp(self.gamma)+1)
        offset = self.beta

        if np.std(x) == 0:
            return x

        return scale * (x - np.mean(x)) / np.std(x) + offset

class SkipConn(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float, out_features:int, skip_from:AbstractLayer, allowed_activations=None, **kwargs):
        super().__init__(prior, learning_rate, out_features, allowed_activations, **kwargs)
        self.prior_to_out = np.random.normal(size=(prior.out_features, out_features))
        self.skip_to_out = np.random.normal(size=(skip_from.out_features, out_features))
        self.skip_from = skip_from

    def update(self):
        _update_weights(self.prior_to_out, self.learning_rate)
        _update_weights(self.skip_to_out, self.learning_rate)

    def __call__(self, x):
        x1 = super().__call__(x)
        x1 = x1 @ self.prior_to_out

        x2 = self.skip_from(x)
        x2 = x2 @ self.skip_to_out

        return self._activation()(x1 + x2)

if __name__ == '__main__':
    # Example network
    x = Input(3) # input could be a multidim vector (e.g., (5,3) you have have 5 embeddings with dims of 3)
    y = BatchNorm(x, 0.1)
    y = SkipConn(y, 0.1, 5, x) # skip conn from input
    y = Attn(y, 0.1, 3)
    y = Dense(y, 0.1, 5)

    y = Conv(y, 0.1, 3)
    y = type(y)(y, y.learning_rate, y.out_features)
    z = BatchNorm(y, 0.1)
    y = Dense(z, 0.1, 4)
    y = SkipConn(y, 0.1, 4, z)
    model = y

    # Prediction
    # Prediction from a layer X first goes through all prior layers: X(x) = X <- X-1 <- X-2 <- ... <- Input(x).
    print(
        # Predict on batches at a time
        'model_1:',
        model([ # you can also pass a numpy array
            [1,-5,3],
            [-1,10,1]
        ])
    )

    # Update
    for l in model.iter():
        l.update()
    print(
        'model_1, Should be slightly different:',
        model([
            [1,-5,3],
            [-1,10,1]
        ])
    )

    #copying network
    layers2 = copy_model(model)
    model2 = layers2
    # prediction should be same
    print(
        'model_2: Should be same:',
        model2([
            [1,-5,3],
            [-1,10,1]
        ])
    )
    for l in model2.iter():
        l.update()
    print(
        'model_2: Should be different:',
        model2([
            [1,-5,3],
            [-1,10,1]
        ])
    )
    print(
        'model_1: Should be same as last model_1 call:',
        model([
            [1,-5,3],
            [-1,10,1]
        ])
    )

    # #testing pickling
    # import pickle
    # with open('model.pkl', 'wb') as f:
    #     pickle.dump(model, f)

    # with open('model.pkl', 'rb') as f:
    #     model = pickle.load(f)
    # print( # should be same as last model_1 call
    #     'model_1: Should be same as last model_1 call:',
    #     model([
    #         [1,-5,3],
    #         [-1,10,1]
    #     ])
    # )