import numpy as np
import tensorflow as tf
import random
from copy import copy
from dataclasses import dataclass

# Using this in place of RELU.
# It's like RELU, but it isn't exactly 0 for x<0, so it's differentiable (or traversable)
def xelu(x):
    return x / (np.exp(-x) + 1)

def _update_tf_weights(layer, learning_rate):
    layer.set_weights([
        w + learning_rate*tf.random.normal(w.shape)
        for w in layer.weights
    ])

@dataclass
class _KerasLayerConfig:
    conf: dict
    weights: list
    build_conf: dict
    layer_t: type

def _keras_layer_to_config(layer):
    conf = layer.get_config()
    if 'activation' in conf and type(conf['activation']) != str:
        conf['activation']['config'] = xelu
    return _KerasLayerConfig(conf, layer.weights, layer.get_build_config(), type(layer))

def _keras_layer_from_config(conf):
    c = conf.layer_t.from_config(conf.conf)
    c.build(conf.build_conf['input_shape'])
    c.set_weights(conf.weights)
    return c

def _copy_keras_layer(layer):
    if not isinstance(layer, tf.keras.Layer):
        return layer.get_copy()

    conf = _keras_layer_to_config(layer)
    c = _keras_layer_from_config(conf)
    return c

def copy_model(last_layer):
    c = [l.get_copy() for l in last_layer.iter()]
    for i in range(len(c)-1):
        c[i+1].prior = c[i]
        if type(c[i+1]) is SkipConn:
            c[i+1].skip_from = c[c[i+1].skip_from]
    return c

class AbstractLayer:
    def __init__(self, prior:"AbstractLayer", learning_rate:float, out_features:int, allowed_activations:list[str]=None, **kwargs):
        self.prior = prior
        self.out_features = out_features
        self.learning_rate = learning_rate
        if allowed_activations is None:
            allowed_activations = ['xelu', 'sigmoid']
        self.activation = random.choice(allowed_activations)
        if self.activation == 'xelu':
            self.activation = xelu

    def update(self):
        pass

    def __call__(self, x):
        return self.prior(x)

    def iter(self, ret=None):
        if ret is None:
            ret = [self]
        return self.prior.iter() + ret

    def get_copy(self):
        c = copy(self)
        c.prior = None
        return c

    def __getstate__(self):
        input_shape = self.iter()[0].out_features
        self.__call__([
            [0]*input_shape
        ])
        return {
            k: (v if not isinstance(v, tf.keras.Layer) else _keras_layer_to_config(v))
            for k,v in self.__dict__.items()
        }

    def __setstate__(self, d):
        self.__dict__ = {
            k: v if type(v) != _KerasLayerConfig else _keras_layer_from_config(v)
            for k,v in d.items()
        }

class Input(AbstractLayer):
    def __init__(self, out_features:tuple, **kwargs):
        super().__init__(None, 0, out_features, kwargs=kwargs)

    def __call__(self, x):
        x = tf.constant(x, dtype=float)
        if len(x.shape) < 2 or x.shape[1:] != self.out_features:
            out_shape_msg = ', '.join([str(d) for d in self.out_features])
            raise ValueError(f'Expected input shape (batch_size, {out_shape_msg}), but got shape: {x.shape}')
        return x

    def iter(self):
        return [self]

class Dense(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float, out_features:int, allowed_activations=None, **kwargs):
        super().__init__(prior, learning_rate, out_features, allowed_activations, kwargs=kwargs)
        self.layer = tf.keras.layers.Dense(out_features, activation=self.activation)

    def update(self):
        _update_tf_weights(self.layer, self.learning_rate)

    def __call__(self, x):
        return self.layer(super().__call__(x))

    def get_copy(self):
        c = super().get_copy()
        c.layer = _copy_keras_layer(c.layer)
        return c

class Conv(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float, out_features:int, allowed_activations=None, **kwargs):
        kernel_size = int(prior.out_features - out_features + 1)
        super().__init__(prior, learning_rate, out_features, allowed_activations, kwargs=kwargs)
        self.layer = tf.keras.layers.Conv1D(1, kernel_size, data_format='channels_last', activation=self.activation)

    def update(self):
        _update_tf_weights(self.layer, self.learning_rate)

    def __call__(self, x):
        x = super().__call__(x)
        expand = len(x.shape) < 3
        if expand:
            x = tf.expand_dims(x, axis=-1)
        y = self.layer(x)
        if expand:
            y = tf.squeeze(y, axis=-1)
        return y

    def get_copy(self):
        c = super().get_copy()
        c.layer = _copy_keras_layer(c.layer)
        return c

class Attn(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float, out_features:int, allowed_activations=None, **kwargs):
        super().__init__(prior, learning_rate, out_features, allowed_activations, kwargs=kwargs)
        self.attn = tf.keras.layers.Attention()
        self.W_q = tf.keras.layers.Dense(out_features, activation=self.activation)
        self.W_k = tf.keras.layers.Dense(out_features, activation=self.activation)
        self.W_v = tf.keras.layers.Dense(out_features, activation=self.activation)

    def update(self):
        _update_tf_weights(self.W_q, self.learning_rate)
        _update_tf_weights(self.W_k, self.learning_rate)
        _update_tf_weights(self.W_v, self.learning_rate)

    def __call__(self, x):
        x = super().__call__(x)
        if len(x.shape) != 3:
            x = tf.expand_dims(x, axis=-1)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        y = self.attn([q, v, k])
        return y

    def get_copy(self):
        c = super().get_copy()
        c.attn = _copy_keras_layer(c.attn)
        c.W_q = _copy_keras_layer(c.W_q)
        c.W_k = _copy_keras_layer(c.W_k)
        c.W_v = _copy_keras_layer(c.W_v)
        return c

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

        return scale * (x - tf.reduce_mean(x)) / tf.math.reduce_std(x) + offset
        x = x - tf.reduce_mean(x)
        x /= tf.math.reduce_std(x)
        return x

class SkipConn(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float, out_features:int, skip_from:AbstractLayer, allowed_activations=None, **kwargs):
        super().__init__(prior, learning_rate, out_features, allowed_activations)
        self.prior_to_out = tf.keras.layers.Dense(out_features, activation=self.activation)
        self.skip_to_out = tf.keras.layers.Dense(out_features, activation=self.activation)
        self.flatten = tf.keras.layers.Flatten()
        self.skip_from = skip_from

    def update(self):
        _update_tf_weights(self.prior_to_out, self.learning_rate)
        _update_tf_weights(self.skip_to_out, self.learning_rate)

    def __call__(self, x):
        x1 = super().__call__(x)
        x1 = self.flatten(x1)
        x1 = self.prior_to_out(x1)

        x2 = self.skip_from(x)
        x2 = self.flatten(x2)
        x2 = self.skip_to_out(x2)

        return x1 + x2

    def get_copy(self):
        c = super().get_copy()
        c.prior_to_out = _copy_keras_layer(c.prior_to_out)
        c.skip_to_out = _copy_keras_layer(c.skip_to_out)
        c.flatten = _copy_keras_layer(c.flatten)
        c.skip_from = self.iter().index(self.skip_from)
        return c

if __name__ == '__main__':
    # Example network
    x = Input((3,)) # input could be a multidim vector (e.g., (5,3) you have have 5 embeddings with dims of 3)
    y = BatchNorm(x, 0.1)
    y = SkipConn(y, 0.1, 5, x) # skip conn from input
    y = Attn(y, 0.1, 3)
    y = Dense(y, 0.1, 5)
    y = Conv(y, 0.1, 3)
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
    model2 = layers2[-1]
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

    #testing pickling
    import dill as pickle
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print( # should be same as last model_1 call
        'model_1: Should be same as last model_1 call:',
        model([
            [1,-5,3],
            [-1,10,1]
        ])
    )