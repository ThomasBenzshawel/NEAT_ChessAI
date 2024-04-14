import numpy as np
import tensorflow as tf
import random

def update_tf_weights(layer, learning_rate):
    layer.set_weights([
        w + learning_rate*tf.random.normal(w.shape)
        for w in layer.weights
    ])

# Using this in place of RELU.
# It's like RELU, but it isn't exactly 0 for x<0, so it's differentiable (or traversable)
def xelu(x):
    return x / (np.exp(-x) + 1)

class AbstractLayer:
    def __init__(self, prior:"AbstractLayer", out_features:int, learning_rate:float, allowed_activations:list[str]=None):
        self.prior = prior
        self.out_features = out_features
        self.learning_rate = learning_rate
        if allowed_activations is None:
            allowed_activations = [xelu, 'sigmoid']
        self.activation = random.choice(allowed_activations)

    def update(self):
        pass

    def __call__(self, x):
        return self.prior(x)

    def iter(self, ret=None):
        if ret is None:
            ret = [self]
        return self.prior.iter() + ret

class Input(AbstractLayer):
    def __init__(self, out_features:tuple):
        super().__init__(None, out_features, 0)

    def __call__(self, x):
        x = tf.constant(x, dtype=float)
        if len(x.shape) < 2 or x.shape[1:] != self.out_features:
            out_shape_msg = ', '.join([str(d) for d in self.out_features])
            raise ValueError(f'Expected input shape (batch_size, {out_shape_msg}), but got shape: {x.shape}')
        return x

    def iter(self):
        return [self]

class Dense(AbstractLayer):
    def __init__(self, prior:AbstractLayer, out_features:int, learning_rate:float, allowed_activations=None):
        super().__init__(prior, out_features, learning_rate, allowed_activations)
        self.layer = tf.keras.layers.Dense(out_features, activation=self.activation)

    def update(self):
        update_tf_weights(self.layer, self.learning_rate)

    def __call__(self, x):
        return self.layer(super().__call__(x))

class Conv(AbstractLayer):
    def __init__(self, prior:AbstractLayer, kernel_size:int, learning_rate:float, allowed_activations=None):
        super().__init__(prior, None, learning_rate, allowed_activations)
        self.layer = tf.keras.layers.Conv1D(1, kernel_size, data_format='channels_last', activation=self.activation)

    def update(self):
        update_tf_weights(self.layer, self.learning_rate)

    def __call__(self, x):
        x = super().__call__(x)
        expand = len(x.shape) < 3
        if expand:
            x = tf.expand_dims(x, axis=-1)
        y = self.layer(x)
        if expand:
            y = tf.squeeze(y, axis=-1)
        return y

class Attn(AbstractLayer):
    def __init__(self, prior:AbstractLayer, out_features:int, learning_rate:float, allowed_activations=None):
        super().__init__(prior, out_features, learning_rate, allowed_activations)
        self.attn = tf.keras.layers.Attention()
        self.W_q = tf.keras.layers.Dense(out_features, activation=self.activation)
        self.W_k = tf.keras.layers.Dense(out_features, activation=self.activation)
        self.W_v = tf.keras.layers.Dense(out_features, activation=self.activation)

    def update(self):
        update_tf_weights(self.W_q, self.learning_rate)
        update_tf_weights(self.W_k, self.learning_rate)
        update_tf_weights(self.W_v, self.learning_rate)

    def __call__(self, x):
        x = super().__call__(x)
        if len(x.shape) != 3:
            x = tf.expand_dims(x, axis=-1)
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        y = self.attn([q, v, k])
        return y

class BatchNorm(AbstractLayer):
    def __init__(self, prior:AbstractLayer, learning_rate:float):
        super().__init__(prior, prior.out_features, learning_rate)
        self.gamma = np.log(np.e - 1)
        self.beta = 0

    def update(self):
        self.beta += self.learning_rate*self.beta
        self.gamma += self.learning_rate*self.gamma

    def __call__(self, x):
        x = super().__call__(x)

        scale = np.log(np.exp(self.gamma)+1)
        offset = self.beta

        return scale * (x - tf.reduce_mean(x)) / tf.math.reduce_std(x) + offset
        x = x - tf.reduce_mean(x)
        x /= tf.math.reduce_std(x)
        return x

class SkipConn(AbstractLayer):
    def __init__(self, prior:AbstractLayer, out_features:int, learning_rate:float, skip_from:AbstractLayer, allowed_activations=None):
        super().__init__(prior, out_features, learning_rate, allowed_activations)
        self.prior_to_out = tf.keras.layers.Dense(out_features, activation=self.activation)
        self.skip_to_out = tf.keras.layers.Dense(out_features, activation=self.activation)
        self.flatten = tf.keras.layers.Flatten()
        self.skip_from = skip_from

    def update(self):
        update_tf_weights(self.prior_to_out, self.learning_rate)
        update_tf_weights(self.skip_to_out, self.learning_rate)

    def __call__(self, x):
        x1 = super().__call__(x)
        x1 = self.flatten(x1)
        x1 = self.prior_to_out(x1)

        x2 = self.skip_from(x)
        x2 = self.flatten(x2)
        x2 = self.skip_to_out(x2)

        return x1 + x2

if __name__ == '__main__':
    # Example network
    x = Input((3,)) # input could be a multidim vector (e.g., (5,3) you have have 5 embeddings with dims of 3)
    y = BatchNorm(x, 0.1)
    y = SkipConn(y, 5, 0.1, x) # skip conn from input
    y = Attn(y, 3, 0.1)
    y = Dense(y, 5, 0.1)
    y = Conv(y, 3, 0.1)
    z = BatchNorm(y, 0.1)
    y = Dense(z, 4, 0.1)
    y = SkipConn(y, 4, 0.1, z)
    model = y

    # Prediction
    # Prediction from a layer X first goes through all prior layers: X(x) = X <- X-1 <- X-2 <- ... <- Input(x).
    print(
        # Predict on batches at a time
        model([ # you can also pass a numpy array
            [1,-5,3],
            [-1,10,1]
        ])
    )

    # Update
    for l in model.iter():
        l.update()
    # prediction should be slightly different
    print(
        model([
            [1,-5,3],
            [-1,10,1]
        ])
    )