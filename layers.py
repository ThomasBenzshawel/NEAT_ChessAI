import numpy as np
import random
import pickle

def copy_model(last_layer):
    return pickle.loads(pickle.dumps(last_layer))

def _update_weights(w, r):
    w += r*np.random.normal(size=w.shape)

def softmax(xs):
    xs = np.clip(xs, -10, 10)
    xs = np.exp(xs)
    return xs / xs.sum(axis=-1, keepdims=True)

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
        self.weights = np.random.normal(size=(prior.out_features,out_features))

    def update(self):
        _update_weights(self.weights, self.learning_rate)

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
        super().__init__(prior, learning_rate, out_features, allowed_activations)
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