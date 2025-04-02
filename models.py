import torch
from torch import nn
import numpy as np

from abc import abstractmethod
from collections.abc import Iterable

from layers import (
    Input,
    Dense,
    BatchNorm,
    Attn,
    Conv,
    SkipConn,

    SupervisedLayer,
    SupervisedDenseLayer,
    SupervisedConvLayer,
)

acc_device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {acc_device} as accelerator device")

class Model:
    """Interface to define behaviors of Neural Networks for
    purposes of NEAT.
    """
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def add_layer(self, layer_type, index, out_features):
        pass

    @abstractmethod
    def remove_layer(self, index):
        pass

    @abstractmethod
    def small_update(self, **params):
        pass

    def __call__(self, X):
        return self.forward(X)


class SupervisedModel(nn.Module, Model):
    def __init__(
            self,
            input_shape,
            n_outputs,
            output_activation='relu',
        ):
        super().__init__()
        self._input_shape = input_shape
        self._n_outputs = n_outputs
        self._layers = []
        self._remove_flatten = False
        if isinstance(input_shape, Iterable):
            self._remove_flatten = True
            self._layers.append(nn.Flatten())
        self.output_layer = SupervisedDenseLayer(input_shape, n_outputs, activation=output_activation)
        self._model = nn.Sequential(*self._layers, self.output_layer)

    def depth(self):
        return len(self._layers)
    
    def forward(self, X):
        return self._model(X)
    
    def remove_layer(self, index):
        if not 0 <= index < len(self._layers):
            raise ValueError(f"Cannot delete a layer outside of depth")
        prev_layer = self._layers[index-1] if index > 0 else None
        next_layer = self._layers[index+1] if index < len(self._layers) - 1 else self.output_layer
        next_layer.input_shape = prev_layer.output_shape if prev_layer != None else self._input_shape
        self._layers.pop(index)
        self._model = nn.Sequential(*self._layers, self.output_layer)

    def add_layer(
            self,
            layer_type,
            index,
            out_features,
            activation='relu',
            kernel_size=3, # only used if layer_type is Conv
        ):
        if not 0 <= index <= len(self._layers):
            raise ValueError(f"Cannot insert outside of depth (depth={self.depth()},attempted insert={index})")
        prev_layer: SupervisedLayer = self._layers[index-1] if index > 0 else None
        next_layer: SupervisedLayer = self._layers[index] if index < len(self._layers) else self.output_layer
        in_shape = prev_layer.output_shape if prev_layer != None else self._input_shape
        if layer_type == Dense:
            l = SupervisedDenseLayer(in_shape, out_features, activation=activation)
        elif layer_type == BatchNorm:
            l = nn.BatchNorm1d(in_shape[-1] if isinstance(in_shape, Iterable) else in_shape)
        elif layer_type == Conv:
            flatten = type(next_layer) != SupervisedConvLayer
            l = SupervisedConvLayer(in_shape,
                                    kernel_size,
                                    out_features,
                                    flatten=flatten,
                                    activation=activation)
        elif layer_type == SkipConn:
            raise NotImplementedError("Skipp Connections are not currently supported in supervised mode")
        
        next_layer.input_shape = l.output_shape
        self._layers.insert(index, l)
        self._model = nn.Sequential(*self._layers, self.output_layer)

class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df, features, targets):
        self.df = df
        columns = list(df.columns)
        self.features = [columns.index(f) for f in features]
        self.targets = [columns.index(t) for t in targets]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        feature_values = torch.tensor(self.df.iloc[idx, self.features].values, dtype=torch.float32, requires_grad=False)
        target_values = torch.tensor(self.df.iloc[idx, self.targets], dtype=torch.float32, requires_grad=False)
        return feature_values, target_values

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.model_selection import train_test_split
    import os

    def gen_xor(N_samples):
        X = (np.random.random((N_samples, 2)) > .5).astype(int)
        y = np.logical_xor(X[:,0], X[:,1])
        df = pd.DataFrame(np.hstack([X, y.reshape((N_samples, 1))]), columns=['A', 'B', 'y'])
        df.to_csv("test_xor_set.csv")

    if not os.path.exists("test_xor_set.csv"):
        gen_xor(100_000)
    df = pd.read_csv("test_xor_set.csv", index_col=0)
    n_features = df.drop(columns=['y']).shape[1]
    features = list(df.columns)
    features.remove('y')
    df['y0'] = (df['y'] == 0).astype(int)
    df['y1'] = (df['y'] == 1).astype(int)
    targets = ['y0', 'y1']

    train_set, test_set = train_test_split(df, test_size=.2)
    train_dataset = DataFrameDataset(train_set, features, targets)
    test_dataset = DataFrameDataset(test_set, features, targets)

    model = SupervisedModel(n_features, 2, output_activation='softmax')
    model.add_layer(Dense, 0, 64, activation='relu')
    # model.add_layer(Dense, 0, 128, activation='relu')
    # model.add_layer(Dense, 1, 64, activation='relu')
    # model.add_layer(Dense, 2, 16, activation='relu')
    # model.add_layer(Dense, 3, 16, activation='relu')

    # model.remove_layer(1)
    print("Model Depth:", model.depth())
    # print("Model Parameters:", [p for p in model.parameters()])
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    loss_fn = nn.CrossEntropyLoss()

    # train model
    EPOCHS = 10
    BATCH_SIZE = 254
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

    losses = []
    # accuracies = []
    full_features = torch.from_numpy(df[features].to_numpy()).float()
    full_targets = torch.from_numpy(df[targets].to_numpy()).float()

    model.train()
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        for batch, (X, y) in enumerate(train_dataloader):
            pred = model(X)
            # print(pred)
            loss = loss_fn(pred, y.resize_((y.shape[0], len(targets))))
            accuracy = (pred.argmax(dim=1) == y.argmax(dim=1)).sum() / pred.shape[0]

            optimizer.zero_grad()
            loss.backward()
            # for layer in model._model:
            #     for name, param in model._model.named_parameters():
            #         if param.grad is not None:
            #             print(f"Layer: {name},\tGradient sum: {param.grad.sum()}")
            #         else:
            #             print(f"Layer {name} has no gradient.")
            # print(model._model.named_parameters()[1].grad.sum())
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * BATCH_SIZE + len(X)
                print(f"\tloss: {loss:>7f} [{current:>5d}/{len(train_dataloader.dataset)}]\tbatch {batch:>5d} accuracy: {accuracy}")
        loss, current = loss.item(), batch * BATCH_SIZE + len(X)
        print(f"\tloss: {loss:>7f} [{current:>5d}/{len(train_dataloader.dataset)}]")
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
        pred = model(full_features)
        # accuracy = (pred == full_targets).sum() / pred.shape[0]
        loss_val = loss_fn(pred, full_targets.resize_((full_targets.shape[0],len(targets)))).item()
        print(f"Loss over epoch {epoch}: {loss_val}")
        losses.append(loss_val)
        # accuracies.append(accuracy)
    
    plt.plot(range(EPOCHS), losses)
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Loss Across Epochs")
    plt.savefig("model_loss_test.png")

    # plt.plot(range(EPOCHS), accuracies)
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.ylim(0,1)
    # plt.title("Accuracy Across Epochs")
    # plt.savefig("model_accuracy_test.png")
