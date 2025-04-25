import torch
from torch import nn
import numpy as np

from abc import abstractmethod
from collections.abc import Iterable

from layers import (
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
        # self._remove_flatten = False
        # if isinstance(input_shape, Iterable):
        #     self._remove_flatten = True
        #     self._layers.append(nn.Flatten())
        self.output_layer = SupervisedDenseLayer(input_shape, n_outputs, activation=output_activation)
        self._model = nn.Sequential(*self._layers, self.output_layer)

    def depth(self):
        return len(self._layers)
    
    def forward(self, X):
        return self._model(X)
    
    def set_shapes(self):
        if len(self._layers) > 0:
            self._layers[0].input_shape = self._input_shape
            for i, layer in enumerate(self._layers[1:]):
                layer.input_shape = self._layers[i].output_shape
            self.output_layer.input_shape = self._layers[-1].output_shape
        else:
            self.output_layer.input_shape = self._input_shape
    
    def remove_layer(self, index):
        if not 0 <= index < len(self._layers):
            raise ValueError(f"Cannot delete a layer outside of depth")
        prev_layer = self._layers[index-1] if index > 0 else None
        next_layer = self._layers[index+1] if index < len(self._layers) - 1 else self.output_layer
        next_layer.input_shape = prev_layer.output_shape if prev_layer != None else self._input_shape
        self._layers.pop(index)
        self.set_shapes()
        self._model = nn.Sequential(*self._layers, self.output_layer)

    def add_layer(
            self,
            layer_type,
            index,
            out_features,
            activation='relu',
            kernel_size=3, # only used if layer_type is Conv
            padding=0, # only used if layer_type is Conv
        ):
        if not 0 <= index <= len(self._layers):
            raise ValueError(f"Cannot insert outside of depth (depth={self.depth()},attempted insert={index})")
        prev_layer: SupervisedLayer = self._layers[index-1] if index > 0 else None
        # next_layer: SupervisedLayer = self._layers[index] if index < len(self._layers) else self.output_layer
        in_shape = prev_layer.output_shape if prev_layer != None else self._input_shape
        if layer_type == Dense:
            l = SupervisedDenseLayer(in_shape, out_features, activation=activation)
        elif layer_type == BatchNorm:
            l = nn.BatchNorm1d(in_shape[-1] if isinstance(in_shape, Iterable) else in_shape)
        elif layer_type == Conv:
            l = SupervisedConvLayer(in_shape,
                                    kernel_size,
                                    out_features,
                                    activation=activation,
                                    padding=padding
                                )
        elif layer_type == SkipConn:
            raise NotImplementedError("Skipp Connections are not currently supported in supervised mode")
        
        self._layers.insert(index, l)
        
        self.set_shapes()
        self._model = nn.Sequential(*self._layers, self.output_layer)

    def append_layer(
        self,
        layer_type,
        out_features,
        activation='relu',
        kernel_size=3, # only used if layer_type is Conv
        padding=0, # only used if layer_type is Conv
    ):
        self.add_layer(layer_type, len(self._layers), out_features, activation=activation, kernel_size=kernel_size, padding=padding)

    def __str__(self):
        return " -> ".join([str(l) for l in self._layers])

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
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader

    device = torch.accelerator.current_accelerator()

    DATASET = "MNIST" # | "XOR"

    EPOCHS = 10
    BATCH_SIZE = 264

    if DATASET == "XOR":
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

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
        test_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

        full_features = torch.from_numpy(df[features].to_numpy()).float()
        full_targets = torch.from_numpy(df[targets].to_numpy()).float()
    elif DATASET == "MNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        model = SupervisedModel((1,28,28), 10, output_activation='softmax')
        model.append_layer(Conv,  128, kernel_size=6)
        model.append_layer(Conv, 1234, kernel_size=3)
        model.append_layer(Conv,  128, kernel_size=6)
        model.append_layer(Conv,  64,  kernel_size=4)
        model.append_layer(Conv,  64,  kernel_size=4)
        model.append_layer(Dense, 128, activation='relu')
        model.append_layer(Dense, 128, activation='relu')
        model.append_layer(Dense, 128, activation='relu')
        model.append_layer(Dense, 128, activation='relu')
        model.append_layer(Dense, 64,  activation='relu')

        model.remove_layer(1)

    # model.remove_layer(1)
    print("Model Depth:", model.depth())
    # move model to GPU
    model.to(device)
    # print("Model Parameters:", [p for p in model.parameters()])
    optimizer = torch.optim.Adam(model.parameters(), lr=.0005)
    loss_fn = nn.CrossEntropyLoss().to(device)

    # train model
    accuracies = []
    batch_unique_predictions = []
    # losses = []

    model.train()
    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
        total_correct = 0
        total_samples = 0
        for batch, (X, y) in enumerate(train_dataloader):
            X = X.cuda()
            y = y.cuda()
            pred = model(X)# .resize_(X.shape[0], X.shape[-2], X.shape[-1], X.shape[1])
            # print(pred)
            pred_flat = pred.argmax(dim=1)
            batch_unique_predictions.append(pred_flat.unique().size(0))
            # print("Unique Predictions:", )
            loss = loss_fn(pred, y)
            correct = (pred_flat == y).sum().item()
            batch_accuracy = correct / pred.size(0)
            total_correct += correct
            total_samples += pred.size(0)

            optimizer.zero_grad()
            loss.backward()
            # for layer in model._model:
            #     for name, param in model._model.named_parameters():
            #         if param.grad is not None:
            #             print(f"Layer: {name},\tGradient sum: {param.grad.abs().sum()}")
            #         else:
            #             print(f"Layer {name} has no gradient.")
            # print(model._model.named_parameters().grad.abs().sum())
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * BATCH_SIZE + len(X)
                print(f"\tloss: {loss:>7f} [{current:>5d}/{len(train_dataloader.dataset)}]\tbatch {batch:>5d} accuracy: {batch_accuracy}")
        loss, current = loss.item(), batch * BATCH_SIZE + len(X)
        print(f"\tloss: {loss:>7f} [{current:>5d}/{len(train_dataloader.dataset)}]")
        # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

        # pred = model(train_dataset)
        # accuracy = (pred == full_targets).sum() / pred.shape[0]
        if DATASET == "XOR":
            loss_val = loss_fn(pred, full_targets.resize_((full_targets.shape[0],len(targets)))).item()
        elif DATASET == "MNIST":
            pass
            # loss_val = loss_fn(pred, test_dataset.).item()
        accuracy = total_correct / total_samples
        print(f"Epoch {epoch} accuracy: {accuracy}")
        accuracies.append(accuracy)
        # accuracies.append(accuracy)

    plt.plot(range(EPOCHS), accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Epochs")
    plt.savefig("mnist_model_test.png")
    plt.show()

    plt.plot(range(len(batch_unique_predictions)), batch_unique_predictions)
    plt.xlabel("Batch")
    plt.ylabel("Unique Predictions")
    plt.title("Unique Predictions per Batch")
    plt.ylim(0,10.5)
    plt.savefig("mnist_unique_predictions.png")
    plt.show()

    # plt.plot(range(EPOCHS), accuracies)
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.ylim(0,1)
    # plt.title("Accuracy Across Epochs")
    # plt.savefig("model_accuracy_test.png")
