import torch
from torch import nn

class MySequential(nn.Sequential):

    def forward(self, features, *args, **kwargs):
        with torch.profiler.record_function("MySeq"):
            return super().forward(features), None, None, None, None, None

def Linear(dim_in, transformer, *args, **kwargs):
    layers = []
    layers.append(nn.BatchNorm1d(dim_in))
    layers.append(nn.ReLU())

    for _ in range(transformer.n_layers):
        layers.append(nn.Linear(dim_in, dim_in, bias=False))
        layers.append(nn.BatchNorm1d(dim_in))
        layers.append(nn.ReLU())

    return MySequential(*layers)