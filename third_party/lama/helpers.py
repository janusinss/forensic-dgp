import torch.nn as nn


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class LearnableSpatialTransformWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        raise ValueError("Spatial-transform configurations are outside the pinned Big-LaMa inference path")
