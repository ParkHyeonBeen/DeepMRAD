import torch
import torch.nn as nn
import torch.nn.functional as F
from Common.Utils import weight_init
from Common.Ensemble import Ensemble
import collections, random

class model(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=(10, 10), init = False):
        super(model, self).__init__()
        self.model1 = nn.ModuleList([nn.Linear(2 * state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.model1.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.model1.append(nn.ReLU())
        self.model2 = nn.Linear(hidden_dim[-1], action_dim)
        self.model3 = self.model1.append(self.model2)

        if init is True:
            for layer in self.model3:
                if not isinstance(layer, nn.ReLU):
                    layer.weight.data.fill_(0.0)
                    layer.bias.data.fill_(0.0)

esb = Ensemble(model().model3)

for layer in esb.model_ensemble:
    if not isinstance(layer, nn.ReLU):
        print(layer.weight.data)
        print(layer.bias.data)

models = []

for i in range(1000):
    _model = model().model3
    models.append(_model)
    esb.add(_model, random.random())

model_ensemble = esb.get_best()

for i, layer in enumerate(model_ensemble):
    if not isinstance(layer, nn.ReLU):
        print(layer.weight.data)