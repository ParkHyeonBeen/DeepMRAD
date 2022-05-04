from collections import deque

import numpy as np

from Network.Model_Network import *
from Common.Utils import weight_init

class Ensemble(nn.Module):
    def __init__(self, model_type, state_dim, action_dim, frameskip, algorithm, args, buffer, net_type, score_len, ensemble_size=2, model_batch_size=5):
        super(Ensemble, self).__init__()

        self.ensemble_size = ensemble_size
        self.model_batch_size = model_batch_size

        self.model_batch = []
        self.score_list = {}
        self.ensemble_list = []
        self.loaded_model = []
        self.buffer = []

        for i in range(self.model_batch_size):
            if model_type == 'modelNN':
                _model = DynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, buffer=buffer, net_type=net_type)
            if model_type == 'inv_modelNN':
                _model = InverseDynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, buffer=buffer, net_type=net_type)
            _model.apply(weight_init)
            self.model_batch.append(_model)
            self.score_list[i] = deque(maxlen=score_len)

        self.base_model = self.model_batch[0]

    def forward(self, *input_val):
        z_list = []
        for model in self.ensemble_list:
            if isinstance(self.base_model, DynamicsNetwork):
                state, action = input_val
                z = model(state, action)
            elif isinstance(self.base_model, InverseDynamicsNetwork):
                state, next_state = input_val
                z = model(state, next_state)
            else:
                raise Exception("check whether the model is included in model network")
            z_list.append(z)
        z_esb = sum(z_list) / self.ensemble_size
        return z_esb

    def train_all(self, training_num):
        cost_list, mse_list, kl_list = [], [], []
        for model in self.model_batch:
            cost, mse, kl = model.train_all(training_num)
            cost_list.append(cost)
            mse_list.append(mse)
            kl_list.append(kl)

        cost_mean = sum(cost_list) / self.ensemble_size
        mse_mean = sum(mse_list) / self.ensemble_size
        kl_mean = sum(kl_list) / self.ensemble_size

        return cost_mean, mse_mean, kl_mean

    def eval_model(self, state, action, next_state):

        self.buffer = []
        self.ensemble_list = []

        for idx, model in enumerate(self.model_batch):
            error = model.eval_model(state, action, next_state)
            self._add(idx, (model, error))
        esemble_list = self._select_bests()
        state_d = (next_state - state)/self.base_model.frameskip

        state_d = torch.tensor(state_d, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.float).cuda()

        if isinstance(self.base_model, DynamicsNetwork):
            z_esb = self.forward(state, action)
            state_d = F.softsign(state_d)
            error_esb = torch.max(torch.abs(z_esb - state_d))
        elif isinstance(self.base_model, InverseDynamicsNetwork):
            z_esb = self.forward(state, next_state)
            error_esb = torch.max(torch.abs(z_esb - action))
        else:
            raise Exception("check whether the model is included in model network")

        error_esb = error_esb.cpu().detach().numpy()

        return error_esb

    def save_ensemble(self, fname : str, root : str):

        ensemble_models = {}
        for i, model in enumerate(self.model_batch):
            ensemble_models['ensemble' + str(i+1)] = model.state_dict()

        if "DNN" in fname:
            path = root + "saved_net/model/DNN/" + fname
        elif "BNN" in fname:
            path = root + "saved_net/model/BNN/" + fname
        else:
            path = root + "saved_net/model/Etc/" + fname

        torch.save(ensemble_models, path)

    def load_ensemble(self, path : str, ensemble_size):

        load_ensemble = torch.load(path)
        load_model = self.model_batch[:ensemble_size]
        for i, model in enumerate(load_model):
            model.load_state_dict(load_ensemble['ensemble' + str(i+1)])
            self.loaded_model.append(model)
        return self.loaded_model

    def _add(self, idx, data):
        self.buffer.append(data[0])
        self.score_list[idx].append(data[1])

    def _select_bests(self):
        score_mean_list = []
        for idx, model in enumerate(self.buffer):
            score_mean_list.append([model, sum(self.score_list[idx])/len(self.score_list[idx])])

        score_mean_list.sort(key=lambda x: x[1])

        for i in range(self.ensemble_size):
            self.ensemble_list.append(score_mean_list[i][0])

        return self.ensemble_list
