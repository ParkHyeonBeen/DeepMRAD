from collections import deque

import numpy as np

from Network.Model_Network import *
from Common.Utils import weight_init

class Ensemble(nn.Module):
    def __init__(self, base_model, score_len, ensemble_size=2, model_batch_size=5):
        super(Ensemble, self).__init__()

        self.ensemble_size = ensemble_size
        self.model_batch_size = model_batch_size

        self.model_batch = []
        self.score_list = {}
        self.ensemble_list = []
        self.ensemble_model = []
        self.buffer = []

        for i in range(self.model_batch_size):
            _model = base_model
            _model.apply(weight_init)
            self.model_batch.append(_model)
            self.score_list[i] = deque(maxlen=score_len)

        self.base_model = self.model_batch[0]

    def forward(self, *input_val):
        z_list = []
        for model in self.ensemble_model:
            if isinstance(self.base_model, DynamicsNetwork):
                state, action = input_val
                z = model(state, action)
            elif isinstance(self.base_model, InverseDynamicsNetwork):
                state_d, next_state = input_val
                z = model(state_d, next_state)
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
            cost, mse, kl = model.eval_model(state, action, next_state)
            self._add(idx, (model, cost, mse, kl))
        esemble_list = self._select_bests()
        state_d = (next_state - state)/self.base_model.frameskip

        state_d = torch.tensor(state_d, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.float).cuda()

        if isinstance(self.base_model, DynamicsNetwork):
            z_esb, kl_esb = self.get_esb_val(state, action)
            state_d = F.softsign(state_d)
            mse_esb = self.base_model.mse_loss(z_esb, state_d)
        elif isinstance(self.base_model, InverseDynamicsNetwork):
            z_esb, kl_esb = self.get_esb_val(state_d, next_state)
            mse_esb = self.base_model.mse_loss(z_esb, action)
        else:
            raise Exception("check whether the model is included in model network")

        mse_esb = mse_esb.cpu().detach().numpy()
        cost_esb = mse_esb + self.base_model.kl_weight * kl_esb

        return cost_esb, mse_esb, kl_esb

    def get_esb_val(self, *input_val):
        z_list, kl_list, mse_list = [], [], []
        for data in self.ensemble_list:
            model, _, mse, kl = data
            if isinstance(model, DynamicsNetwork):
                s, a = input_val
                z = model(s, a)
            elif isinstance(model, InverseDynamicsNetwork):
                s_d, ns = input_val
                z = model(s_d, ns)
            else:
                raise Exception("check whether the model is included in model network")
            z_list.append(z)
            kl_list.append(kl)
            mse_list.append(mse)

        z_esb = sum(z_list) / self.ensemble_size
        kl_esb = sum(kl_list) / self.ensemble_size

        return z_esb, kl_esb

    def save_ensemble(self, fname : str, root : str):

        ensemble_models = {}
        for i, data in enumerate(self.ensemble_list):
            model, _, _, _ = data
            ensemble_models['ensemble' + str(i+1)] = model.state_dict()

        if "DNN" in fname:
            path = root + "saved_net/model/DNN/" + fname
        elif "BNN" in fname:
            path = root + "saved_net/model/BNN/" + fname
        else:
            path = root + "saved_net/model/Etc/" + fname

        torch.save(ensemble_models, path)

    def load_ensemble(self, path : str):
        issingle = False
        load_ensemble = torch.load(path)
        if issingle is True:
            print("use only one model in ensemble list")
            self.base_model.load_state_dict(load_ensemble['ensemble' + str(1)])
            self.ensemble_model.append(self.base_model)
        else:
            load_model = self.model_batch[:self.ensemble_size]
            for i, model in enumerate(load_model):
                model.load_state_dict(load_ensemble['ensemble' + str(i+1)])
                self.ensemble_model.append(model)
        return self.ensemble_model

    def _add(self, idx, data):
        self.buffer.append(data)
        self.score_list[idx].append(data[1])

    # def _select_bests(self, compare_index=1, reverse=True):
    #     for i, data in enumerate(self.buffer):
    #         if i < self.ensemble_size:
    #             self.ensemble_list.append(data)
    #         elif i == self.ensemble_size - 1:
    #             self.ensemble_list.sort(key=lambda x: x[compare_index], reverse=reverse)
    #         else:
    #             if self.ensemble_list[0][1] > data[1]:
    #                 self.ensemble_list[0] = data
    #                 self.ensemble_list.sort(key=lambda x: x[compare_index], reverse=reverse)
    #             else:
    #                 pass
    #     return self.ensemble_list

    def _select_bests(self, reverse=True):
        score_mean_list = []
        for idx, data in enumerate(self.buffer):
            score_mean_list.append([data, sum(self.score_list[idx])/len(self.score_list[idx])])

        score_mean_list.sort(key=lambda x: x[1], reverse=reverse)

        for i in range(self.ensemble_size):
            self.ensemble_list.append(score_mean_list[i][0])
        return self.ensemble_list
