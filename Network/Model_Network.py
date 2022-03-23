import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchbnn as bnn
from torchbnn.utils import freeze, unfreeze

import collections
from Common.Utils import weight_init

class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, frameskip, algorithm, args,
                 net_type=None, hidden_dim=(256, 256)):
        super(DynamicsNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.frameskip = frameskip

        self.args = args

        self.buffer = algorithm.buffer
        self.batch_size = args.batch_size

        if self.net_type == "DNN":

            self.dnmsNNin = nn.ModuleList([nn.Linear(self.state_dim + self.action_dim, hidden_dim[0]), nn.ReLU()])
            for i in range(len(hidden_dim) - 1):
                self.dnmsNNin.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
                self.dnmsNNin.append(nn.ReLU())
            self.dnmsNNout = nn.Linear(hidden_dim[-1], state_dim)
            self.dnmsNN = self.dnmsNNin.append(self.dnmsNNout).cuda()

        if self.net_type == "BNN":
            self.dnmsNNin = nn.ModuleList([
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=(self.state_dim + self.action_dim), out_features=256),
                nn.ReLU()])
            for i in range(2):
                self.dnmsNNin.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim[0], out_features=hidden_dim[0]))
                self.dnmsNNin.append(nn.ReLU())
            self.dnmsNNout = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim[-1], out_features=self.state_dim)
            self.dnmsNN = self.dnmsNNin.append(self.dnmsNNout).cuda()

        self.dnms_optimizer = optim.Adam(self.dnmsNN.parameters(), lr=args.model_lr)
        self.dnms_out_optimizer = optim.Adam(self.dnmsNNout.parameters(), lr=0.0001)

        self.scheduler = lr_scheduler.ExponentialLR(self.dnms_optimizer, gamma=0.99)

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = args.model_kl_weight

        self.apply(weight_init)

    def forward(self, state, action, train=False):

        if type(state) is not torch.Tensor:
            state = torch.tensor(state, dtype=torch.float).cuda()
        if type(action) is not torch.Tensor:
            action = torch.tensor(action, dtype=torch.float).cuda()

        state = F.softsign(state)

        if train is True:
            z = torch.cat([state, action], dim=1)
        else:
            z = torch.cat([state, action])

        for i in range(len(self.dnmsNN)):
            z = self.dnmsNN[i](z)

        z = F.softsign(z)
        return z

    def train_all(self, training_num):
        cost = 0.0
        mse = 0.0
        kl = 0.0

        for i in range(training_num):

            s, a, _, ns, _ = self.buffer.sample(self.batch_size)
            s_d = (ns - s) / self.frameskip
            s_d = F.softsign(s_d)

            z = self.forward(s, a, train=True)
            mse = self.mse_loss(z, s_d)
            kl = self.kl_loss(self.dnmsNN)
            cost = mse + self.kl_weight * kl

            self.dnms_optimizer.zero_grad()
            cost.backward()
            self.dnms_optimizer.step()
        # self.scheduler.step()

        cost = cost.cpu().detach().numpy()
        mse = mse.cpu().detach().numpy()
        kl = kl.cpu().detach().numpy()
        return cost, mse, kl

    def eval_model(self, state, action, next_state):

        state = torch.tensor(state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        state_d = (next_state - state)/self.frameskip
        # print("state", state, "action", action, "next state", next_state, "state dif", state_d)

        cost = 0.0
        mse = 0.0
        kl = 0.0

        if self.net_type == "DNN":
            z = self.forward(state, action)
            state_d = F.softsign(state_d)
            mse = self.mse_loss(z, state_d)
            kl = self.kl_loss(self.dnmsNN)
            cost = mse + self.kl_weight * kl

        if self.net_type == "BNN":
            freeze(self.dnmsNN)
            z = self.forward(state, action)
            state_d = F.softsign(state_d)
            mse = self.mse_loss(z, state_d)
            kl = self.kl_loss(self.dnmsNN)
            cost = mse + self.kl_weight * kl
            unfreeze(self.dnmsNN)

        cost = cost.cpu().detach().numpy()
        mse = mse.cpu().detach().numpy()
        kl = kl.cpu().detach().numpy()
        return cost, mse, kl

class InverseDynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim,
                 frameskip, algorithm,
                 args, net_type=None, hidden_dim=(256, 256)):
        super(InverseDynamicsNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.frameskip = frameskip

        self.args = args

        self.buffer = algorithm.buffer
        self.batch_size = args.batch_size

        if self.net_type == "DNN":
            self.inv_dnmsNNin = nn.ModuleList([nn.Linear(2*self.state_dim, hidden_dim[0]), nn.ReLU()])
            for i in range(len(hidden_dim) - 1):
                self.inv_dnmsNNin.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
                self.inv_dnmsNNin.append(nn.ReLU())
            self.inv_dnmsNNout = nn.Linear(hidden_dim[-1], action_dim)
            self.inv_dnmsNN = self.inv_dnmsNNin.append(self.inv_dnmsNNout).cuda()

        if self.net_type == "BNN":
            self.inv_dnmsNNin = nn.ModuleList([
                bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=(2*self.state_dim), out_features=256),
                nn.ReLU()])
            for i in range(2):
                self.inv_dnmsNNin.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim[0], out_features=hidden_dim[0]))
                self.inv_dnmsNNin.append(nn.ReLU())
            self.inv_dnmsNNout = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim[-1], out_features=self.action_dim)
            self.inv_dnmsNN = self.inv_dnmsNNin.append(self.inv_dnmsNNout).cuda()

        self.inv_dnms_optimizer = optim.Adam(self.inv_dnmsNN.parameters(), lr=args.inv_model_lr)
        self.inv_dnms_out_optimizer = optim.Adam(self.inv_dnmsNNout.parameters(), lr=0.0001)

        self.scheduler = lr_scheduler.ExponentialLR(self.inv_dnms_optimizer, gamma=0.99)

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = args.inv_model_kl_weight

        self.apply(weight_init)

    def forward(self, state_d, next_state, train=False):

        if type(state_d) is not torch.Tensor:
            state_d = torch.tensor(state_d, dtype=torch.float).cuda()
        if type(next_state) is not torch.Tensor:
            next_state = torch.tensor(next_state, dtype=torch.float).cuda()

        state_d = F.softsign(state_d)
        next_state = F.softsign(next_state)

        if train is True:
            z = torch.cat([state_d, next_state], dim=1)
        else:
            z = torch.cat([state_d, next_state])

        for i in range(len(self.inv_dnmsNN)):
            z = self.inv_dnmsNN[i](z)

        z = torch.tanh(z)
        return z

    def train_all(self, training_num):
        cost = 0.0
        mse = 0.0
        kl = 0.0

        for i in range(training_num):

            s, a, _, ns, _ = self.buffer.sample(self.batch_size)
            s_d = (ns - s)/self.frameskip

            z = self.forward(s_d, ns, train=True)
            mse = self.mse_loss(z, a)
            kl = self.kl_loss(self.inv_dnmsNN)
            cost = mse + self.kl_weight * kl

            self.inv_dnms_optimizer.zero_grad()
            cost.backward()
            self.inv_dnms_optimizer.step()
        # self.scheduler.step()

        cost = cost.cpu().detach().numpy()
        mse = mse.cpu().detach().numpy()
        kl = kl.cpu().detach().numpy()
        return cost, mse, kl


    def eval_model(self, state, action, next_state):

        state = torch.tensor(state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()
        state_d = (next_state - state)/self.frameskip
        # print("state", state, "action", action, "next state", next_state, "state dif", state_d)
        cost = 0.0
        mse = 0.0
        kl = 0.0

        if self.net_type == "DNN":
            z = self.forward(state_d, next_state)
            mse = self.mse_loss(z, action)
            kl = self.kl_loss(self.inv_dnmsNN)
            cost = mse + self.kl_weight * kl

        if self.net_type == "BNN":
            freeze(self.inv_dnmsNN)
            z = self.forward(state_d, next_state)
            mse = self.mse_loss(z, action)
            kl = self.kl_loss(self.inv_dnmsNN)
            cost = mse + self.kl_weight * kl
            unfreeze(self.inv_dnmsNN)

        cost = cost.cpu().detach().numpy()
        mse = mse.cpu().detach().numpy()
        kl = kl.cpu().detach().numpy()
        return cost, mse, kl


if __name__ == '__main__':
    pass