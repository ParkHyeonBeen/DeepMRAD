import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchbnn as bnn
from torchbnn.utils import freeze, unfreeze

from Network.Gaussian_Actor import Squashed_Gaussian_Actor
from Common.Utils import weight_init

def OutlayerOptimizer(actor, error):
    outlayer_optimizer = optim.Adam(actor.network_outer.parameters(), lr= 0.001)

    loss = abs(torch.tensor(error, dtype=torch.float)).cuda()

    outlayer_optimizer.zero_grad()
    loss.backwward()
    outlayer_optimizer.step()

class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, algorithm, args, net_type=None, hidden_dim=(256, 256)):
        super(DynamicsNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.state_dim = state_dim
        self.action_dim = action_dim

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

        self.dnms_optimizer = optim.Adam(self.dnmsNN.parameters(), lr=0.001)
        self.dnms_out_optimizer = optim.Adam(self.dnmsNNout.parameters(), lr=0.0001)

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = 0.01

        self.apply(weight_init)

    def forward(self, state, action, train=False):

        if type(state) is not torch.Tensor or type(action) is not torch.Tensor:
            state = torch.tensor(state, dtype=torch.float).cuda()
            action = torch.tensor(action, dtype=torch.float).cuda()

        if train is True:
            z = torch.cat([state, action], dim=1)
        else:
            z = torch.cat([state, action])

        for i in range(len(self.dnmsNN)):
            z = self.dnmsNN[i](z)
        return z

    def train_all(self, training_num):
        mse = 0.0
        kl = 0.0

        for i in range(training_num):

            s, a, _, ns, _ = self.buffer.sample(self.batch_size)

            z = self.forward(s, a, train=True)
            mse = self.mse_loss(z, ns)
            kl = self.kl_loss(self.dnmsNN)
            cost = mse + self.kl_weight * kl

            self.dnms_optimizer.zero_grad()
            cost.backward()
            self.dnms_optimizer.step()

        mse = mse.cpu().detach().numpy()
        kl = kl.cpu().detach().numpy()
        return mse, kl

    def adaptive_train(self, error):
        loss = abs(torch.tensor(error, dtype=torch.float)).cuda()

        self.dnms_out_optimizer.zero_grad()
        loss.backwward()
        self.dnms_out_optimizer.step()

    def eval_model(self, state, action, next_state):

        state = torch.tensor(state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()

        cost = 0.0

        if self.net_type == "DNN":
            z = self.forward(state, action)
            mse = self.mse_loss(z, next_state)
            kl = self.kl_loss(self.dnmsNN)
            cost = mse + self.kl_weight * kl

        if self.net_type == "BNN":
            freeze(self.dnmsNN)
            z = self.forward(state, action)
            mse = self.mse_loss(z, next_state)
            kl = self.kl_loss(self.dnmsNN)
            cost = mse + self.kl_weight * kl
            unfreeze(self.dnmsNN)

        cost = cost.cpu().detach().numpy()
        return cost

class InverseDynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, algorithm, args, net_type=None, hidden_dim=(256, 256)):
        super(InverseDynamicsNetwork, self).__init__()

        if net_type is None:
            self.net_type = args.net_type
        else:
            self.net_type = net_type

        self.state_dim = state_dim
        self.action_dim = action_dim

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

        self.inv_dnms_optimizer = optim.Adam(self.inv_dnmsNN.parameters(), lr=0.001)
        self.inv_dnms_out_optimizer = optim.Adam(self.inv_dnmsNNout.parameters(), lr=0.0001)

        self.mse_loss = nn.MSELoss()
        self.kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        self.kl_weight = 0.01

        self.apply(weight_init)

    def forward(self, state, next_state, train=False):

        if type(state) is not torch.Tensor or type(next_state) is not torch.Tensor:
            state = torch.tensor(state, dtype=torch.float).cuda()
            next_state = torch.tensor(next_state, dtype=torch.float).cuda()

        if train is True:
            z = torch.cat([state, next_state], dim=1)
        else:
            z = torch.cat([state, next_state])

        print(state)
        print(next_state)

        for i in range(len(self.inv_dnmsNN)):
            z = self.inv_dnmsNN[i](z)
        return z

    def train_all(self, training_num):
        mse = 0.0
        kl = 0.0

        for i in range(training_num):

            s, a, _, ns, _ = self.buffer.sample(self.batch_size)

            z = self.forward(s, ns, train=True)
            mse = self.mse_loss(z, a)
            kl = self.kl_loss(self.inv_dnmsNN)
            cost = mse + self.kl_weight * kl

            self.inv_dnms_optimizer.zero_grad()
            cost.backward()
            self.inv_dnms_optimizer.step()

        mse = mse.cpu().detach().numpy()
        kl = kl.cpu().detach().numpy()
        return mse, kl

    def adaptive_train(self, error):

        loss = abs(torch.tensor(error, dtype=torch.float))

        self.inv_dnms_out_optimizer.zero_grad()
        loss.backwward()
        self.inv_dnms_out_optimizer.step()

    def eval_model(self, state, action, next_state):

        state = torch.tensor(state, dtype=torch.float).cuda()
        action = torch.tensor(action, dtype=torch.float).cuda()
        next_state = torch.tensor(next_state, dtype=torch.float).cuda()

        cost = 0.0

        if self.net_type == "DNN":
            z = self.forward(state, next_state)
            mse = self.mse_loss(z, action)
            kl = self.kl_loss(self.inv_dnmsNN)
            cost = mse + self.kl_weight * kl

        if self.net_type == "BNN":
            freeze(self.inv_dnmsNN)

            z = self.forward(state, next_state)
            mse = self.mse_loss(z, action)
            kl = self.kl_loss(self.inv_dnmsNN)
            cost = mse + self.kl_weight * kl
            unfreeze(self.inv_dnmsNN)

        cost = cost.cpu().detach().numpy()
        return cost

if __name__ == '__main__':
    pass