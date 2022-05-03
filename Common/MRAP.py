import torch
import torch.nn.functional as F
import torch.optim as optim
import math

from Common.Utils import *


class MRAP:
    def __init__(self, actor, model_net, test_env, args_tester, frameskip,
                 delta = 0.1,
                 error_limit = 0.5,
                 outlayer_lr = 1e-4):

        self.actor = actor
        self.model_net = model_net
        self.test_env = test_env
        self.args_tester = args_tester
        self.frameskip = frameskip

        self.delta = delta
        self.error_limit = error_limit
        self.outlayer_lr = outlayer_lr

        self.bound = None

        self.update_step = 0
        self.num_update = 0
        self.MRAP_data = DataManager()

    def eval_error(self, state, action, next_state):
        self.update_step += 1

        state_d = torch.tensor((next_state - state) / self.frameskip, dtype=torch.float).cuda()
        state_d_pred = self.model_net(state, action)

        error_softsign = F.softsign(state_d) - state_d_pred
        max_error = max(abs(error_softsign))

        self.MRAP_data.plot_data(max_error.cpu().detach().numpy())

        # print("update rate : ", self.update_step , "/", self.num_update)

        if self._is_need_update(max_error):
            self.update_cycle(max_error, self.bound)
            loss = self.update(error_softsign)
        else:
            loss = 0

        return loss

    def _is_need_update(self, max_error):
        bound_now = -1 if max_error > self.error_limit else int(torch.ceil(-torch.log10(max_error / self.delta)))
        # bound_now = 0 if bound_now < 0 else bound_now

        if self.bound is None or self.bound != bound_now or self.update_step == self.num_update:
            need = True

            self.update_step = 0
            self.bound = bound_now
        else:
            need = False
        return need

    def update_cycle(self, max_error, bound):

        if max_error <= self.delta:
            self.num_update = 10**bound
        elif self.delta < max_error <= self.error_limit:
            a = -9/(self.delta + self.error_limit)
            b = 1 + self.error_limit*9/(self.delta + self.error_limit)
            self.num_update = int(torch.round(a*max_error + b))
        else:
            self.num_update = 1

    def update(self, error):
        outlayer_optimizer = optim.Adam(self.actor.network_outer.parameters(), lr=self.outlayer_lr)

        # plot_data(self.actor.network_outer.weight[0].cpu().detach().numpy())

        loss = torch.sqrt(F.mse_loss(input=error, target=torch.zeros_like(error).cuda()))
        # self.MRAP_data.plot_data(loss.cpu().detach().numpy())

        # outlayer_optimizer.zero_grad()
        # loss.backward()
        # outlayer_optimizer.step()

        return loss

    def eval_action(self, state):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        state = np.expand_dims(np.array(state), axis=0)
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        action, _ = self.actor(state, deterministic=True)

        action = torch.clamp(action[0], min=-1., max=1.)

        return action

