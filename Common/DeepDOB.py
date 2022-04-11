import collections

import torch
import mujoco_py
from Common.Utils import *

class DeepDOB:
    def __init__(self, inv_model_net, test_env, steps_inloop, args_tester, frameskip, action_dim, max_action, min_action):

        self.inv_model_net = inv_model_net
        self.test_env = test_env
        self.steps_inloop = steps_inloop
        self.args_tester = args_tester
        self.frameskip = frameskip

        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action

        self.state = None
        self.done = False
        self.next_state = None
        self.disturbance_estm_n = None
        self.pred_list = collections.deque(maxlen=1)

        self.noise_data = DataManager()
        self.disturbance_data = DataManager(data_name='disturbance')
        self.estimate_data = DataManager(data_name='estimate')

    def reset(self):
        self.state = self.test_env.reset()
        self.disturbance_estm_n = torch.tensor(np.zeros(self.action_dim), dtype=torch.float).cuda().detach()
        self.done = False

        return self.state

    def step(self, env_action, local_step):
        with torch.no_grad():
            reward = 0
            action_tensor = torch.tensor(normalize(env_action, self.max_action, self.min_action), dtype=torch.float).cuda()

            for i in range(self.steps_inloop):
                if i == 0:
                    state_inner = self.state
                # action step
                env_action_dob_n = action_tensor - self.disturbance_estm_n
                env_action_dob = denormalize(env_action_dob_n, self.max_action, self.min_action)
                if self.args_tester.add_noise is True and self.args_tester.noise_to == 'action':
                    env_action_real, noise_list = add_noise(env_action_dob, scale=self.args_tester.noise_scale)

                if self.args_tester.add_disturbance is True and self.args_tester.disturbance_to == 'action':
                    env_action_real, disturbance_list = add_disturbance(env_action_dob, local_step,
                                                     self.test_env.spec.max_episode_steps,
                                                     scale=self.args_tester.disturbance_scale,
                                                     frequency=self.args_tester.disturbance_frequency)
                # real system
                if self.args_tester.noise_to == 'state':
                    env_action_real = env_action_dob

                env_action_real_npy = env_action_real.cpu().detach().numpy()
                # self.estimate_data.plot_data((env_action_real_npy - env_action)[0])
                next_state_inner, reward_inner, self.done, info = self.test_env.step(env_action_real_npy, inner_loop=self.steps_inloop)
                if self.args_tester.add_noise is True and self.args_tester.noise_to == 'state':
                    next_state_inner, noise_list = add_noise(next_state_inner, scale=self.args_tester.noise_scale)
                if self.args_tester.add_disturbance is True and self.args_tester.disturbance_to == 'state':
                    next_state_inner, disturbance_list = add_disturbance(next_state_inner, local_step,
                                                     self.test_env.spec.max_episode_steps,
                                                     scale=self.args_tester.disturbance_scale,
                                                     frequency=self.args_tester.disturbance_frequency)

                if self.done is True:
                    break
                reward += reward_inner

                # predict disturbance
                state_d = (next_state_inner - state_inner) / self.frameskip

                self.disturbance_estm_n = torch.tanh(1.1*(self.inv_model_net(state_d, next_state_inner) - action_tensor))# - 0.1*torch.ones_like(env_action_dob_n).cuda()))
                # self.disturbance_estm_n = self.inv_model_net(state_d, next_state_inner) - action_tensor
                self.pred_list.append(self.disturbance_estm_n)
                self.disturbance_estm_n = sum(self.pred_list)/len(self.pred_list)
                # self.disturbance_data.plot_data(self.disturbance_estm_n[0].cpu().detach().numpy())
                state_inner = next_state_inner

            remain_frameskip = self.test_env.env.frame_skip_origin % self.args_tester.frameskip_inner
            if remain_frameskip != 0:
                self.test_env.env.do_simulation(env_action, remain_frameskip)
                self.next_state = self.test_env.env._get_obs()
            else:
                self.next_state = state_inner

            reward = reward/self.steps_inloop

            self.state = self.next_state
        return self.next_state, reward, self.done, info

    def save_data(self):
        log_path = '/media/phb/Storage/env_mbrl/Results/Integrated_log/'
        # self.disturbance_data.save_data(log_path, "disturbance2" + self.args_tester.disturbance_to+"_"+str(self.args_tester.disturbance_scale))
        # self.noise_data.save_data(log_path, "noise2" + self.args_tester.noise_to+"_"+str(self.args_tester.noise_scale))
        self.estimate_data.save_data(log_path, "estimate4" + self.args_tester.disturbance_to+"_"+str(self.args_tester.disturbance_scale)+str(self.args_tester.noise_scale))