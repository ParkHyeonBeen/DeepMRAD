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
        self.disturbance_pred = None

    def reset(self):
        self.state = self.test_env.reset()
        self.disturbance_pred = torch.tensor(np.zeros(self.action_dim), dtype=torch.float).cuda()
        self.done = False

        return self.state

    def step(self, env_action, local_step):
        reward = 0
        env_action_tensor = torch.tensor(env_action, dtype=torch.float).cuda()
        action_tensor = torch.tensor(normalize(env_action, self.max_action, self.min_action), dtype=torch.float).cuda()

        for i in range(self.steps_inloop):
            if i == 0:
                state_inner = self.state
            # action step
            env_action_dob = env_action_tensor - self.disturbance_pred
            if self.args_tester.add_noise is True:
                env_action_dob = add_noise(env_action_dob, scale=self.args_tester.noise_scale)
            if self.args_tester.add_disturbance is True:
                env_action_dob = add_disturbance(env_action_dob, local_step,
                                                 self.test_env.spec.max_episode_steps,
                                                 scale=self.args_tester.disturbance_scale,
                                                 frequency=self.args_tester.disturbance_frequency)
            # real system
            env_action_real_npy = env_action_dob.cpu().detach().numpy()
            next_state_inner, reward_inner, self.done, info = self.test_env.step(env_action_real_npy, inner_loop=self.steps_inloop)
            if self.done is True:
                break
            reward += reward_inner

            # predict disturbance
            state_d = (next_state_inner - state_inner) / self.frameskip

            self.disturbance_pred = denormalize(self.inv_model_net(state_d, next_state_inner) - action_tensor,
                                           self.max_action, self.min_action)

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