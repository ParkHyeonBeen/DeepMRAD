import cv2

from Common.Utils import *
from Common.Utils_model import *
from Common.DeepDOB import DeepDOB
from Common.MRAP import MRAP
from Common.Ensemble_model import Ensemble

from Network.Model_Network import *

class Model_trainer():
    def __init__(self, env, test_env, algorithm,
                 state_dim, action_dim,
                 max_action, min_action,
                 args, args_tester=None,
                 ensemble_mode=False):

        self.args = args
        self.args_tester = args_tester

        self.domain_type = self.args.domain_type
        self.env_name = self.args.env_name
        self.env = env
        self.test_env = test_env

        self.algorithm = algorithm

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.max_action = max_action
        self.min_action = min_action

        self.discrete = self.args.discrete
        self.max_step = self.args.max_step

        self.eval = self.args.eval
        self.eval_episode = self.args.eval_episode
        self.eval_step = self.args.eval_step

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0
        self.test_num = 0

        # score
        self.score = None
        self.cost = None

        self.train_mode = None
        if args.train_mode == 'offline':
            self.train_mode = self.offline_train
        elif args.train_mode == 'online':
            self.train_mode = self.online_train
        elif args.train_mode == 'batch':
            self.train_mode = self.batch_train

        assert self.train_mode is not None

        self.log = self.args.log

        self.model = self.args.model
        self.model_freq = self.args.model_freq
        self.buffer = self.args.buffer
        self.buffer_freq = self.args.buffer_freq

        self.steptime = self.env.env.dt
        self.steptime_xml = (self.steptime / self.env.env.frame_skip)
        self.frameskip = self.test_env.env.frame_skip
        self.frameskip_origin = self.test_env.env.frame_skip_origin

        if self.args_tester is None:
            self.render = self.args.render
            self.path = self.args.path
            self.ensemble_mode = args.ensemble_mode
        else:
            self.frameskip_inner = self.args_tester.frameskip_inner
            self.steps_inloop = self.test_env.env.frame_skip_origin//self.frameskip
            self.render = self.args_tester.render
            self.path = self.args_tester.path
            self.test_episode = self.args_tester.test_episode
            self.ensemble_mode = ensemble_mode

        self.deepdob = None
        self.mrap = None
        self.eval_data = DataManager()

        # For Testing
        if self.args_tester is not None:
            if "DNN" in args_tester.modelnet_name:
                self.model_net, self.inv_model_net = \
                    create_models(self.state_dim, self.action_dim, self.frameskip, self.algorithm, self.args, bnn=False,
                                  ensemble_mode=self.ensemble_mode)
            elif "BNN" in args_tester.modelnet_name:
                self.model_net, self.inv_model_net = \
                    create_models(self.state_dim, self.action_dim, self.frameskip, self.algorithm, self.args, dnn=False,
                                  ensemble_mode=self.ensemble_mode)
            load_models(args_tester, self.model_net, self.inv_model_net, ensemble_mode=self.ensemble_mode)

            if self.args_tester.develop_mode == 'DeepDOB':
                self.deepdob = DeepDOB(self.inv_model_net, self.test_env, self.steps_inloop, self.args_tester,
                                       self.frameskip_inner, self.action_dim, self.max_action, self.min_action)
            if self.args_tester.develop_mode == 'MRAP':
                self.mrap = MRAP(self.algorithm.actor, self.model_net, self.test_env,
                                 self.args_tester, self.frameskip_origin)
        # For training
        else:
            self.model_net_DNN, self.model_net_BNN, self.inv_model_net_DNN, self.inv_model_net_BNN = \
                create_models(self.state_dim, self.action_dim, self.frameskip, self.algorithm, self.args,
                              ensemble_mode=self.ensemble_mode)

    def offline_train(self, d, local_step):
        if d:
            return True
        return False

    def online_train(self, d, local_step):
        return True

    def batch_train(self, d, local_step):#VPG, TRPO, PPO only
        if d or local_step == self.algorithm.batch_size:
            return True
        return False

    def evaluate(self):
        self.eval_data.save_data(self.path, "saved_log/Eval_by" + str(self.total_step // self.eval_step))
        self.eval_data.init_data()

        self.eval_num += 1
        episode = 0
        reward_list = []
        alive_cnt = 0

        eval_cost = np.zeros(4)

        while True:
            self.local_step = 0
            eval_cost_temp = np.zeros(4)
            if episode >= self.eval_episode:
                break
            episode += 1
            eval_reward = 0
            observation = self.test_env.reset()

            if '-ram-' in self.env_name:  # Atari Ram state
                observation = observation / 255.

            done = False

            while not done:
                self.local_step += 1
                action = self.algorithm.eval_action(observation)
                env_action = denormalize(action, self.max_action, self.min_action)

                next_observation, reward, done, _ = self.test_env.step(env_action)

                _, mse_list, _ = eval_models(observation, action, next_observation,
                                              self.model_net_DNN, self.model_net_BNN,
                                              self.inv_model_net_DNN, self.inv_model_net_BNN)
                if episode == self.eval_episode:
                    self.eval_data.put_data(mse_list)
                    eval_cost_temp += mse_list

                if self.render == True:
                    if self.domain_type in {'gym', "atari"}:
                        self.test_env.render()
                    elif self.domain_type in {'procgen'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array'))
                        cv2.waitKey(1)
                    elif self.domain_type in {'dmc', 'dmcr'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                eval_reward += reward
                observation = next_observation

                if self.local_step == self.env.spec.max_episode_steps:
                    alive_cnt += 1

            eval_cost = eval_cost_temp / self.local_step
            reward_list.append(eval_reward)

        score_now = sum(reward_list) / len(reward_list)
        alive_rate = alive_cnt / self.eval_episode

        if self.eval_num == 1:
            self.score = score_now
            self.cost = eval_cost

        _score = save_policys(self.algorithm, self.score, score_now, alive_rate, self.path)

        if _score is not None:
            self.score = _score

        cost_with_index = save_models(self.args, self.cost, eval_cost, self.path,
                                  self.model_net_DNN, self.model_net_BNN,
                                  self.inv_model_net_DNN, self.inv_model_net_BNN)
        if cost_with_index is not None:
            self.cost[cost_with_index[0]] = cost_with_index[1]

        self.eval_data.save_data(self.path, "saved_log/Eval_" + str(self.total_step // self.eval_step))
        self.eval_data.init_data()

        print("Eval  | Average Reward: {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}"
              .format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list), np.std(reward_list), 100*alive_rate))
        print("Cost  | DNN: {:.7f}, BNN: {:.7f}, invDNN: {:.7f}, invBNN: {:.7f} "
              .format(eval_cost[0], eval_cost[1], eval_cost[2], eval_cost[3]))
        self.test_env.close()

    def run(self):
        reward_list = []
        while True:
            if self.total_step > self.max_step:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0

            observation = self.env.reset()
            done = False

            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.render == True:
                    if self.domain_type in {'gym', "atari"}:
                        self.env.render()
                    elif self.domain_type in {'procgen'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array'))
                        cv2.waitKey(1)
                    elif self.domain_type in {'dmc', 'dmcr'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                if '-ram-' in self.env_name:  # Atari Ram state
                    observation = observation / 255.

                if self.total_step <= self.algorithm.training_start:
                    env_action = self.env.action_space.sample()
                    action = normalize(env_action, self.max_action, self.min_action)
                else:
                    if self.algorithm.buffer.on_policy == False:
                        action = self.algorithm.get_action(observation)
                    else:
                        action, log_prob = self.algorithm.get_action(observation)
                    env_action = denormalize(action, self.max_action, self.min_action)

                next_observation, reward, done, info = self.env.step(env_action)

                if self.local_step + 1 == 1000:
                    real_done = 0.
                else:
                    real_done = float(done)

                self.episode_reward += reward

                if self.env_name == 'Pendulum':
                    reward = (reward + 8.1368) / 8.1368

                if self.algorithm.buffer.on_policy == False:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done)
                else:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done, log_prob)

                observation = next_observation

                if self.total_step >= self.algorithm.training_start and self.train_mode(done, self.local_step):
                    loss_list = self.algorithm.train(self.algorithm.training_step)
                    _, mse_list, kl_list = train_alls(self.args.training_step,
                                                              self.model_net_DNN, self.model_net_BNN,
                                                              self.inv_model_net_DNN, self.inv_model_net_BNN)
                    saveData = np.hstack((mse_list, kl_list))
                    self.eval_data.put_data(saveData)

                if self.eval is True and self.total_step % self.eval_step == 0:
                    self.evaluate()
                    if self.args.numpy is False:
                        df = pd.DataFrame(reward_list)
                        df.to_csv(self.path + "saved_log/reward" + ".csv")
                    else:
                        df = np.array(reward_list)
                        np.save(self.path + "saved_log/reward" + ".npy", df)

            reward_list.append(self.episode_reward)
            print("Train | Episode: {}, Reward: {:.2f}, Local_step: {}, Total_step: {}".format(
                self.episode, self.episode_reward, self.local_step, self.total_step))
        self.env.close()

    def test(self):
        self.test_num += 1
        episode = 0
        reward_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
            alive = False
            if episode >= self.test_episode:
                break
            episode += 1
            eval_reward = 0
            loss = 0

            if self.args_tester.develop_mode == 'DeepDOB':
                observation = self.deepdob.reset()
            else:
                observation = self.test_env.reset()

            if '-ram-' in self.env_name:  # Atari Ram state
                observation = observation / 255.

            done = False

            while not done:
                self.local_step += 1

                if self.args_tester.develop_mode == 'MRAP':
                    action = self.mrap.eval_action(observation)
                    env_action = denormalize(action, self.max_action, self.min_action).cpu().detach().numpy()
                else:
                    action = self.algorithm.eval_action(observation)   # policy update 하려면, eval_action 내부의 .cpu().numpy() 제거
                    env_action = denormalize(action, self.max_action, self.min_action)

                if self.args_tester.develop_mode == 'DeepDOB':
                    next_observation, reward, done, _ = self.deepdob.step(env_action, self.local_step)
                else:
                    if self.args_tester.add_noise is True and self.args_tester.noise_to == 'action':
                        env_action = add_noise(env_action, scale=self.args_tester.noise_scale)
                    if self.args_tester.add_disturbance is True and self.args_tester.disturbance_to == 'action':
                        env_action = add_disturbance(env_action, self.local_step,
                                                     self.env.spec.max_episode_steps,
                                                     scale=self.args_tester.disturbance_scale,
                                                     frequency=self.args_tester.disturbance_frequency)
                    next_observation, reward, done, _ = self.test_env.step(env_action)

                    if self.args_tester.add_noise is True and self.args_tester.disturbance_to == 'state':
                        next_observation = add_noise(next_observation, scale=self.args_tester.noise_scale)
                    if self.args_tester.add_disturbance is True and self.args_tester.noise_to == 'state':
                        next_observation = add_disturbance(next_observation, self.local_step,
                                                           self.test_env.spec.max_episode_steps,
                                                           scale=self.args_tester.disturbance_scale,
                                                           frequency=self.args_tester.disturbance_frequency)

                if self.args_tester.develop_mode == 'MRAP':
                    loss += self.mrap.eval_error(observation, action, next_observation)

                if self.render == True:
                    if self.domain_type in {'gym', "atari"}:
                        self.test_env.render()
                    elif self.domain_type in {'procgen'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array'))
                        cv2.waitKey(1)
                    elif self.domain_type in {'dmc', 'dmcr'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                eval_reward += reward
                observation = next_observation

                if self.local_step == self.env.spec.max_episode_steps:
                    alive_cnt += 1
                    alive = True

            # print("Eval of {}th episode  | Episode Reward {:.2f}, alive : {}".format(episode, eval_reward, alive))
            reward_list.append(eval_reward)

        print(
            "Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}".format(
                sum(reward_list) / len(reward_list), max(reward_list), min(reward_list), np.std(reward_list),
                100 * (alive_cnt / self.test_episode)))
        self.test_env.close()
        return sum(reward_list) / len(reward_list),\
               max(reward_list), min(reward_list),\
               np.std(reward_list),\
               100 * (alive_cnt / self.test_episode)


