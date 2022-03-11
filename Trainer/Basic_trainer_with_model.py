import math, random, time
import numpy as np

import cv2
import torch

from Common.Utils import *
from Common.DeepDOB import DeepDOB
from Common.MRAP import MRAP

from Network.Model_Network import *

class Model_trainer():
    def __init__(self, env, test_env, algorithm,
                 state_dim, action_dim,
                 max_action, min_action,
                 args, args_tester=None):

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
        self.total_score = None
        self.best_score = None
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
        self.frameskip_inner = self.args_tester.frameskip_inner
        self.frameskip_origin = self.test_env.env.frame_skip_origin

        if self.args_tester is None:
            self.render = self.args.render
            self.path = self.args.path

        else:
            self.steps_inloop = self.test_env.env.frame_skip_origin//self.frameskip
            self.render = self.args_tester.render
            self.path = self.args_tester.path
            self.test_episode = self.args_tester.test_episode

        self.deepdob = None
        self.mrap = None

        self.model_net_DNN = DynamicsNetwork(self.state_dim, self.action_dim, self.frameskip, self.algorithm, self.args, net_type="DNN")
        self.model_net_BNN = DynamicsNetwork(self.state_dim, self.action_dim, self.frameskip, self.algorithm, self.args, net_type="BNN")

        self.inv_model_net_DNN = InverseDynamicsNetwork(self.state_dim, self.action_dim,
                                                        self.frameskip, self.algorithm,
                                                        self.args, net_type="DNN")
        self.inv_model_net_BNN = InverseDynamicsNetwork(self.state_dim, self.action_dim,
                                                        self.frameskip, self.algorithm,
                                                        self.args, net_type="BNN")

        if self.args_tester is not None:
            if "DNN" in self.args_tester.modelnet_name:
                if self.args_tester.prev_result is True:
                    path_model = self.path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/DNN/" + self.args_tester.modelnet_name
                    path_invmodel = self.path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/DNN/inv" + self.args_tester.modelnet_name
                else:
                    path_model = self.path + self.args_tester.result_index + "saved_net/model/DNN/" + self.args_tester.modelnet_name
                    path_invmodel = self.path + self.args_tester.result_index + "saved_net/model/DNN/inv" + self.args_tester.modelnet_name

                self.model_net_DNN.load_state_dict(torch.load(path_model))
                self.inv_model_net_DNN.load_state_dict(torch.load(path_invmodel))
                self.deepdob = DeepDOB(self.inv_model_net_DNN, self.test_env, self.steps_inloop, self.args_tester,
                                       self.frameskip_inner, self.action_dim, self.max_action, self.min_action)
                self.mrap = MRAP(self.algorithm.actor, self.model_net_DNN, self.test_env, self.args_tester,
                                 self.frameskip_origin)

            if "BNN" in self.args_tester.modelnet_name:
                if self.args_tester.prev_result is True:
                    path_model = self.path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/BNN/" + self.args_tester.modelnet_name
                    path_invmodel = self.path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/BNN/inv" + self.args_tester.modelnet_name
                else:
                    path_model = self.path + "saved_net/model/BNN/" + self.args_tester.modelnet_name
                    path_invmodel = self.path + "saved_net/model/BNN/inv" + self.args_tester.modelnet_name

                self.model_net_BNN.load_state_dict(torch.load(path_model))
                self.inv_model_net_BNN.load_state_dict(torch.load(path_invmodel))
                self.deepdob = DeepDOB(self.inv_model_net_BNN, self.test_env, self.steps_inloop, self.args_tester,
                                       self.frameskip_inner, self.action_dim, self.max_action, self.min_action)
                self.mrap = MRAP(self.algorithm.actor, self.model_net_BNN, self.test_env, self.args_tester,
                                 self.frameskip_origin)


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
        save_data(self.path, "saved_log/Eval_by" + str(self.total_step // self.eval_step))
        init_data()

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
                cost_DNN = self.model_net_DNN.eval_model(observation, action, next_observation)
                cost_BNN = self.model_net_BNN.eval_model(observation, action, next_observation)
                cost_invDNN = self.inv_model_net_DNN.eval_model(observation, action, next_observation)
                cost_invBNN = self.inv_model_net_BNN.eval_model(observation, action, next_observation)

                if episode == 1:
                    saveData = np.hstack((cost_DNN, cost_BNN, cost_invDNN, cost_invBNN))
                    put_data(saveData)

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
                eval_cost_temp[0] += cost_DNN
                eval_cost_temp[1] += cost_BNN
                eval_cost_temp[2] += cost_invDNN
                eval_cost_temp[3] += cost_invBNN
                observation = next_observation

                if self.local_step == self.env.spec.max_episode_steps:
                    alive_cnt += 1

            eval_cost += eval_cost_temp/self.local_step
            reward_list.append(eval_reward)

        score_now = sum(reward_list) / len(reward_list)
        alive_rate = alive_cnt / self.eval_episode
        eval_cost = eval_cost/self.eval_episode

        if self.eval_num == 1:
            self.score = score_now
            self.total_score = score_now*alive_rate
            self.best_score = score_now*alive_rate
            self.cost = eval_cost

        if score_now > self.score:
            sava_network(self.algorithm.actor, "policy_better", self.path)
            self.score = score_now
        if alive_rate > 0.9:
            sava_network(self.algorithm.actor, "policy_current", self.path)
        if alive_cnt != 0 and score_now*alive_rate > self.total_score:
            sava_network(self.algorithm.actor, "policy_total", self.path)
            self.total_score = score_now*alive_rate
        if alive_rate >= 0.9 and score_now*alive_rate > self.best_score:
            sava_network(self.algorithm.actor, "policy_best", self.path)
            self.best_score = score_now*alive_rate

        if self.cost[0] > eval_cost[0]:
            sava_network(self.model_net_DNN, "modelDNN_better", self.path)
            self.cost[0] = eval_cost[0]
        if self.cost[1] > eval_cost[1]:
            sava_network(self.model_net_BNN, "modelBNN_better", self.path)
            self.cost[1] = eval_cost[1]
        if self.cost[2] > eval_cost[2]:
            sava_network(self.inv_model_net_DNN, "invmodelDNN_better", self.path)
            self.cost[2] = eval_cost[2]
        if self.cost[3] > eval_cost[3]:
            sava_network(self.inv_model_net_BNN, "invmodelBNN_better", self.path)
            self.cost[3] = eval_cost[3]

        save_data(self.path, "saved_log/Eval_" + str(self.total_step // self.eval_step))
        init_data()

        print("Eval  | Average Reward: {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}"
              .format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list), np.std(reward_list), 100*alive_rate))
        # print("Cost | DNN: ", eval_cost[0], " BNN: ", eval_cost[1]," invDNN: ", eval_cost[2], " invBNN: ", eval_cost[3])
        print("Cost  | DNN: {:.7f}, BNN: {:.7f}, invDNN: {:.7f}, inv.BNN: {:.7f} "
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
                    loss_mse_DNN, loss_kl_DNN = self.model_net_DNN.train_all(self.algorithm.training_step)
                    loss_mse_BNN, loss_kl_BNN = self.model_net_BNN.train_all(self.algorithm.training_step)

                    loss_mse_invDNN, loss_kl_invDNN = self.inv_model_net_DNN.train_all(self.algorithm.training_step)
                    loss_mse_invBNN, loss_kl_invBNN = self.inv_model_net_BNN.train_all(self.algorithm.training_step)

                    saveData = np.hstack((loss_mse_DNN, loss_kl_DNN,
                                            loss_mse_BNN, loss_kl_BNN,
                                            loss_mse_invDNN, loss_kl_invDNN,
                                            loss_mse_invBNN, loss_kl_invBNN))
                    put_data(saveData)

                if self.eval is True and self.total_step % self.eval_step == 0:
                    self.evaluate()
                    if self.args.numpy is False:
                        df = pd.DataFrame(reward_list)
                        df.to_csv(self.path + "saved_log/reward" + ".csv")
                    else:
                        df = np.array(reward_list)
                        np.save(self.path + "saved_log/reward" + ".npy", df)

            reward_list.append(self.episode_reward)
            print("Train | Episode: {}, Reward: {:.2f}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step))
        self.env.close()

    def test(self):
        self.test_num += 1
        episode = 0
        reward_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
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
                    env_action = denormalize(action, self.max_action, self.min_action, istest=True).cpu().detach().numpy()
                else:
                    action = self.algorithm.eval_action(observation)   # policy update 하려면, eval_action 내부의 .cpu().numpy() 제거
                    env_action = denormalize(action, self.max_action, self.min_action, istest=True)

                if self.args_tester.develop_mode == 'DeepDOB':
                    next_observation, reward, done, _ = self.deepdob.step(env_action, self.local_step)
                else:
                    if self.args_tester.add_noise is True:
                        env_action = add_noise(env_action, scale=self.args_tester.noise_scale)
                    if self.args_tester.add_disturbance is True:
                        env_action = add_disturbance(env_action, self.local_step,
                                                     self.env.spec.max_episode_steps,
                                                     scale=self.args_tester.disturbance_scale,
                                                     frequency=self.args_tester.disturbance_frequency)
                    next_observation, reward, done, _ = self.test_env.step(env_action)

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
                    # print(episode, "th pos :", state[0:7])
                    alive_cnt += 1

            reward_list.append(eval_reward)
            print("loss:", loss/self.local_step)

        print(
            "Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}".format(
                sum(reward_list) / len(reward_list), max(reward_list), min(reward_list), np.std(reward_list),
                100 * (alive_cnt / self.test_episode)))
        self.test_env.close()


