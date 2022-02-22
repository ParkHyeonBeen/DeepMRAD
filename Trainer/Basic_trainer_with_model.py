import math, random, time
import numpy as np

import cv2
import torch

from Common.Utils import *
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
        self.render = self.args.render
        self.max_step = self.args.max_step

        self.eval = self.args.eval
        self.eval_episode = self.args.eval_episode
        self.eval_step = self.args.eval_step

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0

        self.train_mode = None

        self.path = self.args.path

        # score
        self.score = 0
        self.total_score = 0
        self.best_score = 0

        self.cost = np.array([1e1, 1e1, 1e1, 1e1])

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

        self.model_net_DNN = DynamicsNetwork(self.state_dim, self.action_dim, self.algorithm, self.args, net_type="DNN")
        self.model_net_BNN = DynamicsNetwork(self.state_dim, self.action_dim, self.algorithm, self.args, net_type="BNN")

        self.inv_model_net_DNN = InverseDynamicsNetwork(self.state_dim, self.action_dim, self.algorithm, self.args, net_type="DNN")
        self.inv_model_net_BNN = InverseDynamicsNetwork(self.state_dim, self.action_dim, self.algorithm, self.args, net_type="BNN")

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
        save_data(self.path, "saved_csv/Eval_by" + str(self.total_step // self.eval_step))
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

                if self.discrete == False:
                    env_action = self.max_action * np.clip(action, -1, 1)
                else:
                    env_action = action

                next_observation, reward, done, _ = self.test_env.step(env_action)
                cost_DNN = self.model_net_DNN.eval_model(observation, env_action, next_observation)
                cost_BNN = self.model_net_BNN.eval_model(observation, env_action, next_observation)
                cost_invDNN = self.inv_model_net_DNN.eval_model(observation, env_action, next_observation)
                cost_invBNN = self.inv_model_net_BNN.eval_model(observation, env_action, next_observation)

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

        if score_now > self.score:
            sava_network(self.algorithm.actor, "policy_better", self.path)
            self.score = score_now
        if alive_rate > 0.9:
            sava_network(self.algorithm.actor, "policy_current", self.path)
        if alive_cnt != 0 and score_now*alive_rate > self.total_score:
            sava_network(self.algorithm.actor, "policy_total", self.path)
            self.total_score = score_now*alive_rate
        if alive_rate >= 0.8 and score_now*alive_rate > self.best_score:
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

        save_data(self.path, "saved_csv/Eval_" + str(self.total_step // self.eval_step))
        init_data()

        print("Eval  | Average Reward: {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}"
              .format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list), np.std(reward_list), 100*alive_rate))
        # print("Cost | DNN: ", eval_cost[0], " BNN: ", eval_cost[1]," invDNN: ", eval_cost[2], " invBNN: ", eval_cost[3])
        print("Cost  | DNN: {:.2f}, BNN: {:.2f}, invDNN: {:.2f}, inv.BNN: {:.2f} "
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
                   action = self.env.action_space.sample()
                   next_observation, reward, done, _ = self.env.step(action)
                else:
                    if self.algorithm.buffer.on_policy == False:
                        action = self.algorithm.get_action(observation)
                    else:
                        action, log_prob = self.algorithm.get_action(observation)

                    if self.discrete == False:
                        env_action = self.max_action * np.clip(action, -1, 1)
                    else:
                        env_action = action

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

                if self.eval == True and self.total_step % self.eval_step == 0:
                    self.evaluate()
                    df = pd.DataFrame(reward_list)
                    df.to_csv(self.path + "saved_csv/reward" + ".csv")

            reward_list.append(self.episode_reward)
            print("Train | Episode: {}, Reward: {:.2f}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step))
        self.env.close()

    def test(self):
        predicted_model = None
        predicted_inv_model = None

        if "DNN" in self.args_tester.modelnet_name:
            predicted_model = DynamicsNetwork(
                self.state_dim, self.action_dim, self.algorithm, self.args, net_type="DNN").cuda()
            predicted_model.load_state_dict(
                torch.load(self.args_tester.path + "model/DNN/" + self.args_tester.modelnet_name))

            predicted_inv_model = InverseDynamicsNetwork(
                self.state_dim, self.action_dim, self.algorithm, self.args, net_type="DNN").cuda()
            predicted_inv_model.load_state_dict(
                torch.load(self.args_tester.path + "model/DNN/inv" + self.args_tester.modelnet_name))

        if "BNN" in self.args_tester.modelnet_name:
            predicted_model = DynamicsNetwork(
                self.state_dim, self.action_dim, self.algorithm, self.args, net_type="BNN").cuda()
            predicted_model.load_state_dict(
                torch.load(self.args_tester.path + "model/BNN/" + self.args_tester.modelnet_name))

            predicted_inv_model = InverseDynamicsNetwork(
                self.state_dim, self.action_dim, self.algorithm, self.args, net_type="BNN").cuda()
            predicted_inv_model.load_state_dict(
                torch.load(self.args_tester.path + "model/BNN/inv" + self.args_tester.modelnet_name))

        self.eval_num += 1
        episode = 0
        error_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
            if episode >= self.eval_episode:
                break
            episode += 1
            eval_error = 0
            observation = self.test_env.reset()

            if '-ram-' in self.env_name:  # Atari Ram state
                observation = observation / 255.

            done = False
            disturbance_pred = 0
            while not done:
                self.local_step += 1
                action_NN = self.algorithm.eval_action(observation)

                if self.discrete == False:
                    action_NN = self.max_action * np.clip(action_NN, -1, 1)

                next_state_pred = predicted_model(observation, action_NN)

                for i in range(self.test_env.env.frame_skip//self.args_tester.inner_skip):
                    action_noise = add_noise(action_NN)
                    action_dob = action_noise - disturbance_pred
                    self.test_env.env.do_simulation(action_dob, self.args_tester.inner_skip)
                    ob_inner = self.test_env.env._get_obs()
                    ob_inner = np.append(ob_inner, [linalg, self.error_ang])
                    disturbance_pred = predicted_inv_model(observation, ob_inner) - action_NN

                state_error = ob_inner - next_state_pred
                OutlayerOptimizer(self.algorithm.actor, state_error)
                predicted_model.adaptive_train(state_error)
                predicted_inv_model.adaptive_train(state_error)

                if self.render == True:
                    if self.domain_type in {'gym', "atari"}:
                        self.test_env.render()
                    elif self.domain_type in {'procgen'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array'))
                        cv2.waitKey(1)
                    elif self.domain_type in {'dmc', 'dmcr'}:
                        cv2.imshow("{}_{}_{}".format(self.algorithm.name, self.domain_type, self.env_name), self.env.render(mode='rgb_array', height=240, width=320))
                        cv2.waitKey(1)

                eval_error += abs(state_error)
                observation = ob_inner

                if self.local_step == self.env.spec.max_episode_steps:
                    print(episode, "th pos :", observation[0:7])
                    alive_cnt += 1
            error_list.append(eval_error/self.local_step)

        save_path(self.args.path, "path_normal")
        print("Eval  | Average error {:.2f}, Max error: {:.2f}, Min error: {:.2f}, Stddev error: {:.2f}, alive rate : {:.2f}".format(sum(error_list)/len(error_list), max(error_list), min(error_list), np.std(error_list), 100*(alive_cnt/self.eval_episode)))
        self.test_env.close()


