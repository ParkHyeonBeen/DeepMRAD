import math, random, time

import cv2
from Common.Utils import *

class Basic_trainer():
    def __init__(self, env, test_env, algorithm, max_action, min_action, args, args_tester=None):

        self.args = args
        self.args_tester = args_tester

        self.domain_type = self.args.domain_type
        self.env_name = self.args.env_name
        self.env = env
        self.test_env = test_env

        self.algorithm = algorithm

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

        if self.args_tester is None:
            self.render = self.args.render
            self.path = self.args.path
        else:
            self.render = self.args_tester.render
            self.path = self.args_tester.path
            self.test_episode = self.args_tester.test_episode

        # score
        self.score = 0
        self.robust_score = 0
        self.total_score = 0
        self.best_score = 0

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
        self.eval_num += 1
        episode = 0
        reward_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
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

            reward_list.append(eval_reward)
        score_now = sum(reward_list) / len(reward_list)
        alive_rate = alive_cnt / self.eval_episode

        if score_now > self.score:
            torch.save(self.algorithm.actor.state_dict(),
                       self.path + "policy_better")
            self.score = score_now
        if alive_rate > 0.9:
            torch.save(self.algorithm.actor.state_dict(),
                       self.path + "policy_current")
        if alive_cnt != 0 and score_now*alive_rate > self.total_score:
            torch.save(self.algorithm.actor.state_dict(),
                       self.path + "policy_total")
            self.total_score = score_now*alive_rate
        if alive_rate >= 0.8 and score_now*alive_rate > self.best_score:
            torch.save(self.algorithm.actor.state_dict(),
                       self.path + "policy_best")
            self.best_score = score_now*alive_rate

        print("Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}".format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list), np.std(reward_list), 100*alive_rate))
        self.test_env.close()

    def run(self):
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

                if self.eval == True and self.total_step % self.eval_step == 0:
                    self.evaluate()

            # print("alpha : ", self.algorithm.log_alpha.exp())
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
            observation = self.test_env.reset()

            if '-ram-' in self.env_name:  # Atari Ram state
                observation = observation / 255.

            done = False

            while not done:
                self.local_step += 1
                action = self.algorithm.eval_action(observation)
                env_action = denormalize(action, self.max_action, self.min_action)
                if self.args_tester.add_noise is True:
                    env_action = add_noise(env_action, scale=self.args_tester.noise_scale)
                if self.args_tester.add_disturbance is True:
                    env_action = add_disturbance(env_action, self.local_step,
                                                     self.env.spec.max_episode_steps,
                                                     scale=self.args_tester.disturbance_scale,
                                                     frequency=self.args_tester.disturbance_frequency)
                next_observation, reward, done, _ = self.test_env.step(env_action)

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
                    # print(episode, "th pos :", observation[0:7])
                    alive_cnt += 1

            reward_list.append(eval_reward)

        print("Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}".format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list), np.std(reward_list), 100*(alive_cnt/self.eval_episode)))
        self.test_env.close()


