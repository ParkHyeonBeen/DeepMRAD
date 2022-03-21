import argparse, sys, os
import torch
from pathlib import Path

sys.path.append(str(Path('Tester.py').parent.absolute()))   # 절대 경로에 추가

from Algorithm import *
from Trainer import *
from Example import *

from Common.Utils import set_seed, gym_env

def hyperparameters():
    parser = argparse.ArgumentParser(description='Tester of algorithms')

    # related to development
    parser.add_argument('--test-on', default=False, type=bool, help="You must turn on when you test")
    parser.add_argument('--develop-mode', default='DeepDOB', help="Basic, DeepDOB, MRAP")
    parser.add_argument('--frameskip_inner', default=1, type=int, help='frame skip in inner loop ')

    parser.add_argument('--path', default="X:/env_mbrl/Results/", help='path for save')
    parser.add_argument('--result-index', default="Result/", help='result to check')
    parser.add_argument('--prev-result', default=True, type=bool, help='if previous result, True')
    parser.add_argument('--prev-result-fname', default="0308_Walker2d-v3", help='choose the result to view')
    parser.add_argument('--modelnet-name', default="modelDNN_better", help='modelDNN_better, modelBNN_better')
    parser.add_argument('--policynet-name', default="policy_best", help='best, better, current, total')

    # setting real world
    parser.add_argument('--add_noise', default=False, type=bool, help="if True, add noise to action")
    parser.add_argument('--noise_scale', default=0.1, type=float, help='white noise having the noise scale')

    parser.add_argument('--add_disturbance', default=True, type=bool, help="if True, add disturbance to action")
    parser.add_argument('--disturbance_scale', default=0.5, type=float, help='choose disturbance scale')
    parser.add_argument('--disturbance_frequency', default=[2, 4, 8], type=list, help='choose disturbance frequency')

    # environment
    parser.add_argument('--env-name', default='Humanoid-v3', help='refer to gym.envs.__init__.py')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--test-episode', default=10, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--algorithm', default='SAC_v2', type=str, help='you should choose same algorithm with loaded network')
    parser.add_argument('--domain-type', default='gym', type=str, help='gym or dmc, dmc/image')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')

    args = parser.parse_args()

    return args

def main(args_tester):

    if args_tester.test_on is False:
        raise Exception(" You must turn on args_tester.test_on if you wanna test ")

    args = None

    if args_tester.algorithm == 'SAC_v2':
        from Example.run_SACv2 import hyperparameters
    if args_tester.algorithm == 'DDPG':
        from Example.run_DDPG import hyperparameters

    args = hyperparameters()

    if args_tester.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device: ", device)
    # random seed setting
    random_seed = set_seed(args.random_seed)
    print("Random Seed:", random_seed)

    #env setting
    if args_tester.prev_result is True:
        env_name = args_tester.prev_result_fname[5:]
    else:
        env_name = args.env_name

    print(env_name)
    env, test_env = gym_env(env_name, random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    algorithm = None

    if args_tester.algorithm == 'SAC_v2':
        algorithm = SAC_v2(state_dim, action_dim, device, args)

    elif args_tester.algorithm == 'DDPG':
        algorithm = DDPG(state_dim, action_dim, device, args)

    if args_tester.prev_result is True:
        path_policy = args_tester.path + "storage/" + args_tester.prev_result_fname + "/saved_net/policy/" + args_tester.policynet_name
    else:
        path_policy = args_tester.path + args_tester.result_index + "saved_net/policy/" + args_tester.policynet_name

    algorithm.actor.load_state_dict(torch.load(path_policy))

    print("Training of", args.domain_type + '_' + args.env_name)
    print("Test about", args_tester.develop_mode)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Action range: {:.2f} ~ {:.2f}".format(min(min_action)/10, max(max_action)/10))
    print("step time: {} (frame skip: {})".format(env.env.dt, env.env.frame_skip))
    # if env.env.frame_skip <= args_tester.frameskip_inner:
    #     raise Exception(" please check your frameskip_inner ")

    trainer = None
    if args_tester.develop_mode == 'Basic':
        trainer = Basic_trainer(
            env, test_env, algorithm, max_action, min_action, args, args_tester)
    else:
        trainer = Model_trainer(
            env, test_env, algorithm, state_dim, action_dim, max_action, min_action, args, args_tester)
    trainer.test()

if __name__ == '__main__':
    args_tester = hyperparameters()
    main(args_tester)