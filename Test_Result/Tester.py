import argparse
import torch
from Algorithm import *

from Trainer import *
from Example import *
from Common.Utils import set_seed, gym_env

def hyperparameters():
    parser = argparse.ArgumentParser(description='Tester of algorithms')

    # related to development
    parser.add_argument('--develop-mode', default=False, type=bool, help="you should choose whether basic or model_based")
    parser.add_argument('--frameskip_inner', default=1, type=int, help='frame skip in inner loop ')

    parser.add_argument('--path', default="X:/env_mbrl/Results/saved_net/", help='path for save')
    parser.add_argument('--modelnet-name', default="modelDNN_better", help='modelDNN_better, modelBNN_better')
    parser.add_argument('--policynet-name', default="policy_best", help='best, better, current, total')

    # environment
    parser.add_argument('--algorithm', default='SAC_v2', type=str, help='you should choose same algorithm with loaded network')
    parser.add_argument('--domain-type', default='gym', type=str, help='gym or dmc, dmc/image')
    parser.add_argument('--env-name', default='Thrower-v2', help='Pendulum-v0, MountainCarContinuous-v0')
    parser.add_argument('--render', default=True, type=bool)
    parser.add_argument('--test-episode', default=10, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')

    args = parser.parse_args()

    return args

def main(args_tester):

    args = None

    if args_tester.algorithm == 'SAC_v2':
        from Example.run_SACv2 import hyperparameters
    if args_tester.algorithm == 'DDPG':
        from Example.run_SACv2 import hyperparameters

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
    env, test_env = gym_env(args.env_name, random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    algorithm = None

    if args_tester.algorithm == 'SAC_v2':
        algorithm = SAC_v2(state_dim, action_dim, device, args)

    elif args_tester.algorithm == 'DDPG':
        algorithm = DDPG(state_dim, action_dim, device, args)

    algorithm.actor.load_state_dict(torch.load(args_tester.path + "policy/" + args_tester.policynet_name))

    print("Training of", args.domain_type + '_' + args.env_name)
    print("Algorithm:", algorithm.name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Action range: {:.2f} ~ {:.2f}".format(min(min_action), max(max_action)))
    print("step time: {} (frame skip: {})".format(env.env.dt, env.env.frame_skip))
    if env.env.frame_skip <= args_tester.frameskip_inner:
        raise Exception(" please check your frameskip_inner ")

    trainer = None
    if args_tester.develop_mode is False:
        trainer = Basic_trainer(
            env, test_env, algorithm, max_action, min_action, args, args_tester)
    else:
        trainer = Model_trainer(
            env, test_env, algorithm, state_dim, action_dim, max_action, min_action, args, args_tester)
    trainer.test()

if __name__ == '__main__':
    args_tester = hyperparameters()
    main(args_tester)