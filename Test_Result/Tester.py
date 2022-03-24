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
    parser.add_argument('--test-on', default=True, type=bool, help="You must turn on when you test")
    parser.add_argument('--develop-mode', default='MRAP', help="Basic, DeepDOB, MRAP")
    parser.add_argument('--frameskip_inner', default=1, type=int, help='frame skip in inner loop ')

    # environment
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--test-episode', default=10, type=int, help='Number of episodes to perform evaluation')

    # result to watch
    parser.add_argument('--path', default="X:/env_mbrl/Results/", help='path for save')
    parser.add_argument('--result-index', default="Result/", help='result to check')
    parser.add_argument('--prev-result', default=False, type=bool, help='if previous result, True')
    parser.add_argument('--prev-result-fname', default="0308_Walker2d-v3", help='choose the result to view')
    parser.add_argument('--modelnet-name', default="modelBNN_better", help='modelDNN_better, modelBNN_better')
    parser.add_argument('--policynet-name', default="policy_best", help='best, better, current, total')

    # setting real world
    parser.add_argument('--add_noise', default=False, type=bool, help="if True, add noise to action")
    parser.add_argument('--noise_scale', default=0.1, type=float, help='white noise having the noise scale')

    parser.add_argument('--add_disturbance', default=True, type=bool, help="if True, add disturbance to action")
    parser.add_argument('--disturbance_scale', default=0.3, type=float, help='choose disturbance scale')
    parser.add_argument('--disturbance_frequency', default=[2, 4, 8], type=list, help='choose disturbance frequency')

    # Etc
    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')

    args = parser.parse_args()

    return args

def main(args_tester):

    if args_tester.test_on is False:
        raise Exception(" You must turn on args_tester.test_on if you wanna test ")

    if args_tester.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_policy, env_name, algorithm_name, state_dim, action_dim, max_action, min_action, modelbased_mode, ensemble_mode\
        = load_config(args_tester)

    args, algorithm = get_algorithm_info(algorithm_name, state_dim, action_dim, device)

    random_seed = set_seed(args.random_seed)
    env, test_env = gym_env(env_name, random_seed)
    algorithm.actor.load_state_dict(torch.load(path_policy))

    print("Device: ", device)

    print("Random Seed:", random_seed)
    print("Domain type:", args.domain_type)
    print("Model based mode is", modelbased_mode)
    print("Test about", args_tester.develop_mode)
    print("Ensemble mode is", ensemble_mode)

    print("Environment", env_name)
    print("Algorithm:", algorithm_name)
    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Action range: {:.2f} ~ {:.2f}".format(min(min_action), max(max_action)))
    print("step time: {} (frame skip: {})".format(env.env.dt, env.env.frame_skip))

    if args_tester.develop_mode == 'Basic':
        trainer = Basic_trainer(
            env, test_env, algorithm, max_action, min_action, args, args_tester)
    else:
        trainer = Model_trainer(
            env, test_env, algorithm,
            state_dim, action_dim, max_action, min_action,
            args, args_tester, ensemble_mode)
    trainer.test()

if __name__ == '__main__':
    args_tester = hyperparameters()
    main(args_tester)