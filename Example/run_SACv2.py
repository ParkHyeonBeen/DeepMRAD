import argparse, sys, os
from pathlib import Path
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Algorithm.SAC_v2 import SAC_v2
from Algorithm.ImageRL.SAC import ImageSAC_v2

from Trainer import *
from Common.Utils import *

torch.autograd.set_detect_anomaly(True)

def hyperparameters():
    parser = argparse.ArgumentParser(description='Soft Actor Critic (SAC) v2 example')

    # note in txt
    parser.add_argument('--note',
                        default="change bnn std 0.1 --> 0.01",
                        type=str, help='note about what to change')

    #environment
    parser.add_argument('--domain-type', default='gym', type=str, help='gym or dmc, dmc/image, suite')
    parser.add_argument('--env-name', default='Walker2d-v3', help='Pendulum-v0, MountainCarContinuous-v0, Door')
    parser.add_argument('--robots', default='Panda', help='if domain type is suite, choose the robots')
    parser.add_argument('--discrete', default=False, type=bool, help='Always Continuous')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=5000001, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=5, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #sac
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='online', help='offline, online')
    parser.add_argument('--training-step', default=1, type=int)
    parser.add_argument('--train-alpha', default=False, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--alpha-lr', default=0.0001, type=float)
    parser.add_argument('--encoder-lr', default=0.001, type=float)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--critic-update', default=1, type=int)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')
    #image
    parser.add_argument('--frame-stack', default=3, type=int)
    parser.add_argument('--frame-skip', default=8, type=int)
    parser.add_argument('--image-size', default=84, type=int)
    parser.add_argument('--layer-num', default=4, type=int)
    parser.add_argument('--filter-num', default=32, type=int)
    parser.add_argument('--encoder-tau', default=0.05, type=float)
    parser.add_argument('--feature-dim', default=50, type=int)

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')
    parser.add_argument('--log', default=False, type=bool, help='use tensorboard summary writer to log, if false, cannot use the features below')
    parser.add_argument('--tensorboard', default=True, type=bool, help='when logged, write in tensorboard')
    parser.add_argument('--file', default=False, type=bool, help='when logged, write log')
    parser.add_argument('--numpy', default=False, type=bool, help='when logged, save log in numpy')

    parser.add_argument('--model', default=False, type=bool, help='when logged, save model')
    parser.add_argument('--model-freq', default=10000, type=int, help='model saving frequency')
    parser.add_argument('--buffer', default=False, type=bool, help='when logged, save buffer')
    parser.add_argument('--buffer-freq', default=10000, type=int, help='buffer saving frequency')

    # estimate a model dynamics
    parser.add_argument('--modelbased-mode', default=False, type=bool, help="you should choose whether basic or model_base")
    parser.add_argument('--develop-mode', '-dm', default='DeepDOB', help="Both, DeepDOB, MRAP")
    parser.add_argument('--ensemble-mode', default=False, type=bool, help="you should choose whether using an ensemble ")
    parser.add_argument('--ensemble-size', default=2, type=int, help="ensemble size")
    parser.add_argument('--model-batch-size', default=5, type=int, help="model batch size to use for ensemble")
    parser.add_argument('--net-type', default="all", help='all, DNN, BNN')
    parser.add_argument('--model-lr-dnn', default=0.001, type=float)
    parser.add_argument('--model-lr-bnn', default=0.001, type=float)
    parser.add_argument('--model-kl-weight', default=0.05, type=float)
    parser.add_argument('--inv-model-lr-dnn', default=0.001, type=float)
    parser.add_argument('--inv-model-lr-bnn', default=0.01, type=float)
    parser.add_argument('--inv-model-kl-weight', default=0.01, type=float)
    parser.add_argument('--use-random-buffer', default=False, type=bool, help="add random action to training data")

    # save path
    parser.add_argument('--path', default="/media/phb/Storage/env_mbrl/Results/Result2/", help='path for save')

    args = parser.parse_known_args()

    if len(args) != 1:
        args = args[0]

    return args

def main(args):

    if args.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Device: ", device)
    # random seed setting
    random_seed = set_seed(args.random_seed)
    print("Random Seed:", random_seed)
    print("Domain type:", args.domain_type)
    print("Develop mode:", args.develop_mode)
    print("Model based mode:", args.modelbased_mode)
    print("Ensemble mode:", args.ensemble_mode)

    #env setting
    if args.domain_type == 'gym':
        env, test_env = gym_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc':
        env, test_env = dmc_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc/image':
        env, test_env = dmc_image_env(args.env_name, args.image_size, args.frame_stack, args.frame_skip, random_seed)

    elif args.domain_type == 'dmcr':
        env, test_env = dmcr_env(args.env_name, args.image_size, args.frame_skip, random_seed, 'classic')

    elif args.domain_type == 'suite':
        env, test_env = suite_env(args.env_name, args.robots, args.render, False, False)

    if args.domain_type == 'suite':
        state_dim = 0
        for key in env.active_observables:
            if type(env.observation_spec()[key]) is int:
                state_dim += 1
            else:
                state_dim += len(env.observation_spec()[key])

    elif args.domain_type in {'dmc/image', 'dmcr'}:
        state_dim = env.observation_space.shape
    else:
        state_dim = env.observation_space.shape[0]

    if args.domain_type == 'suite':
        action_dim = env.action_dim
        max_action = env.action_spec[1]
        min_action = env.action_spec[0]
    else:
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high
        min_action = env.action_space.low

    if args.domain_type in {'gym', 'dmc', 'suite'}:
        algorithm = SAC_v2(state_dim, action_dim, device, args)
    elif args.domain_type in {'dmc/image', 'dmcr'}:
        algorithm = ImageSAC_v2(state_dim, action_dim, device, args)

    create_config(algorithm.name, args, env, state_dim, action_dim, max_action, min_action)

    if args.modelbased_mode is False:
        trainer = Basic_trainer(
            env, test_env, algorithm, state_dim, action_dim, max_action, min_action, args)
    else:
        trainer = Model_trainer(
            env, test_env, algorithm, state_dim, action_dim, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

