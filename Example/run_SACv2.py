import argparse, sys, os
from pathlib import Path
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Algorithm.SAC_v2 import SAC_v2
from Algorithm.ImageRL.SAC import ImageSAC_v2

from Trainer import *
from Common.Utils import *

def hyperparameters():
    parser = argparse.ArgumentParser(description='Soft Actor Critic (SAC) v2 example')

    # note in txt
    parser.add_argument('--note',
                        default="train for 5 ensemble among 5 models in ",
                        type=str, help='note about what to change')

    #environment
    parser.add_argument('--domain-type', default='gym', type=str, help='gym or dmc, dmc/image')
    parser.add_argument('--env-name', default='HalfCheetah-v3', help='Pendulum-v0, MountainCarContinuous-v0')
    parser.add_argument('--discrete', default=False, type=bool, help='Always Continuous')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=2000001, type=int, help='Maximum training step')
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
    parser.add_argument('--modelbased-mode', default=True, type=bool, help="you should choose whether basic or model_base")
    parser.add_argument('--ensemble-mode', default=True, type=bool, help="you should choose whether using an ensemble ")
    parser.add_argument('--ensemble-size', default=2, type=int, help="ensemble size")
    parser.add_argument('--model-batch-size', default=5, type=int, help="model batch size to use for ensemble")
    parser.add_argument('--net-type', default="DNN", help='DNN, BNN')
    parser.add_argument('--model-lr', default=0.001, type=float)
    parser.add_argument('--model-kl-weight', default=0.05, type=float)
    parser.add_argument('--inv-model-lr', default=0.001, type=float)
    parser.add_argument('--inv-model-kl-weight', default=0.1, type=float)

    # save path
    parser.add_argument('--path', default="/media/phb/Storage/env_mbrl/Results/Result/", help='path for save')

    args = parser.parse_args()

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

    #env setting
    if args.domain_type == 'gym':
        env, test_env = gym_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc':
        env, test_env = dmc_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc/image':
        env, test_env = dmc_image_env(args.env_name, args.image_size, args.frame_stack, args.frame_skip, random_seed)

    elif args.domain_type == 'dmcr':
        env, test_env = dmcr_env(args.env_name, args.image_size, args.frame_skip, random_seed, 'classic')

    state_dim = env.observation_space.shape[0]

    if args.domain_type in {'dmc/image', 'dmcr'}:
        state_dim = env.observation_space.shape

    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    if args.domain_type in {'gym', 'dmc'}:
        algorithm = SAC_v2(state_dim, action_dim, device, args)
    elif args.domain_type in {'dmc/image', 'dmcr'}:
        algorithm = ImageSAC_v2(state_dim, action_dim, device, args)

    create_config(algorithm.name, args, env, state_dim, action_dim, max_action, min_action)

    if args.modelbased_mode is False:
        trainer = Basic_trainer(
            env, test_env, algorithm, max_action, min_action, args)
    else:
        trainer = Model_trainer(
            env, test_env, algorithm, state_dim, action_dim, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

