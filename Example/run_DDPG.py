import argparse, sys
from pathlib import Path
import torch

sys.path.append(str(Path('run_SACv2.py').parent.absolute()))   # 절대 경로에 추가


from Algorithm.DDPG import DDPG
from Common.Utils import set_seed, gym_env, dmc_env

from Trainer import *

def hyperparameters():
    parser = argparse.ArgumentParser(description='Deep Deterministic Policy Gradient(DDPG) example')

    # note in txt
    parser.add_argument('--note',
                        default="use frame skip instead of steptime to compute differential states",
                        type=str, help='note about what to change')

    #environment
    parser.add_argument('--domain_type', default='gym', type=str, help='gym or dmc')
    parser.add_argument('--env-name', default='Humanoid-v3', help='Pendulum-v0, MountainCarContinuous-v0')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--discrete', default=False, type=bool, help='Always Continuous')
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--max-step', default=2000001, type=int, help='Maximum training step')
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--eval-step', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=5, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')
    #ddpg
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--train-mode', default='offline', help='offline, online')
    parser.add_argument('--training-step', default=200, type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--noise-scale', default=0.1, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')

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
    parser.add_argument('--develop-mode', default=True, type=bool, help="you should choose whether basic or model_base")
    parser.add_argument('--ensemble-mode', default=True, type=bool, help="you should choose whether using an ensemble ")
    parser.add_argument('--net-type', default="DNN", help='DNN, BNN')
    parser.add_argument('--model-lr', default=0.001, type=float)
    parser.add_argument('--model-kl-weight', default=0.05, type=float)
    parser.add_argument('--inv-model-lr', default=0.001, type=float)
    parser.add_argument('--inv-model-kl-weight', default=0.1, type=float)

    # save path
    parser.add_argument('--path', default="X:/env_mbrl/Results/Result/", help='path for save')

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

    #env setting
    if args.domain_type == 'gym':
        env, test_env = gym_env(args.env_name, random_seed)

    elif args.domain_type == 'dmc':
        env, test_env = dmc_env(args.env_name, random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high
    min_action = env.action_space.low

    algorithm = DDPG(state_dim, action_dim, device, args)

    with open(args.path + 'config.txt', 'w') as f:

        print("Training of", args.domain_type + '_' + args.env_name, file=f)
        print("Algorithm:", algorithm.name, file=f)
        print("State dim:", state_dim, file=f)
        print("Action dim:", action_dim, file=f)
        print("Action range: {:.2f} ~ {:.2f}".format(min(min_action), max(max_action)), file=f)
        print("step size: {} (frame skip: {})".format(env.env.dt, env.env.frame_skip), file=f)

        print("save path : ", args.path, file=f)
        print("model lr : {}, model klweight : {}, inv model lr : {}, inv model klweight : {}".
              format(args.model_lr, args.model_kl_weight, args.inv_model_lr, args.inv_model_kl_weight), file=f)

        print("consideration note : ", args.note, file=f)

    if args.develop_mode is False:
        trainer = Basic_trainer(
            env, test_env, algorithm, max_action, min_action, args)
    else:
        trainer = Model_trainer(
            env, test_env, algorithm, state_dim, action_dim, max_action, min_action, args)
    trainer.run()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)

