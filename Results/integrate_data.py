import numpy as np
import matplotlib.pyplot as plt
import os, argparse, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))\

from Common.Utils import *

def hyperparameters():
    parser = argparse.ArgumentParser(description='Result viewer')

    parser.add_argument('--watch-cost', default='True', type=str2bool, help='if you wanna watch cost graph, True')
    parser.add_argument('--watch-reward', default='False', type=str2bool, help='if you wanna watch reward graph, True')
    parser.add_argument('--is-eval', default='True', type=str2bool, help='whether at evaluation or training')

    parser.add_argument('--data-type', default="normal", type=str, help="normal, path")
    parser.add_argument('--file-type', default=".csv", type=str, help=".csv, .npy")
    parser.add_argument('--start-index', default=1, type=int, help='start index of plot to be viewed')
    parser.add_argument('--data-index', default=1, type=int, help='data index to be viewed')
    parser.add_argument('--data-form', default="all", type=str, help='all, mean, std')

    parser.add_argument('--path', default="/media/phb/Storage/env_mbrl/Results/", help='path of saved data')
    parser.add_argument('--result-index', default="hopper_dnn_esb", type=str, help='result to check')
    parser.add_argument('--prev-result', default='False', type=str2bool, help='if previous result, True')
    parser.add_argument('--prev-result-fname', default="0501_Hopper/", help='choose the result to view')

    args = parser.parse_args()

    return args

def main(args):

    if args.prev_result is False:
        path_base = args.path + args.result_index + '/saved_log/'
    else:
        path_base = args.path + 'storage/_prev/trash/' + args.prev_result_fname + 'saved_log/'
    i = args.start_index

    if args.watch_cost is True:
        cost_data = None

        while True:
            if args.is_eval is True:
                path = path_base + 'Eval_' + str(i) + args.file_type
            else:
                path = path_base + 'Eval_by' + str(i) + args.file_type
            isfile = os.path.isfile(path)
            if isfile is False:
                break
            print(path)
            file = np.loadtxt(path, skiprows=1, delimiter = ',', dtype ='float')
            cost_data_ = np.asfarray(np.array(file[:, args.data_index]), float)

            if args.is_eval is False:
                cost_data_ = np.sqrt(cost_data_)

            if args.data_form == "mean":
                cost_data_ = np.mean(cost_data_)
            if args.data_form == "std":
                cost_data_ = np.std(cost_data_)

            if i == 1:
                cost_data = cost_data_
            else:
                cost_data = np.hstack((cost_data, cost_data_))
            i += 1

        plt.figure('cost' + str(args.data_index))
        plt.plot(cost_data, "o")
    else:
        pass

    if args.watch_reward is True:
        path = path_base + 'reward' + args.file_type
        file = np.loadtxt(path, skiprows=1, delimiter = ',', dtype ='float')
        data = np.asfarray(np.array(file[:, 1]), float)
        plt.figure('reward')
        plt.plot(data)

    plt.show()

if __name__ == '__main__':
    args = hyperparameters()
    main(args)