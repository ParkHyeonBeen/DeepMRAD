import numpy as np
import matplotlib.pyplot as plt
import os, argparse

def hyperparameters():
    parser = argparse.ArgumentParser(description='Result viewer')

    parser.add_argument('--watch-cost', default=True, type=bool, help='if you wanna watch cost graph, True')
    parser.add_argument('--watch-reward', default=False, type=bool, help='if you wanna watch reward graph, True')
    parser.add_argument('--is-eval', default=False, type=bool, help='whether at evaluation or training')

    parser.add_argument('--data-type', default="normal", type=str, help="normal, path")
    parser.add_argument('--file-type', default=".csv", type=str, help=".csv, .npy")
    parser.add_argument('--start-index', default=1, type=int, help='start index of plot to be viewed')
    parser.add_argument('--data-index', default=7, type=int, help='data index to be viewed')

    parser.add_argument('--path', default="X:/env_mbrl/Results/", help='path of saved data')
    parser.add_argument('--prev-result', default=True, type=bool, help='if previous result, True')
    parser.add_argument('--prev-result-fname', default="0317_Hopper-v3/", help='choose the result to view')

    args = parser.parse_args()

    return args

def main(args):

    if args.prev_result is False:
        path_base = args.path + 'Result/saved_log/'
    else:
        path_base = args.path + 'storage/' + args.prev_result_fname + 'saved_log/'
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
            if i == 1:
                cost_data = cost_data_
            else:
                cost_data = np.hstack((cost_data, cost_data_))
            i += 1

        plt.figure('cost' + str(args.data_index))
        plt.plot(cost_data)
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