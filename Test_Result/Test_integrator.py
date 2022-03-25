import time

import numpy as np
import Tester
from Tester import hyperparameters
from Common.Utils import *

model_name = 'Walker2d-v3'
max_action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
min_action = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
develop_list = ['DeepDOB']
model_list = ["modelDNN_better", "modelBNN_better"]
save_dir = 'X:/env_mbrl/Results/Integrated_log/'
num_dist = 40

print("model name:", model_name)
print("action range: +-1")
print("the number of disturbance:", num_dist)

print("start time : %s" % time.strftime("%Y%m%d-%H%M%S"))
start_time = time.time()
for mode in develop_list:
    if mode == 'Basic':
        print("start Basic PG algorithm")
        init_data()
        for i in np.linspace(0, 1, num_dist+1):
            print("current disturbance scale :", i)
            args = hyperparameters(disturbance_scale=i)
            reward_avg, reward_max, reward_min, reward_std, alive_rate = Tester.main(args)
            saveData = np.array([reward_avg, reward_max, reward_min, reward_std, alive_rate])
            put_data(saveData)
        save_data(save_dir, 'Basic')
        print("finish time of Basic: %s" % time.strftime("%Y%m%d-%H%M%S"))
        print("elapsed time : ", time.time() - start_time)
    else:
        print("start DeepDOB algorithm")
        for model in model_list:
            print("start time of "+model+": %s" % time.strftime("%Y%m%d-%H%M%S"))
            start_time = time.time()
            print("The model to test :", model)
            init_data()
            for i in np.linspace(0, 1, num_dist+1):
                print("current disturbance scale :", i)
                args = hyperparameters(develop_mode=mode, disturbance_scale=i, model_name=model)
                reward_avg, reward_max, reward_min, reward_std, alive_rate = Tester.main(args)
                saveData = np.array([reward_avg, reward_max, reward_min, reward_std, alive_rate])
                put_data(saveData)
            save_data(save_dir, model)
            print("finish time of" + model + ": %s" % time.strftime("%Y%m%d-%H%M%S"))
            print("elapsed time : ", time.time() - start_time)

