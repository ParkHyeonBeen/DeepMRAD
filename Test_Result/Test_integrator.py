import time

import numpy as np
import Tester
from Tester import hyperparameters
from Common.Utils import *

Tester_data = DataManager()

result_fname = "0327_Walker2d-v3_esb"
num_test = 100
policy_name = "policy_current"
develop_list = ['Basic']
model_list = ["modelDNN_current", "modelBNN_current"]
save_dir = 'X:/env_mbrl/Results/Integrated_log/'
num_dist = 20
dist_kind = 'normal'    # 'normal', 'irregular'
add_to = 'state'

print("result name:", result_fname)
print("action range: +-1")
print("the kind of external force:", dist_kind)
print("the number of external force:", num_dist)

print("start time : %s" % time.strftime("%Y%m%d-%H%M%S"))
start_time = time.time()
for mode in develop_list:
    if mode == 'Basic':
        print("start Basic PG algorithm")
        Tester_data.init_data()
        for i in np.linspace(0, 0.1, num_dist+1):
            # print("current external force scale :", i)
            if dist_kind == 'irregular':
                args = hyperparameters(result_fname=result_fname,
                                       num_test=num_test,
                                       develop_mode=mode,
                                       disturbance_scale=i,
                                       add_to=add_to,
                                       policy_name=policy_name
                                       )
            else:
                args = hyperparameters(result_fname=result_fname,
                                       num_test=num_test,
                                       develop_mode=mode,
                                       noise_scale=i,
                                       add_to=add_to,
                                       policy_name=policy_name
                                       )
            reward_avg, reward_max, reward_min, reward_std, alive_rate = Tester.main(args)
            saveData = np.array([reward_avg, reward_max, reward_min, reward_std, alive_rate])
            Tester_data.put_data(saveData)
        Tester_data.save_data(save_dir, result_fname + '_' + mode + '_' + dist_kind)
        print("finish time of Basic: %s" % time.strftime("%Y%m%d-%H%M%S"))
        print("elapsed time : ", time.time() - start_time)
    else:
        print("start DeepDOB algorithm")
        for model in model_list:
            print("start time of "+model+": %s" % time.strftime("%Y%m%d-%H%M%S"))
            start_time = time.time()
            print("The model to test :", model)
            Tester_data.init_data()
            for i in np.linspace(0, 0.1, num_dist+1):
                # print("current external force scale :", i)
                if dist_kind == 'irregular':
                    args = hyperparameters(result_fname=result_fname,
                                           num_test=num_test,
                                           develop_mode=mode,
                                           disturbance_scale=i,
                                           add_to=add_to,
                                           policy_name=policy_name,
                                           model_name=model
                                           )
                else:
                    args = hyperparameters(result_fname=result_fname,
                                           num_test=num_test,
                                           develop_mode=mode,
                                           noise_scale=i,
                                           add_to=add_to,
                                           policy_name=policy_name,
                                           model_name=model
                                           )
                reward_avg, reward_max, reward_min, reward_std, alive_rate = Tester.main(args)
                saveData = np.array([reward_avg, reward_max, reward_min, reward_std, alive_rate])
                Tester_data.put_data(saveData)
            Tester_data.save_data(save_dir, result_fname + '_' + model + '_' + dist_kind)
            print("finish time of" + model + ": %s" % time.strftime("%Y%m%d-%H%M%S"))
            print("elapsed time : ", time.time() - start_time)