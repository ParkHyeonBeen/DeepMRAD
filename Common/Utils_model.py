from Common.Ensemble_model import Ensemble
from Common.Utils import *
from Network.Model_Network import *

def save_model(network, fname : str, root : str):
    if "DNN" in fname:
        torch.save(network.state_dict(), root + "saved_net/model/DNN/" + fname)
    elif "BNN" in fname:
        torch.save(network.state_dict(), root + "saved_net/model/BNN/" + fname)
    else:
        torch.save(network.state_dict(), root + "saved_net/model/Etc/" + fname)


def create_models(state_dim, action_dim, frameskip, algorithm, args, args_tester,
                  buffer=None, dnn=True, bnn=True, ensemble_mode=False):

    model_net_DNN = None
    inv_model_net_DNN = None
    model_net_BNN = None
    inv_model_net_BNN = None

    if args_tester is not None:
        ensemble_size = args_tester.ensemble_size
    else:
        ensemble_size = args.ensemble_size

    if ensemble_mode is True:
        if dnn is True and args.develop_mode != 'DeepDOB':
            model_net_DNN = Ensemble('modelNN', state_dim, action_dim, frameskip, algorithm, args,
                                     buffer=buffer, net_type="DNN", score_len=args.eval_episode,
                                     ensemble_size=ensemble_size, model_batch_size=args.model_batch_size)
        if dnn is True and args.develop_mode != 'MRAP':
            inv_model_net_DNN = Ensemble('inv_modelNN', state_dim, action_dim, frameskip, algorithm, args,
                                         buffer=buffer, net_type="DNN", score_len=args.eval_episode,
                                         ensemble_size=ensemble_size, model_batch_size=args.model_batch_size)
        if bnn is True and args.develop_mode != 'DeepDOB':
            model_net_BNN = Ensemble('modelNN', state_dim, action_dim, frameskip, algorithm, args,
                                     buffer=buffer, net_type="BNN", score_len=args.eval_episode,
                                     ensemble_size=ensemble_size, model_batch_size=args.model_batch_size)
        if bnn is True and args.develop_mode != 'MRAP':
            inv_model_net_BNN = Ensemble('inv_modelNN', state_dim, action_dim, frameskip, algorithm, args,
                                         buffer=buffer, net_type="BNN", score_len=args.eval_episode,
                                         ensemble_size=ensemble_size, model_batch_size=args.model_batch_size)
    else:
        if dnn is True and args.develop_mode != 'DeepDOB':
            model_net_DNN = DynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args,
                                            buffer=buffer, net_type="DNN")
        if dnn is True and args.develop_mode != 'MRAP':
            inv_model_net_DNN = InverseDynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args,
                                                       buffer=buffer, net_type="DNN")
        if bnn is True and args.develop_mode != 'DeepDOB':
            model_net_BNN = DynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args,
                                            buffer=buffer, net_type="BNN")
        if bnn is True and args.develop_mode != 'MRAP':
            inv_model_net_BNN = InverseDynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args,
                                                       buffer=buffer, net_type="BNN")

    return list(filter(None, [model_net_DNN, inv_model_net_DNN, model_net_BNN, inv_model_net_BNN]))

def eval_models(state, action, next_state, models):
    error_list = []
    for model in models:
        _error = model.eval_model(state, action, next_state)
        error_list.append(_error)
    errors = np.hstack(error_list)
    return errors

def train_alls(training_step, models):
    cost_list = []
    mse_list = []
    kl_list = []

    if len(models) == 0:
        raise Exception("your models is empty now")

    for model in models:
        _cost, _mse, _kl = model.train_all(training_step)
        cost_list.append(_cost)
        mse_list.append(_mse)
        kl_list.append(_kl)

    costs = np.hstack(cost_list)
    mses = np.hstack(mse_list)
    kls = np.hstack(kl_list)

    return costs, mses, kls

def save_models(args, loss, eval_loss, path, models):
    print(loss, eval_loss)

    if args.develop_mode == "DeepDOB":
        if args.net_type == "DNN":
            name_list = ["invmodelDNN"]
        elif args.net_type == "BNN":
            name_list = ["invmodelBNN"]
        else:
            name_list = ["invmodelDNN", "invmodelBNN"]
    elif args.develop_mode == "MRAP":
        if args.net_type == "DNN":
            name_list = ["modelDNN"]
        elif args.net_type == "BNN":
            name_list = ["modelBNN"]
        else:
            name_list = ["modelDNN", "modelBNN"]
    else:
        name_list = ["modelDNN", "modelBNN", "invmodelDNN", "invmodelBNN"]

    if len(name_list) != len(models) or len(models) == 0:
        raise Exception("your models is empty now")

    if args.ensemble_mode is True:
        for i, model in enumerate(models):
            if loss[i] > eval_loss[i]:
                model.save_ensemble(name_list[i] + "_esb_better", path)
                return [i, eval_loss[i]]
            else:
                model.save_ensemble(name_list[i] + "_esb_current", path)

    else:
        for i, model in enumerate(models):
            if loss[i] > eval_loss[i]:
                save_model(model, name_list[i] + "_better", path)
                return [i, eval_loss[i]]
            else:
                save_model(model, name_list[i] + "_current", path)

def load_models(args_tester, model, ensemble_mode=False):

    path = args_tester.path
    path_model = None
    path_invmodel = None

    if "DNN" in args_tester.modelnet_name:
        if args_tester.prev_result is True:
            path_model = path + "storage/_prev/trash/" + args_tester.prev_result_fname + "/saved_net/model/DNN/" + args_tester.modelnet_name
            path_invmodel = path + "storage/_prev/trash/" + args_tester.prev_result_fname + "/saved_net/model/DNN/inv" + args_tester.modelnet_name
        else:
            path_model = path + args_tester.result_index + "saved_net/model/DNN/" + args_tester.modelnet_name
            path_invmodel = path + args_tester.result_index + "saved_net/model/DNN/inv" + args_tester.modelnet_name

    if "BNN" in args_tester.modelnet_name:
        if args_tester.prev_result is True:
            path_model = path + "storage/_prev/trash/" + args_tester.prev_result_fname + "/saved_net/model/BNN/" + args_tester.modelnet_name
            path_invmodel = path + "storage/_prev/trash/" + args_tester.prev_result_fname + "/saved_net/model/BNN/inv" + args_tester.modelnet_name
        else:
            path_model = path + args_tester.result_index + "saved_net/model/BNN/" + args_tester.modelnet_name
            path_invmodel = path + args_tester.result_index + "saved_net/model/BNN/inv" + args_tester.modelnet_name

    if ensemble_mode is True:
        if args_tester.develop_mode == "MRAP":
            model.load_ensemble(path_model, args_tester.ensemble_size)
        if args_tester.develop_mode == "DeepDOB":
            model.load_ensemble(path_invmodel, args_tester.ensemble_size)
    else:
        if args_tester.develop_mode == "MRAP":
            model.load_state_dict(torch.load(path_model))
        if args_tester.develop_mode == "DeepDOB":
            model.load_state_dict(torch.load(path_invmodel))


def validate_measure(error_list):
    error_max = np.max(error_list, axis=0)
    mean = np.mean(error_list, axis=0)
    std = np.std(error_list, axis=0)
    loss = np.sqrt(mean**2 + std**2)

    return [loss, mean, std, error_max]

def get_random_action_batch(observation, env_action, test_env, model_buffer, max_action, min_action):

    env_action_noise, _ = add_noise(env_action, scale=0.1)
    action_noise = normalize(env_action_noise, max_action, min_action)
    next_observation, reward, done, info = test_env.step(env_action_noise)
    model_buffer.add(observation, action_noise, reward, next_observation, float(done))

def set_sync_env(env, test_env):

    position = env.sim.data.qpos.flat.copy()
    velocity = env.sim.data.qvel.flat.copy()

    test_env.set_state(position, velocity)