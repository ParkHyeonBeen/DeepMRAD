import torch
import torch.nn as nn

from Common.Ensemble_model import Ensemble
from Network.Model_Network import *

def save_model(network, fname : str, root : str):
    if "DNN" in fname:
        torch.save(network.state_dict(), root + "saved_net/model/DNN/" + fname)
    elif "BNN" in fname:
        torch.save(network.state_dict(), root + "saved_net/model/BNN/" + fname)
    else:
        torch.save(network.state_dict(), root + "saved_net/model/Etc/" + fname)


def create_models(state_dim, action_dim, frameskip, algorithm, args, dnn=True, bnn=True, ensemble_mode=False):

    model_net_DNN = None
    inv_model_net_DNN = None
    model_net_BNN = None
    inv_model_net_BNN = None

    if ensemble_mode is True:
        if dnn is True:
            model_net_DNN = Ensemble(
                DynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, net_type="DNN"),
                ensemble_size=args.ensemble_size, model_batch_size=args.model_batch_size)
            inv_model_net_DNN = Ensemble(
                InverseDynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, net_type="DNN"),
                ensemble_size=args.ensemble_size, model_batch_size=args.model_batch_size)
        if bnn is True:
            model_net_BNN = Ensemble(
                DynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, net_type="BNN"),
                ensemble_size=args.ensemble_size, model_batch_size=args.model_batch_size)
            inv_model_net_BNN = Ensemble(
                InverseDynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, net_type="BNN"),
                ensemble_size=args.ensemble_size, model_batch_size=args.model_batch_size)
    else:
        if dnn is True:
            model_net_DNN = DynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, net_type="DNN")
            inv_model_net_DNN = InverseDynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, net_type="DNN")
        if bnn is True:
            model_net_BNN = DynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, net_type="BNN")
            inv_model_net_BNN = InverseDynamicsNetwork(state_dim, action_dim, frameskip, algorithm, args, net_type="BNN")

    if dnn is True and bnn is not True:
        return model_net_DNN, inv_model_net_DNN
    elif dnn is not True and bnn is True:
        return model_net_BNN, inv_model_net_BNN
    elif dnn is True and bnn is True:
        return model_net_DNN, model_net_BNN, inv_model_net_DNN, inv_model_net_BNN
    else:
        raise Exception("True at least one model")


def eval_models(state, action, next_state, *models):
    cost_list = []
    mse_list = []
    kl_list = []
    for model in models:
        _cost, _mse, _kl = model.eval_model(state, action, next_state)
        cost_list.append(_cost)
        mse_list.append(_mse)
        kl_list.append(_kl)

    costs = np.hstack(cost_list)
    mses = np.hstack(mse_list)
    kls = np.hstack(kl_list)

    return costs, mses, kls

def train_alls(training_step, *models):
    cost_list = []
    mse_list = []
    kl_list = []

    for model in models:
        _cost, _mse, _kl = model.train_all(training_step)
        cost_list.append(_cost)
        mse_list.append(_mse)
        kl_list.append(_kl)

    costs = np.hstack(cost_list)
    mses = np.hstack(mse_list)
    kls = np.hstack(kl_list)

    return costs, mses, kls

def save_models(args, cost, eval_cost, path, *models):

    name_list = ["modelDNN", "modelBNN", "invmodelDNN", "invmodelBNN"]

    if args.ensemble_mode is True:
        for i, model in enumerate(models):
            if cost[i] > eval_cost[i]:
                model.save_ensemble(name_list[i] + "_better", path)
                return [i, eval_cost[i]]
            else:
                model.save_ensemble(name_list[i] + "_current", path)

    else:
        for i, model in enumerate(models):
            if cost[i] > eval_cost[i]:
                save_model(model, name_list[i] + "_better", path)
                return [i, eval_cost[i]]
            else:
                save_model(model, name_list[i] + "_current", path)

def load_models(args_tester, model_net, inv_model_net, ensemble_mode=False):

    path = args_tester.path
    path_model = None
    path_invmodel = None

    if "DNN" in args_tester.modelnet_name:
        if args_tester.prev_result is True:
            path_model = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/DNN/" + args_tester.modelnet_name
            path_invmodel = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/DNN/inv" + args_tester.modelnet_name
        else:
            path_model = path + args_tester.result_index + "saved_net/model/DNN/" + args_tester.modelnet_name
            path_invmodel = path + args_tester.result_index + "saved_net/model/DNN/inv" + args_tester.modelnet_name

    if "BNN" in args_tester.modelnet_name:
        if args_tester.prev_result is True:
            path_model = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/BNN/" + args_tester.modelnet_name
            path_invmodel = path + "storage/" + args_tester.prev_result_fname + "/saved_net/model/BNN/inv" + args_tester.modelnet_name
        else:
            path_model = path + args_tester.result_index + "saved_net/model/BNN/" + args_tester.modelnet_name
            path_invmodel = path + args_tester.result_index + "saved_net/model/BNN/inv" + args_tester.modelnet_name

    if ensemble_mode is True:
        model_net.load_ensemble(path_model)
        inv_model_net.load_ensemble(path_invmodel)
    else:
        model_net.load_state_dict(torch.load(path_model))
        inv_model_net.load_state_dict(torch.load(path_invmodel))
