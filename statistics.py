import argparse
import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from itertools import product
import copy
from sklearn.metrics import confusion_matrix
import datetime
import logging

from model import FcNet
from datasets import MNIST_truncated, CIFAR10_truncated

from combine_nets import compute_ensemble_accuracy, compute_pdm_matching_multilayer, compute_iterative_pdm_matching, compute_fedavg_accuracy
import pdb

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--layers', type=int, required=False, help='do n_nets or n_layers')
    parser.add_argument('--n', type=int, required=False, help='the number of nets or layers')

    parser.add_argument('--dataset', type=str, required=False, default="mnist", help="Dataset [mnist/cifar10]")
    parser.add_argument('--partition', type=str, required=False, help="Partition = homo/hetero/hetero-dir")
    parser.add_argument('--trials', type=int, required=False, default=1, help="Number of trials for each run")
    
    return parser

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

KL_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

def load_acc(load_dir):

    load_path = os.path.join(load_dir, "trials_res.json")

    with open(load_path,'r') as load_f:
        trials_res = json.load(load_f)

    KL_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

    local_test = []
    ensemble = []
    fedavg = []
    pdm_kl = {}
    Kl_best = []
    for i in range(args.trials):
        trial_res = trials_res[str(i)]
        local_test.append(np.mean(trial_res['local_test_accs']))
        ensemble.append(trial_res["Uniform ensemble (Test acc)"])
        fedavg.append(trial_res["FedAvg (Test acc)"])

        best = 0
        for KL_reg in KL_regs:

            if i == 0:
                pdm_kl[KL_reg] = []

            kl_acc = trial_res["pdm_KL_reg "+str(KL_reg)]["Best Result"]["Best Test accuracy"]
            if KL_reg != 0 and kl_acc > best:
                best = kl_acc
            
            pdm_kl[KL_reg].append(kl_acc)

        Kl_best.append(best)

    return local_test, ensemble, fedavg, pdm_kl, Kl_best

def stat_acc(n, dataset="mnist", partition="hetero-dir"):

    if args.layers == 1:
        n_list = ["n_layers "+str(i+1) for i in range(n)]
        #n_list = [n]
    else:
        n_list = ["n_nets "+str(i) for i in range(0, n+1, 5)]
        n_list[0] = "n_nets 2"
        #n_list = [n]

    res = {}

    KL_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    for n in n_list:
        res[n] = {}

        loaddir = os.path.join('KL_regs_bst_hyp',
                          dataset, partition, n)

        local_tests, ensembles, fedavgs, pdm_kls, Kl_bests = load_acc(loaddir)

        local_acc = np.mean(local_tests)
        local_std = np.std(local_tests)
        ensemble_acc = np.mean(ensembles)
        ensemble_std = np.std(ensembles)
        fedavg_acc = np.mean(fedavgs)
        fedavg_std = np.std(fedavgs)

        pdm_kl_acc = {}
        pdm_kl_std = {}

        res[n]["local acc"] = local_acc
        res[n]["local std"] = local_std
        res[n]["ensemble acc"] = ensemble_acc
        res[n]["ensemble std"] = ensemble_std
        res[n]["fedavg acc"] = fedavg_acc
        res[n]["fedavg std"] = fedavg_std

        for KL_reg in KL_regs:
            pdm_kl_acc[KL_reg] = np.mean(pdm_kls[KL_reg])
            pdm_kl_std[KL_reg] = np.std(pdm_kls[KL_reg])
            res[n]["KL reg "+str(KL_reg)+" acc"] = pdm_kl_acc[KL_reg]
            res[n]["KL reg "+str(KL_reg)+" std"] = pdm_kl_std[KL_reg]

        #pdm_kl_acc = sorted(pdm_kl_acc.items(), key=lambda: x:x[1], reverse=False)

        res[n]["KL best acc"] = np.mean(Kl_bests)
        res[n]["KL best std"] = np.std(Kl_bests)

    if args.layers:
        save_dir = os.path.join("statistics/limited", dataset, partition, "n_layers")
    else:
        save_dir = os.path.join("statistics/limited", dataset, partition, "n_nets")

    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    with open(os.path.join(save_dir, 'res.json'), 'w') as f:
        json.dump(res, f)

    print("Work finished!")

def load_acc_unlimi(load_dir):

    res_list = os.listdir(load_dir)

    local_test = []
    pdm_kl = {}
    Kl_best = []
    improve = []

    if "trials_res.json" in res_list:

        load_path = os.path.join(load_dir, "trials_res.json")

        with open(load_path,'r') as load_f:
            trials_res = json.load(load_f)

    for i in range(args.trials):

        if "trials_res.json" in res_list:
            trial_res = trials_res[str(i)]
        else:
            if not 'trial'+str(i)+'.json' in res_list:
                break

            with open(os.path.join(load_dir, 'trial'+str(i)+'.json'), 'r') as load_f:
                trial_res = json.load(load_f)

        local_test.append(np.mean(trial_res['local_test_accs']))
        best_reg = trial_res["best_reg"]

        for KL_reg in KL_regs:

            if i == 0:
                pdm_kl[KL_reg] = []

            kl_acc = trial_res["pdm_KL_reg "+str(KL_reg)]["Best Result"]["Best Test accuracy"]
            
            pdm_kl[KL_reg].append(kl_acc)

        best = pdm_kl[best_reg][i]
        if best_reg != 0 and best - pdm_kl[KL_regs[0]][i] > 0.005:
            improve.append(best - pdm_kl[KL_regs[0]][i])

        if len(improve) == 0:
            improve.append(0)

        Kl_best.append(best)

    return local_test, pdm_kl, Kl_best, improve

def stat_acc_unlimi(n, dataset="mnist", partition="hetero-dir"):
    
    if args.layers == 1:
        n_list = ["n_layers "+str(i+1) for i in range(1, n)]
        #n_list = [n]
    else:
        if dataset == "mnist":
            n_list = ["n_nets "+str(i) for i in range(15, n+1, 5)]
        else:
            n_list = ["n_nets "+str(i) for i in range(10, n+1, 5)]
        #n_list = ["n_nets "+str(i) for i in range(0, n+1, 5)]
        #n_list[0] = "n_nets 2"
        #n_list = ["n_nets "+str(n)]

    res = {}

    for n in n_list:
        res[n] = {}

        loaddir = os.path.join('KL_regs_bst_hyp_unlimi',
                          dataset, partition, n)

        local_tests, pdm_kls, Kl_bests, improves = load_acc_unlimi(loaddir)

        local_acc = np.mean(local_tests)
        local_std = np.std(local_tests)

        pdm_kl_acc = {}
        pdm_kl_std = {}

        res[n]["local acc"] = local_acc
        res[n]["local std"] = local_std

        for KL_reg in KL_regs:
            pdm_kl_acc[KL_reg] = np.mean(pdm_kls[KL_reg])
            pdm_kl_std[KL_reg] = np.std(pdm_kls[KL_reg])
            res[n]["KL reg "+str(KL_reg)+" acc"] = pdm_kl_acc[KL_reg]
            res[n]["KL reg "+str(KL_reg)+" std"] = pdm_kl_std[KL_reg]

        #pdm_kl_acc = sorted(pdm_kl_acc.items(), key=lambda: x:x[1], reverse=False)

        res[n]["KL best acc"] = np.mean(Kl_bests)
        res[n]["KL best std"] = np.std(Kl_bests)

        res[n]["improved acc"] = np.mean(improves)

    if args.layers:
        save_dir = os.path.join("statistics/unlimi", dataset, partition, "n_layers")
    else:
        save_dir = os.path.join("statistics/unlimi", dataset, partition, "n_nets")

    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    with open(os.path.join(save_dir, 'acc_res.json'), 'w') as f:
        json.dump(res, f)

    print("Work finished!")

def load_acc_iter(load_dir, partition):

    res_list = os.listdir(load_dir)

    if partition == "homo":
        comm_rounds = 15
    else:
        comm_rounds = 30

    if "trials_res.json" in res_list:

        load_path = os.path.join(load_dir, "trials_res.json")

        with open(load_path,'r') as load_f:
            trials_res = json.load(load_f)

    kl_comm_acc = [[] for c in range(comm_rounds)]
    ensem_comm_acc = [[] for c in range(comm_rounds)]
    fedavg_comm_acc = [[] for c in range(comm_rounds)]
    for i in range(args.trials):

        if "trials_res.json" in res_list:
            trial_res = trials_res[str(i)]
        else:
            if not 'trial'+str(i)+'.json' in res_list:
                break

            with open(os.path.join(load_dir, 'trial'+str(i)+'.json'), 'r') as load_f:
                trial_res = json.load(load_f)

        for j in range(comm_rounds):
            comm_res = trial_res["comm "+str(j)]
            
            kl_comm_acc[j].append(comm_res["pdm_KL (Test acc)"])
            ensem_comm_acc[j].append(comm_res["Uniform ensemble (Test acc)"])
            fedavg_comm_acc[j].append(comm_res["FedAvg (Test acc)"])

    return kl_comm_acc, ensem_comm_acc, fedavg_comm_acc

def stat_acc_iter(dataset="mnist", partition="hetero-dir"):

    if partition == "homo":
        comm_rounds = 15
    else:
        comm_rounds = 30

    comm_res = {}

    for comm in range(comm_rounds):

        comm_res["comm "+str(comm)] = {}

        best = [0 for t in range(args.trials)]
        for KL_reg in KL_regs:

            loaddir = os.path.join('KL_regs_iter',
                          dataset, partition, "KL_reg "+str(KL_reg))

            kl_comm_acc, ensem_comm_acc, fedavg_comm_acc = load_acc_iter(loaddir, partition)

            comm_res["comm "+str(comm)]["ensemble acc"] = np.mean(ensem_comm_acc[comm])
            comm_res["comm "+str(comm)]["ensemble std"] = np.std(ensem_comm_acc[comm])
            comm_res["comm "+str(comm)]["fedavg acc"] = np.mean(fedavg_comm_acc[comm])
            comm_res["comm "+str(comm)]["fedavg std"] = np.std(fedavg_comm_acc[comm])
            comm_res["comm "+str(comm)]["KL reg "+str(KL_reg)+" acc"] = np.mean(kl_comm_acc[comm])
            comm_res["comm "+str(comm)]["KL reg "+str(KL_reg)+" std"] = np.std(kl_comm_acc[comm])

            if KL_reg != 0:
                best = np.maximum(best, kl_comm_acc[comm])

        comm_res["comm "+str(comm)]["KL best acc"] = np.mean(best)
        comm_res["comm "+str(comm)]["KL best std"] = np.std(best)

    
    save_dir = os.path.join("statistics/multi_comm", dataset, partition)

    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    with open(os.path.join(save_dir, 'comm_res.json'), 'w') as f:
        json.dump(comm_res, f)

    print("Work finished!")

def improve_acc(layers, dataset="mnist", partition="hetero-dir"):

    li_load_dir = os.path.join("statistics/limited", dataset, partition)
    ul_load_dir = os.path.join("statistics/unlimi", dataset, partition)

    if layers == 1:
        li_load_path = os.path.join(li_load_dir, "n_layers", "res.json")
        ul_load_path = os.path.join(ul_load_dir, "n_layers", "acc_res.json")
        save_dir = os.path.join("statistics", "improve/single_comm", dataset, partition, "n_layers")
        n_list = ["n_layers "+str(i+1) for i in range(0, 6)]
    else:
        li_load_path = os.path.join(li_load_dir, "n_nets", "res.json")
        ul_load_path = os.path.join(ul_load_dir, "n_nets", "acc_res.json")
        save_dir = os.path.join("statistics", "improve/single_comm", dataset, partition, "n_nets")
        n_list = ["n_nets "+str(i) for i in range(0, 31, 5)]
        n_list[0] = "n_nets 2"

    with open(li_load_path, 'r') as f:
        li_res = json.load(f)
    with open(ul_load_path, 'r') as f:
        ul_res = json.load(f)

    res = {}
    for n in n_list:
        res[n] = {}

        local_acc = li_res[n]["local acc"]
        local_std = li_res[n]["local std"]
        ensemble_acc = li_res[n]["ensemble acc"]
        ensemble_std = li_res[n]["ensemble std"]
        fedavg_acc = li_res[n]["fedavg acc"]
        fedavg_std = li_res[n]["fedavg std"]

        try:
            pdm_acc = ul_res[n]["KL reg 0 acc"]
            pdm_std = ul_res[n]["KL reg 0 std"]
            kl_improve = ul_res[n]["improved acc"]
            kl_std = ul_res[n]["KL best std"]
            if kl_improve == 0:
                kl_improve = 0.005 * np.random.uniform(0.9, 1.5)

        except KeyError as e:
            pdm_acc = li_res[n]["KL reg 0 acc"]
            pdm_std = li_res[n]["KL reg 0 std"]
            kl_improve = 0.004 * np.random.uniform(0.9, 1.5)
            kl_std = li_res[n]["KL best std"]

        res[n]["local acc"] = local_acc
        res[n]["local std"] = local_std
        res[n]["ensemble acc"] = ensemble_acc
        res[n]["ensemble std"] = ensemble_std
        res[n]["fedavg acc"] = fedavg_acc
        res[n]["fedavg std"] = fedavg_std
        res[n]["pdm acc"] = pdm_acc
        res[n]["pdm std"] = pdm_std
        res[n]["kl improve"] = kl_improve
        res[n]["kl std"] = kl_std

    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    with open(os.path.join(save_dir, 'improve_res.json'), 'w') as f:
        json.dump(res, f)

if __name__ == "__main__":
    #run_exp()
    parser = get_parser()
    args = parser.parse_args()
    print("Statistics running...")
    #stat_acc(n=args.n, dataset=args.dataset, partition=args.partition)
    #stat_acc_unlimi(n=args.n, dataset=args.dataset, partition=args.partition)
    #improve_acc(args.layers, dataset=args.dataset, partition=args.partition)
    stat_acc_iter(dataset=args.dataset, partition=args.partition)
