import argparse
import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from itertools import product
import copy
from sklearn.metrics import confusion_matrix
import datetime

from utils import *
from datasets import MNIST_truncated, CIFAR10_truncated

from combine_nets import compute_ensemble_accuracy, compute_pdm_matching_multilayer, compute_iterative_pdm_matching, compute_fedavg_accuracy
from combine_nets import BBP_MAP, weights_prob_selfI_stats
import pdb
from ot_fusion import wasserstein_ensemble
from ot_fusion import parameters
ot_args = parameters.get_parameters()
ot_args.exact = True
ot_args.correction  = True
ot_args.weight_stats  = True
ot_args.activation_histograms  = True
ot_args.past_correction = True
ot_args.print_distances = True
# for the groud metric calculation
# ot_args.act_num_samples = 200
# ot_args.ground_metric_normalize = None
# ot_args.not_squared = True
# ot_args.dist_normalize = True
# for something I don't know yet
ot_args.activation_seed = 21
ot_args.prelu_acts = True
ot_args.recheck_acc = True

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--layers', type=int, required=True, help='do n_nets or n_layers')
    parser.add_argument('--n', nargs='+', type=int, required=True, help='the number of nets or layers')

    parser.add_argument('--loaddir', type=str, required=False, help='Load weights directory path')
    parser.add_argument('--logdir', type=str, required=False, help='Log directory path')
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--dataset', type=str, required=False, default="mnist", help="Dataset [mnist/cifar10]")
    parser.add_argument('--datadir', type=str, required=False, default="./data/mnist", help="Data directory")
    parser.add_argument('--init_seed', type=int, required=False, default=0, help="Random seed")

    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))

    parser.add_argument('--n_layers', type=int , required=False, default=1, help="Number of hidden layers")

    parser.add_argument('--n_nets', type=int , required=False, default=10, help="Number of nets to initialize")
    parser.add_argument('--model', type=str, required=False, default="fcnet", help="The model of which to train")
    parser.add_argument('--partition', type=str, required=False, help="Partition = homo/hetero/hetero-dir")
    parser.add_argument('--experiment', required=False, default="None", type=lambda s: s.split(','), help="Type of experiment to run. [none/w-ensemble/u-ensemble/pdm/all]")
    parser.add_argument('--trials', type=int, required=False, default=1, help="Number of trials for each run")
    
    parser.add_argument('--lr', type=float, required=False, default=0.01, help="Learning rate")
    parser.add_argument('--batch_size', type=int, required=False, default=32, help="Learning rate")
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Epochs")
    parser.add_argument('--reg', type=float, required=False, default=1e-6, help="L2 regularization strength")
    parser.add_argument('--retrain', type=bool, default=True, help="Do we need retrain the init weights?")

    parser.add_argument('--alpha', type=float, required=False, default=0.5, help="Dirichlet distribution constant used for data partitioning")

    parser.add_argument('--communication_rounds', type=int, required=False, default=None, help="How many iterations of PDM matching should be done")
    parser.add_argument('--lr_decay', type=float, required=False, default=1.0, help="Decay LR after every PDM iterative communication")
    parser.add_argument('--iter_epochs', type=int, required=False, default=5, help="Epochs for PDM-iterative method")
    parser.add_argument('--reg_fac', type=float, required=False, default=0.0, help="Regularization factor for PDM Iter")

    parser.add_argument('--pdm_sig', type=float, required=False, default=1.0, help="PDM sigma param")
    parser.add_argument('--pdm_sig0', type=float, required=False, default=1.0, help="PDM sigma0 param")
    parser.add_argument('--pdm_gamma', type=float, required=False, default=1.0, help="PDM gamma param")

    parser.add_argument('--device', type=str, required=False, default=1.0, help="Device to run")
    parser.add_argument('--num_pool_workers', type=int, required=True, help='the num of workers')
    
    return parser

def trans_next_conv_layer_forward(layer_weight, next_layer_shape):
    reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
    return reshaped

def trans_next_conv_layer_backward(layer_weight, next_layer_shape):
    reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
    reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
    return reshaped

def train_net(net_id, net, train_dataloader, test_dataloader, args, epochs, lr, reg, reg_base_weights=None,
              save_path=None, device="cpu"):

    logger.debug('Training network %s' % str(net_id))
    logger.debug('n_training: %d' % len(train_dataloader))
    logger.debug('n_test: %d' % len(test_dataloader))

    net.to(device)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.debug('>> Pre-Training Training accuracy: %f' % train_acc)
    logger.debug('>> Pre-Training Test accuracy: %f' % test_acc)

    if args.model == "fcnet":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                               lr=args.lr, weight_decay=0.0, amsgrad=True)      # L2_reg=0 because it's manually added later
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    losses, running_losses = [], []

    for epoch in range(args.epochs):
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            #pdb.set_trace()
            l2_reg = torch.zeros(1)
            l2_reg.requires_grad = True
            l2_reg = l2_reg.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            if reg_base_weights is None:
                # Apply standard L2-regularization
                for param in net.parameters():
                    l2_reg = l2_reg + 0.5 * torch.pow(param, 2).sum()
            else:
                # Apply Iterative PDM regularization
                for pname, param in net.named_parameters():
                    if "bias" in pname:
                        continue

                    layer_i = int(pname.split('.')[1])

                    if pname.split('.')[2] == "weight":
                        weight_i = layer_i * 2
                        transpose = True

                    ref_param = reg_base_weights[weight_i]
                    ref_param = ref_param.T if transpose else ref_param

                    l2_reg = l2_reg + 0.5 * torch.pow((param - torch.from_numpy(ref_param).float()), 2).sum()
            #pdb.set_trace()
            #l2_reg = (reg * l2_reg).to(device)
            loss = loss + args.reg * l2_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            losses.append(loss.item())

        logger.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))

        train_acc = compute_accuracy(net, train_dataloader, device=device)
        test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        return train_acc, test_acc

def local_train(nets, args, net_dataidx_map, device="cpu"):
    # save local dataset
    local_datasets = []
    local_train_accs = []
    local_test_accs = []

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        train_dl_local, test_dl_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        weight_path = os.path.join(args.weight_dir, "model " + str(net_id) + ".pkl")
        local_datasets.append((train_dl_local, test_dl_local))
        if args.retrain:
            train_acc, test_acc = train_net(net_id, net, train_dl_local, test_dl_local, args, args.epochs, args.lr,
                                            args.reg, save_path=None, device=device)
            # saving the trained models here
            with open(weight_path, "wb") as f_:
                torch.save(net.state_dict(), f_)
        else:
            net.load_state_dict(torch.load(weight_path))
            train_acc = compute_accuracy(net, train_dl_local, device=device)
            test_acc, conf_matrix = compute_accuracy(net, test_dl_local, get_confusion_matrix=True, device=device)

        local_train_accs.append(train_acc)
        local_test_accs.append(test_acc)

    nets_list = list(nets.values())
    return nets_list, local_train_accs, local_test_accs

def load_new_state(nets, new_weights):

    for netid, net in nets.items():

        statedict = net.state_dict()
        weights = new_weights[netid]

        # Load weight into the network
        i = 0
        layer_i = 0

        while i < len(weights):
            weight = weights[i]
            i += 1
            bias = weights[i]
            i += 1

            statedict['layers.%d.weight' % layer_i] = torch.from_numpy(weight.T)
            statedict['layers.%d.bias' % layer_i] = torch.from_numpy(bias)
            layer_i += 1

        net.load_state_dict(statedict)

    return nets

def run_exp(n):
    logger.info("Current n  is %d " % n)
    #args.n_nets = n_nets
    #args.logdir = os.path.join(logdir, "n_nets "+str(n_nets))
    #gpu_id = int(n_layer+2 % 4)
    #gpu_id = 3 if n%4 == 0 else 1
    gpu_id = 0
    device_str = "cuda:" + str(gpu_id)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("Current device is :", device)

    if args.layers == 1:
        if args.dataset == "mnist":
            args.net_config = list(map(int, ("784, "+"100, "*n+"10").split(', ')))
        elif args.dataset == "cifar10":
            args.net_config = list(map(int, ("3072, "+"100, "*n+"10").split(', ')))

        log_dir = os.path.join(args.logdir, "n_layers "+str(n))
        load_dir = os.path.join(args.loaddir, "n_layers "+str(n))
    else:
        args.n_nets = n
        log_dir = os.path.join(args.logdir, "n_nets "+str(n))
        load_dir = os.path.join(args.loaddir, "n_nets "+str(n))

    log_dir = os.path.join(log_dir, "_chaos")

    if not os.path.exists(log_dir):
        mkdirs(log_dir)

    filename = os.path.join(log_dir, 'experiment_log-%d-%d.log' % (args.init_seed, args.trials))

    logger.debug("Experiment arguments: %s" % str(args))

    trials_res = {}
    trials_res["Experiment arguments"] = str(args)

    logger.info("The total trials of n_nets %d is " % args.n_nets, args.trials)

    for trial in range(args.trials):
        #save_dir = os.path.join(log_dir, "trial "+str(trial))
        args.weight_dir = os.path.join(load_dir, "trial "+str(trial))
        if not os.path.exists(args.weight_dir):
            mkdirs(args.weight_dir)

        trials_res[trial] = {}
        seed = trial + args.init_seed
        trials_res[trial]['seed'] = seed
        logger.info("Executing Trial %d " % trial)
        logger.debug("#" * 100)
        logger.debug("Executing Trial %d with seed %d" % (trial, seed))

        np.random.seed(seed)
        torch.manual_seed(seed)

        logger.info("Partitioning data")
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
                        args.dataset, args.datadir, args.logdir, args.partition, args.n_nets, args.alpha)
        trials_res[trial]['Data statistics'] = str(traindata_cls_counts)
        n_classes = len(np.unique(y_train))

        averaging_weights = np.zeros((args.n_nets, n_classes), dtype=np.float32)

        for i in range(n_classes):
            total_num_counts = 0
            worker_class_counts = [0] * args.n_nets
            for j in range(args.n_nets):
                if i in traindata_cls_counts[j].keys():
                    total_num_counts += traindata_cls_counts[j][i]
                    worker_class_counts[j] = traindata_cls_counts[j][i]
                else:
                    total_num_counts += 0
                    worker_class_counts[j] = 0
            averaging_weights[:, i] = np.array(worker_class_counts) / total_num_counts

        logger.info("averaging_weights: {}".format(averaging_weights))

        logger.info("Initializing nets")
        nets, model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_nets, args)

        # local_train_accs = []
        # local_test_accs = []
        start = datetime.datetime.now()
        # for net_id, net in nets.items():
        #     dataidxs = net_dataidx_map[net_id]
        #     logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        #
        #     #save_path = os.path.join(save_dir, "model "+str(net_id)+".pkl")
        #     weight_path = os.path.join(weight_dir, "model "+str(net_id)+".pkl")
        #     train_dl, test_dl = get_dataloader(args.dataset, args.datadir, 32, 32, dataidxs)
        #
        #     net.load_state_dict(torch.load(weight_path))
        #     net.to(device)
        #     #trainacc, testacc = train_net(net_id, net, train_dl, test_dl, args.epochs, args.lr, args.reg,
        #     #                             save_path=save_path, device=device)
        #
        #     train_acc = compute_accuracy(net, train_dl, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dl, get_confusion_matrix=True, device=device)
        #
        #
        #     local_train_accs.append(train_acc)
        #     local_test_accs.append(test_acc)
        ### local training stage
        nets_list, local_train_accs, local_test_accs = local_train(nets, args, net_dataidx_map, device=device)

        end = datetime.datetime.now()
        timing = (end - start).seconds
        trials_res[trial]['loading time'] = timing

        trials_res[trial]['local_train_accs'] = local_train_accs
        trials_res[trial]['local_test_accs'] = local_test_accs
        train_dl, test_dl = get_dataloader(args.dataset, args.datadir, 32, 32)

        logger.debug("*"*50)
        logger.debug("Running experiments \n")

        nets_list = list(nets.values())

        if ("u-ensemble" in args.experiment) or ("all" in args.experiment):
            logger.info("Computing Uniform ensemble accuracy")
            uens_train_acc, _ = compute_ensemble_accuracy(nets_list, train_dl, n_classes,  uniform_weights=True, device=device)
            uens_test_acc, _ = compute_ensemble_accuracy(nets_list, test_dl, n_classes, uniform_weights=True, device=device)

            logger.debug("Uniform ensemble (Train acc): %f" % uens_train_acc)
            logger.debug("Uniform ensemble (Test acc): %f" % uens_test_acc)

            trials_res[trial]["Uniform ensemble (Train acc)"] = uens_train_acc
            trials_res[trial]["Uniform ensemble (Test acc)"] = uens_test_acc

        if ("FedAvg" in args.experiment) or ("all" in args.experiment):
            logger.info("Computing FedAvg accuracy")
            avg_train_acc, avg_test_acc, _, _ = compute_fedavg_accuracy(nets_list, train_dl, test_dl, traindata_cls_counts, n_classes, device=device)

            logger.debug("FedAvg (Train acc): %f" % avg_train_acc)
            logger.debug("FedAvg (Test acc): %f" % avg_test_acc)

            trials_res[trial]["FedAvg (Train acc)"] = avg_train_acc
            trials_res[trial]["FedAvg (Test acc)"] = avg_test_acc

        if ("pdm_KL" in args.experiment) or ("all" in args.experiment):
            logger.info("Computing hungarian matching")
            start = datetime.datetime.now()

            KL_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

            best_reg = 0
            comp = 0
          
            for KL_reg in KL_regs:
                start = datetime.datetime.now()

                trials_res[trial]["pdm_KL_reg "+str(KL_reg)] = {}

                if args.model == "fcnet":
                    res = compute_pdm_matching_multilayer(
                        nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1], it=args.iter_epochs,
                        sigma=args.pdm_sig,
                        sigma0=args.pdm_sig0, gamma=args.pdm_gamma, device=device, KL_reg=KL_reg, unlimi=True
                    )
                else:
                    res = BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map,
                                  traindata_cls_counts, averaging_weights, args, args.net_config[-1], it=args.iter_epochs,
                                  sigma=args.pdm_sig, sigma0=args.pdm_sig0, gamma=args.pdm_gamma,
                                  device=device, KL_reg=KL_reg, unlimi=True)

                end = datetime.datetime.now()
                timing = (end - start).seconds

                weights = res['weights']
                logger.debug("****** PDM_KL matching ******** ")
                logger.debug("Best Sigma0: %s. Best sigma: %s Best gamma: %s. Best Test accuracy: %s. Train acc: %s. \n"
                          % (str(res['sigma0']), str(res['sigma']), str(res['gamma']), str(res['test_accuracy']), str(res['train_accuracy'])))
                logger.debug("PDM_KL log: %s " % str(res))

                trials_res[trial]["pdm_KL_reg "+str(KL_reg)]["Best Result"] = {"Best Sigma0": res['sigma0'], "Best sigma": res['sigma'],
                                                         "Best gamma": res['gamma'], "Best Test accuracy": res['test_accuracy'],
                                                         "Train acc": res['train_accuracy']}
                trials_res[trial]["pdm_KL_reg "+str(KL_reg)]["shape"] = res['shapes']
                trials_res[trial]["pdm_KL_reg "+str(KL_reg)]["PDM log"] = str(res)

                stats_layers = weights_prob_selfI_stats(weights, layer_type, res['sigma0'], args)
                prob_layers = stats_layers['probability']
                selfI_layers = stats_layers['self information']
                trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['prob mean layers'] = np.mean(prob_layers, axis=1)
                trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['prob std layers'] = np.std(prob_layers, axis=1)
                trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['selfI mean layers'] = np.mean(selfI_layers, axis=1)
                trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['selfI std layers'] = np.std(selfI_layers, axis=1)

                best_test_acc = res['test_accuracy']
                if best_test_acc > comp:
                    best_reg = KL_reg
                    comp = best_test_acc

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["pdm_KL_reg "+str(KL_reg)]["timing"] = timing

            trials_res[trial]["best_reg"] = best_reg

        if ("SPAHM_KL" in args.experiment) or ("all" in args.experiment):
            logger.info("Computing hungarian matching")
            start = datetime.datetime.now()

            KL_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

            best_reg = 0
            comp = 0

            for KL_reg in KL_regs:
                start = datetime.datetime.now()

                trials_res[trial]["SPAHM_KL_reg " + str(KL_reg)] = {}

                if args.model == "fcnet":
                    res = compute_pdm_matching_multilayer(
                        nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1], it=args.iter_epochs,
                        sigma=args.pdm_sig,
                        sigma0=args.pdm_sig0, gamma=args.pdm_gamma, device=device, KL_reg=KL_reg, unlimi=True,
                        SPAHM=True
                    )
                else:
                    res = BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map,
                                  traindata_cls_counts, averaging_weights, args, args.net_config[-1],
                                  it=args.iter_epochs,
                                  sigma=args.pdm_sig, sigma0=args.pdm_sig0, gamma=args.pdm_gamma,
                                  device=device, KL_reg=KL_reg, unlimi=True)

                end = datetime.datetime.now()
                timing = (end - start).seconds

                weights = res['weights']
                logger.debug("****** SPAHM_KL matching ******** ")
                logger.debug("Best Sigma0: %s. Best sigma: %s Best gamma: %s. Best Test accuracy: %s. Train acc: %s. \n"
                             % (str(res['sigma0']), str(res['sigma']), str(res['gamma']), str(res['test_accuracy']),
                                str(res['train_accuracy'])))
                logger.debug("SPAHM_KL log: %s " % str(res))

                trials_res[trial]["SPAHM_KL_reg " + str(KL_reg)]["Best Result"] = {"Best Sigma0": res['sigma0'],
                                                                                 "Best sigma": res['sigma'],
                                                                                 "Best gamma": res['gamma'],
                                                                                 "Best Test accuracy": res[
                                                                                     'test_accuracy'],
                                                                                 "Train acc": res['train_accuracy']}
                trials_res[trial]["SPAHM_KL_reg " + str(KL_reg)]["shape"] = res['shapes']
                trials_res[trial]["SPAHM_KL_reg " + str(KL_reg)]["SPAHM log"] = str(res)

                # stats_layers = weights_prob_selfI_stats(weights, layer_type, res['sigma0'], args)
                # prob_layers = stats_layers['probability']
                # selfI_layers = stats_layers['self information']
                # trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['prob mean layers'] = np.mean(prob_layers, axis=1)
                # trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['prob std layers'] = np.std(prob_layers, axis=1)
                # trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['selfI mean layers'] = np.mean(selfI_layers, axis=1)
                # trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['selfI std layers'] = np.std(selfI_layers, axis=1)

                best_test_acc = res['test_accuracy']
                if best_test_acc > comp:
                    best_reg = KL_reg
                    comp = best_test_acc

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["SPAHM_KL_reg " + str(KL_reg)]["timing"] = timing

            trials_res[trial]["best_reg"] = best_reg

        if ("nafi_KL" in args.experiment) or ("all" in args.experiment):
            logger.info("Computing hungarian matching")
            start = datetime.datetime.now()

            # KL_regs = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

            KL_regs = [0, 0.001, 2, 0.005, 0.0001, 0.05, 0.1, 0.01, 0.5, 1, 5, 10]

            best_reg = 0
            comp = 0

            for KL_reg in KL_regs:
                start = datetime.datetime.now()

                trials_res[trial]["nafi_KL_reg " + str(KL_reg)] = {}

                if args.model == "fcnet":
                    res = compute_pdm_matching_multilayer(
                        nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1], it=args.iter_epochs,
                        sigma=args.pdm_sig,
                        sigma0=args.pdm_sig0, gamma=args.pdm_gamma, device=device, KL_reg=KL_reg, unlimi=True,
                        SPAHM=False, nafi=True
                    )
                else:
                    res = BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map,
                                  traindata_cls_counts, averaging_weights, args, args.net_config[-1],
                                  it=args.iter_epochs,
                                  sigma=args.pdm_sig, sigma0=args.pdm_sig0, gamma=args.pdm_gamma,
                                  device=device, KL_reg=KL_reg, unlimi=True)

                end = datetime.datetime.now()
                timing = (end - start).seconds

                weights = res['weights']
                logger.debug("****** nafi_KL matching ******** ")
                logger.debug("Best Sigma0: %s. Best sigma: %s Best gamma: %s. Best Test accuracy: %s. Train acc: %s. \n"
                             % (str(res['sigma0']), str(res['sigma']), str(res['gamma']), str(res['test_accuracy']),
                                str(res['train_accuracy'])))
                logger.debug("nafi_KL log: %s " % str(res))

                trials_res[trial]["nafi_KL_reg " + str(KL_reg)]["Best Result"] = {"Best Sigma0": res['sigma0'],
                                                                                 "Best sigma": res['sigma'],
                                                                                 "Best gamma": res['gamma'],
                                                                                 "Best Test accuracy": res[
                                                                                     'test_accuracy'],
                                                                                 "Train acc": res['train_accuracy']}
                trials_res[trial]["nafi_KL_reg " + str(KL_reg)]["shape"] = res['shapes']
                trials_res[trial]["nafi_KL_reg " + str(KL_reg)]["nafi log"] = str(res)

                # stats_layers = weights_prob_selfI_stats(weights, layer_type, res['sigma0'], args)
                # prob_layers = stats_layers['probability']
                # selfI_layers = stats_layers['self information']
                # trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['prob mean layers'] = np.mean(prob_layers, axis=1)
                # trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['prob std layers'] = np.std(prob_layers, axis=1)
                # trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['selfI mean layers'] = np.mean(selfI_layers, axis=1)
                # trials_res[trial]["pdm_KL_reg " + str(KL_reg)]['selfI std layers'] = np.std(selfI_layers, axis=1)

                best_test_acc = res['test_accuracy']
                if best_test_acc > comp:
                    best_reg = KL_reg
                    comp = best_test_acc

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["nafi_KL_reg " + str(KL_reg)]["timing"] = timing

            trials_res[trial]["best_reg"] = best_reg

        if ("pdm_I" in args.experiment) or ("all" in args.experiment):
            
            if trial == 0:
                log_dir = os.path.join(log_dir, 'self_info_cost1')
                if not os.path.exists(log_dir):
                    mkdirs(log_dir)

            logger.info("Computing hungarian matching")
            start = datetime.datetime.now()

            #I_regs = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            I_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

            best_reg = 0
            comp = 0
          
            for I_reg in I_regs:
                start = datetime.datetime.now()

                trials_res[trial]["pdm_I_reg "+str(I_reg)] = {}

                best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, best_assignments, res = compute_pdm_matching_multilayer(
                    nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1], it=args.iter_epochs, sigma=args.pdm_sig, 
                    sigma0=args.pdm_sig0, gamma=args.pdm_gamma, device=device, I_reg=I_reg, unlimi=True
                )
                end = datetime.datetime.now()
                timing = (end - start).seconds
                logger.debug("****** PDM_I matching ******** ")
                logger.debug("Best Sigma0: %s. Best sigma: %s Best gamma: %s. Best Test accuracy: %s. Train acc: %s. \n"
                          % (str(best_sigma0), str(best_sigma), str(best_gamma), str(best_test_acc), str(best_train_acc)))

                logger.debug("PDM_I log: %s " % str(res))

                trials_res[trial]["pdm_I_reg "+str(I_reg)]["Best Result"] = {"Best Sigma0": best_sigma0, "Best sigma": best_sigma, 
                                                         "Best gamma": best_gamma, "Best Test accuracy": best_test_acc,
                                                         "Train acc": best_train_acc}
                trials_res[trial]["pdm_I_reg "+str(I_reg)]["PDM log"] = str(res)
                #trials_res[trial]["pdm_I_reg "+str(I_reg)]["Assignments"] = best_assignments
                
                if best_test_acc > comp:
                    best_reg = I_reg
                    comp = best_test_acc

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["pdm_I_reg "+str(I_reg)]["timing"] = timing

                logger.info("Trial %d I_reg %f finished!!!" % (trial, I_reg))

            trials_res[trial]["best_reg"] = best_reg

        if ("pdm_neg_I" in args.experiment) or ("all" in args.experiment):
            
            if trial == 0:
                log_dir = os.path.join(log_dir, 'self_info_neg_cost')
                if not os.path.exists(log_dir):
                    mkdirs(log_dir)

            logger.info("Computing hungarian matching")
            start = datetime.datetime.now()

            #I_regs = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
            I_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

            best_reg = 0
            comp = 0
          
            for I_reg in I_regs:
                start = datetime.datetime.now()

                trials_res[trial]["pdm_neg_I_reg "+str(I_reg)] = {}

                best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, best_assignments, res = compute_pdm_matching_multilayer(
                    nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1], it=args.iter_epochs, sigma=args.pdm_sig, 
                    sigma0=args.pdm_sig0, gamma=args.pdm_gamma, device=device, I_reg=-I_reg, unlimi=True
                )
                end = datetime.datetime.now()
                timing = (end - start).seconds
                logger.debug("****** PDM_neg_I matching ******** ")
                logger.debug("Best Sigma0: %s. Best sigma: %s Best gamma: %s. Best Test accuracy: %s. Train acc: %s. \n"
                          % (str(best_sigma0), str(best_sigma), str(best_gamma), str(best_test_acc), str(best_train_acc)))

                logger.debug("PDM_neg_I log: %s " % str(res))

                trials_res[trial]["pdm_neg_I_reg "+str(I_reg)]["Best Result"] = {"Best Sigma0": best_sigma0, "Best sigma": best_sigma, 
                                                         "Best gamma": best_gamma, "Best Test accuracy": best_test_acc,
                                                         "Train acc": best_train_acc}
                trials_res[trial]["pdm_neg_I_reg "+str(I_reg)]["PDM log"] = str(res)
                #trials_res[trial]["pdm_I_reg "+str(I_reg)]["Assignments"] = best_assignments
                
                if best_test_acc > comp:
                    best_reg = I_reg
                    comp = best_test_acc

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["pdm_neg_I_reg "+str(I_reg)]["timing"] = timing

                logger.info("Trial %d I_neg_reg %f finished!!!" % (trial, I_reg))

            trials_res[trial]["best_reg"] = best_reg

        if ("pdm_coff" in args.experiment) or ("all" in args.experiment):

            if trial == 0:
                log_dir = os.path.join(log_dir, 'coff_cost')
                if not os.path.exists(log_dir):
                    mkdirs(log_dir)

            logger.info("Computing hungarian matching")
            start = datetime.datetime.now()

            coffs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

            best_coff = 0
            comp = 0
          
            for coff in coffs:
                start = datetime.datetime.now()

                trials_res[trial]["pdm_coff 1-"+str(coff)] = {}

                best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, best_assignments, res = compute_pdm_matching_multilayer(
                    nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1], it=args.iter_epochs, sigma=args.pdm_sig, 
                    sigma0=args.pdm_sig0, gamma=args.pdm_gamma, device=device, coff=1-coff, unlimi=True
                )
                end = datetime.datetime.now()
                timing = (end - start).seconds
                logger.debug("****** PDM_coff matching ******** ")
                logger.debug("Best Sigma0: %s. Best sigma: %s Best gamma: %s. Best Test accuracy: %s. Train acc: %s. \n"
                          % (str(best_sigma0), str(best_sigma), str(best_gamma), str(best_test_acc), str(best_train_acc)))

                logger.debug("PDM_coff log: %s " % str(res))

                trials_res[trial]["pdm_coff 1-"+str(coff)]["Best Result"] = {"Best Sigma0": best_sigma0, "Best sigma": best_sigma, 
                                                         "Best gamma": best_gamma, "Best Test accuracy": best_test_acc,
                                                         "Train acc": best_train_acc}
                trials_res[trial]["pdm_coff 1-"+str(coff)]["PDM log"] = str(res)
                
                if best_test_acc > comp:
                    best_coff = coff
                    comp = best_test_acc

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["pdm_coff 1-"+str(coff)]["timing"] = timing

                logger.info("Trial %d coff %f finished!!!" % (trial, coff))

            trials_res[trial]["best_coff"] = str(best_coff)

        if ("pdm_fix" in args.experiment) or ("all" in args.experiment):

            if trial == 0:
                log_dir = os.path.join(log_dir, 'fix_cost')
                if not os.path.exists(log_dir):
                    mkdirs(log_dir)

            logger.info("Computing hungarian matching")
            start = datetime.datetime.now()

            coffs = [0, 1]

            best_coff = 0
            comp = 0
          
            for coff in coffs:
                start = datetime.datetime.now()

                trials_res[trial]["pdm_fix "+str(coff)] = {}

                best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, best_assignments, res = compute_pdm_matching_multilayer(
                    nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1], it=args.iter_epochs, sigma=args.pdm_sig, 
                    sigma0=args.pdm_sig0, gamma=args.pdm_gamma, device=device, fix_coff=coff, unlimi=True
                )
                end = datetime.datetime.now()
                timing = (end - start).seconds
                logger.debug("****** PDM_fix matching ******** ")
                logger.debug("Best Sigma0: %s. Best sigma: %s Best gamma: %s. Best Test accuracy: %s. Train acc: %s. \n"
                          % (str(best_sigma0), str(best_sigma), str(best_gamma), str(best_test_acc), str(best_train_acc)))

                logger.debug("PDM_fix log: %s " % str(res))

                trials_res[trial]["pdm_fix "+str(coff)]["Best Result"] = {"Best Sigma0": best_sigma0, "Best sigma": best_sigma, 
                                                         "Best gamma": best_gamma, "Best Test accuracy": best_test_acc,
                                                         "Train acc": best_train_acc}
                trials_res[trial]["pdm_fix "+str(coff)]["PDM log"] = str(res)
                
                if best_test_acc > comp:
                    best_coff = coff
                    comp = best_test_acc

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["pdm_fix "+str(coff)]["timing"] = timing

                logger.info("Trial %d coff %f finished!!!" % (trial, coff))

            trials_res[trial]["best_coff"] = str(best_coff)

        if ("pdm_iterative" in args.experiment) or ("all" in args.experiment):
            logger.info("Running Iterative PDM matching procedure")
            logger.debug("Running Iterative PDM matching procedure")

            for KL_reg in KL_regs:
                logger.debug("Parameter setting: sigma0 = %f, sigma = %f, gamma = %f" % (args.pdm_sig0, args.pdm_sig, args.pdm_gamma))

                iter_nets = copy.deepcopy(nets)
                assignment = None
                lr_iter = args.lr
                reg_iter = args.reg

                # Run for communication rounds iterations
                for i, comm_round in enumerate(range(args.communication_rounds)):

                    it = 3

                    iter_nets_list = list(iter_nets.values())

                    net_weights_new, train_acc, test_acc, new_shape, assignment, hungarian_weights, \
                    conf_matrix_train, conf_matrix_test = compute_iterative_pdm_matching(
                        iter_nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1],
                        sigma, sigma0, gamma, it, old_assignment=assignment, device=device, KL_reg=0
                    )

                    logger.debug("Communication: %d, Train acc: %f, Test acc: %f, Shapes: %s" % (comm_round, train_acc, test_acc, str(new_shape)))
                    logger.debug('CENTRAL MODEL CONFUSION MATRIX')
                    logger.debug('Train data confusion matrix: \n %s' % str(conf_matrix_train))
                    logger.debug('Test data confusion matrix: \n %s' % str(conf_matrix_test))

                    iter_nets = load_new_state(iter_nets, net_weights_new)

                    expepochs = args.iter_epochs if args.iter_epochs is not None else args.epochs

                    # Train these networks again
                    for net_id, net in iter_nets.items():
                        dataidxs = net_dataidx_map[net_id]
                        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

                        net_train_dl, net_test_dl = get_dataloader(args.dataset, args.datadir, 32, 32, dataidxs)
                        train_net(net_id, net, net_train_dl, net_test_dl, expepochs, lr_iter, reg_iter, net_weights_new[net_id],
                                device=device)

                    lr_iter *= args.lr_decay
                    reg_iter *= args.reg_fac

        if ("ot_fusion" in args.experiment) or ("all" in args.experiment):
            logger.info("Computing ot fusion")
            start = datetime.datetime.now()

            # KL_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

            best_reg = 0
            comp = 0

            ## the net to aligned with
            aligned_net = nets_list[0]

            for i in range(1,len(nets_list)):
                fusion_models = [nets_list[i], aligned_net]
                ot_args.input_dim = 784
                ot_args.hidden_dims = [100]
                ot_args.output_dim = 10
                ot_args.model_name = 'fcnet'
                geometric_acc, aligned_net = wasserstein_ensemble.geometric_ensembling_modularized(ot_args, fusion_models,
                                                                                                       train_dl, test_dl)

            trials_res[trial]["ot_fusion_acc"] = geometric_acc


        logger.debug("Trial %d completed" % trial)
        logger.debug("#"*100)

        with open(os.path.join(log_dir, 'trial'+str(trial)+'.json'), 'w') as f:
            json.dump(trials_res[trial], f) 

    with open(os.path.join(log_dir, 'trials_res.json'), 'w') as f:
        json.dump(trials_res, f)

    return trials_res

abli_res = {}

from multiprocessing import Pool
import contextlib

z = lambda x: list(map(int, x.split(',')))
def abli_exp(n=[10], dataset="mnist"):

    #if n_nets > 10:
    #    n_nets_list = [i for i in range(15, n_nets+1, 5)]
    #else:
    #    n_nets_list = [i for i in range(0, n_nets+1, 5)]
    #    n_nets_list[0] += 2
    #if n_nets > 30:
    #    n_nets_list = [i for i in range(n_nets, 30+1, 5)]
    #else:
    #    n_nets_list = [i for i in range(0, n_nets+1, 5)]
    #    n_nets_list[0] = 2
    #    n_layers_list = [i+1 for i in range(n_layers)]
    # partitions = ["hetero-dir", "homo"]
    # partitions = ["homo"]
    partitions = ["hetero-dir"]
    #n_layer_list = [i+2 for i in range(n_layers-1)] # don't need layer_num 1
    if args.layers == 1:
        #n_list = [i+1 for i in range(n)]
        n_list = n
    else:
        #n_list = [i for i in range(0, n+1, 5)]
        #n_list[0] = 2
        n_list = n

    #args.experiments = "u-ensemble,pdm,pdm_KL"
    args.dataset = dataset
    args.datadir = os.path.join("data", dataset)
    #if dataset == "mnist":
        #args.net_config = "784, 100, 10"
    #else:
        #args.net_config = "3071, 100, 10"
    
    #args.epochs = 10
    #args.reg = 1e-6
    #args.lr_decay = 0.99
    #args.iter_epochs = 5
    #args.device = torch.device(device if torch.cuda.is_available() else "cpu")

    #now = datetime.datetime.now()
    #KL_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

    for partition in partitions:
        #logdir = os.path.join('log_abli', now.strftime("%Y-%m-%d %H"),
        #                      dataset, partition)
        # logdir = os.path.join('KL_regs_bst_hyp_unlimi_2',
        #                       dataset, partition)
        logdir = os.path.join('KL_regs_bst_hyp_unlimi', dataset, partition, args.experiment[0])
        args.loaddir = os.path.join('saved_weights', dataset, partition)
        args.logdir = logdir
        args.partition = partition
        logger.info("Partition type is ", partition)
        abli_res = {}
        
        # set the sig, sig0, gamma
        if (dataset == "mnist" or dataset == "fashionmnist") and partition == "homo":
            args.pdm_sig0 = 10
            args.pdm_sig = 0.5
            args.pdm_gamma = 10
        elif (dataset == "mnist" or dataset == "fashionmnist") and partition == "hetero-dir":
            args.pdm_sig0 = 3
            args.pdm_sig = 0.9
            args.pdm_gamma = 1
        elif dataset == "cifar10" and partition == "homo":
            args.pdm_sig0 = 10
            args.pdm_sig = 1
            args.pdm_gamma = 10
        elif dataset == "cifar10" and partition == "hetero-dir":
            args.pdm_sig0 = 10
            args.pdm_sig = 1
            args.pdm_gamma = 50

        with contextlib.closing(Pool(args.num_pool_workers)) as po:
            # pool_results = po.map_async(run_exp, (n for n in n_list))
            # results_list = pool_results.get()
            run_exp(n_list[0])

        for n, result in zip(n_list, results_list):
            abli_res[n] = result

        with open(os.path.join(logdir, 'abliation_experiment.json'), 'w') as f:
            json.dump(abli_res, f)


if __name__ == "__main__":
    #run_exp()
    parser = get_parser()
    args = parser.parse_args()
    logger.info("Abliation experiment running...")
    abli_exp(n=args.n, dataset=args.dataset)
