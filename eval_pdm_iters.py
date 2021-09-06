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

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--layers', type=int, required=True, help='do n_nets or n_layers')
    parser.add_argument('--n', type=int, required=True, help='the number of nets or layers')

    parser.add_argument('--loaddir', type=str, required=False, help='Load weights directory path')
    parser.add_argument('--logdir', type=str, required=False, help='Log directory path')
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--dataset', type=str, required=False, default="mnist", help="Dataset [mnist/cifar10]")
    parser.add_argument('--datadir', type=str, required=False, default="./data/mnist", help="Data directory")
    parser.add_argument('--init_seed', type=int, required=False, default=0, help="Random seed")

    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))

    parser.add_argument('--n_layers', type=int , required=False, default=1, help="Number of hidden layers")

    parser.add_argument('--n_nets', type=int , required=False, default=10, help="Number of nets to initialize")
    parser.add_argument('--partition', type=str, required=False, help="Partition = homo/hetero/hetero-dir")
    parser.add_argument('--experiment', required=False, default="None", type=lambda s: s.split(','), help="Type of experiment to run. [none/w-ensemble/u-ensemble/pdm/all]")
    parser.add_argument('--trials', type=int, required=False, default=1, help="Number of trials for each run")
    
    parser.add_argument('--lr', type=float, required=False, default=0.01, help="Learning rate")
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Epochs")
    parser.add_argument('--reg', type=float, required=False, default=1e-6, help="L2 regularization strength")

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

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def parse_class_dist(net_class_config):

    cls_net_map = {}

    for net_idx, net_classes in enumerate(net_class_config):
        for net_cls in net_classes:
            if net_cls not in cls_net_map:
                cls_net_map[net_cls] = []
            cls_net_map[net_cls].append(net_idx)

    return cls_net_map

def record_net_data_stats(y_train, net_dataidx_map, logdir):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {int(unq[i]): unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logging.debug('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


def partition_data(dataset, datadir, logdir, partition, n_nets, alpha=0.5):

    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist_data(datadir)
    elif dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    
    n_train = X_train.shape[0]

    if partition == "homo":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

    elif partition == "hetero-dir":
        min_size = 0
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < 10:
            idx_batch = [[] for _ in range(n_nets)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p*(len(idx_j)<N/n_nets) for p,idx_j in zip(proportions,idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def init_nets(net_configs, dropout_p, n_nets):

    input_size = net_configs[0]
    output_size = net_configs[-1]
    hidden_sizes = net_configs[1:-1]

    nets = {net_i: None for net_i in range(n_nets)}

    for net_i in range(n_nets):
        net = FcNet(input_size, hidden_sizes, output_size, dropout_p)

        nets[net_i] = net

    return nets

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):

    if dataset == 'mnist':
        dl_obj = MNIST_truncated
    elif dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

    transform = transforms.Compose([transforms.ToTensor()])

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform, download=True)

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl


def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device)

            out = model(x)
            _, pred_label = torch.max(out.data, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, reg, reg_base_weights=None,
              save_path=None, device="cpu"):

    logging.debug('Training network %s' % str(net_id))
    logging.debug('n_training: %d' % len(train_dataloader))
    logging.debug('n_test: %d' % len(test_dataloader))

    net.to(device)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logging.debug('>> Pre-Training Training accuracy: %f' % train_acc)
    logging.debug('>> Pre-Training Test accuracy: %f' % test_acc)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0, amsgrad=True)      # L2_reg=0 because it's manually added later

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    losses, running_losses = [], []

    for epoch in range(epochs):
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
            loss = loss + reg * l2_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            losses.append(loss.item())

        logging.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))

    if save_path:
        torch.save(net.state_dict(), save_path)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logging.debug('>> Training accuracy: %f' % train_acc)
    logging.debug('>> Test accuracy: %f' % test_acc)

    logging.debug(' ** Training complete **')

    return train_acc, test_acc


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
    print("Current n  is %d " % args.n)
    KL_regs = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]
    pdm_iters = [5, 10, 15, 20]
    #args.n_nets = n_nets
    #args.logdir = os.path.join(logdir, "n_nets "+str(n_nets))
    #gpu_id = int(n_layer+2 % 4)
    #gpu_id = 3 if n%4 == 0 else 1
    gpu_id = 2
    device_str = "cuda:" + str(gpu_id)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Current device is :", device)

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

    if not os.path.exists(log_dir):
        mkdirs(log_dir)
    #with open(os.path.join(log_dir, 'experiment_arguments.json'), 'w') as f:
        #json.dump(str(args), f)
    #print("the log_dir is", args.logdir)
    filename = os.path.join(log_dir, 'experiment_log-%d-%d.log' % (args.init_seed, args.trials))
    #print("the log filename is", filename)
    logging.basicConfig(
        filename=filename,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logging.debug("Experiment arguments: %s" % str(args))

    trials_res = {}
    trials_res["Experiment arguments"] = str(args)

    print("The total trials of n_nets %d is " % args.n_nets, args.trials)

    for trial in range(args.trials):
        #save_dir = os.path.join(log_dir, "trial "+str(trial))
        weight_dir = os.path.join(load_dir, "trial "+str(trial))

        trials_res[trial] = {}
        seed = trial + args.init_seed
        trials_res[trial]['seed'] = seed
        print("Executing Trial %d " % trial)
        logging.debug("#" * 100)
        logging.debug("Executing Trial %d with seed %d" % (trial, seed))

        np.random.seed(seed)
        torch.manual_seed(seed)

        print("Partitioning data")
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
                        args.dataset, args.datadir, args.logdir, args.partition, args.n_nets, args.alpha)
        trials_res[trial]['Data statistics'] = str(traindata_cls_counts)
        n_classes = len(np.unique(y_train))

        print("Initializing nets")
        nets = init_nets(args.net_config, args.dropout_p, args.n_nets)

        local_train_accs = []
        local_test_accs = []
        start = datetime.datetime.now()
        for net_id, net in nets.items():
            dataidxs = net_dataidx_map[net_id]
            print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

            #save_path = os.path.join(save_dir, "model "+str(net_id)+".pkl")
            weight_path = os.path.join(weight_dir, "model "+str(net_id)+".pkl")
            train_dl, test_dl = get_dataloader(args.dataset, args.datadir, 32, 32, dataidxs)
            
            net.load_state_dict(torch.load(weight_path))
            net.to(device)
            #trainacc, testacc = train_net(net_id, net, train_dl, test_dl, args.epochs, args.lr, args.reg, 
            #                             save_path=save_path, device=device)

            train_acc = compute_accuracy(net, train_dl, device=device)
            test_acc, conf_matrix = compute_accuracy(net, test_dl, get_confusion_matrix=True, device=device)


            local_train_accs.append(train_acc)
            local_test_accs.append(test_acc)

        end = datetime.datetime.now()
        timing = (end - start).seconds
        trials_res[trial]['loading time'] = timing

        trials_res[trial]['local_train_accs'] = local_train_accs
        trials_res[trial]['local_test_accs'] = local_test_accs
        train_dl, test_dl = get_dataloader(args.dataset, args.datadir, 32, 32)

        logging.debug("*"*50)
        logging.debug("Running experiments \n")

        nets_list = list(nets.values())

        if ("u-ensemble" in args.experiment) or ("all" in args.experiment):
            print("Computing Uniform ensemble accuracy")
            uens_train_acc, _ = compute_ensemble_accuracy(nets_list, train_dl, n_classes,  uniform_weights=True, device=device)
            uens_test_acc, _ = compute_ensemble_accuracy(nets_list, test_dl, n_classes, uniform_weights=True, device=device)

            logging.debug("Uniform ensemble (Train acc): %f" % uens_train_acc)
            logging.debug("Uniform ensemble (Test acc): %f" % uens_test_acc)

            trials_res[trial]["Uniform ensemble (Train acc)"] = uens_train_acc
            trials_res[trial]["Uniform ensemble (Test acc)"] = uens_test_acc

        if ("FedAvg" in args.experiment) or ("all" in args.experiment):
            print("Computing FedAvg accuracy")
            avg_train_acc, avg_test_acc, _, _ = compute_fedavg_accuracy(nets_list, train_dl, test_dl, traindata_cls_counts, n_classes, device=device)

            logging.debug("FedAvg (Train acc): %f" % avg_train_acc)
            logging.debug("FedAvg (Test acc): %f" % avg_test_acc)

            trials_res[trial]["FedAvg (Train acc)"] = avg_train_acc
            trials_res[trial]["FedAvg (Test acc)"] = avg_test_acc

        if ("pdm_KL" in args.experiment) or ("all" in args.experiment):
            print("Computing hungarian matching")
            start = datetime.datetime.now()
          
            for pdm_it in pdm_iters:
                start = datetime.datetime.now()

                trials_res[trial]["pdm_pdm_it "+str(pdm_it)] = {}

                best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, res = compute_pdm_matching_multilayer(
                    nets_list, train_dl, test_dl, traindata_cls_counts, args.net_config[-1], it=pdm_it, sigma=args.pdm_sig, 
                    sigma0=args.pdm_sig0, gamma=args.pdm_gamma, device=device, KL_reg=0, unlimi=True
                )
                end = datetime.datetime.now()
                timing = (end - start).seconds
                logging.debug("****** PDM_KL matching ******** ")
                logging.debug("Best Sigma0: %s. Best sigma: %s Best gamma: %s. Best Test accuracy: %s. Train acc: %s. \n"
                          % (str(best_sigma0), str(best_sigma), str(best_gamma), str(best_test_acc), str(best_train_acc)))

                logging.debug("PDM_KL log: %s " % str(res))

                trials_res[trial]["pdm_pdm_it "+str(pdm_it)]["Best Result"] = {"Best Sigma0": best_sigma0, "Best sigma": best_sigma, 
                                                         "Best gamma": best_gamma, "Best Test accuracy": best_test_acc,
                                                         "Train acc": best_train_acc}
                trials_res[trial]["pdm_pdm_it "+str(pdm_it)]["PDM log"] = str(res)

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["pdm_pdm_it "+str(pdm_it)]["timing"] = timing

        if ("pdm_iterative" in args.experiment) or ("all" in args.experiment):
            print("Running Iterative PDM matching procedure")
            logging.debug("Running Iterative PDM matching procedure")

            for KL_reg in KL_regs:
                logging.debug("Parameter setting: sigma0 = %f, sigma = %f, gamma = %f" % (args.pdm_sig0, args.pdm_sig, args.pdm_gamma))

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

                    logging.debug("Communication: %d, Train acc: %f, Test acc: %f, Shapes: %s" % (comm_round, train_acc, test_acc, str(new_shape)))
                    logging.debug('CENTRAL MODEL CONFUSION MATRIX')
                    logging.debug('Train data confusion matrix: \n %s' % str(conf_matrix_train))
                    logging.debug('Test data confusion matrix: \n %s' % str(conf_matrix_test))

                    iter_nets = load_new_state(iter_nets, net_weights_new)

                    expepochs = args.iter_epochs if args.iter_epochs is not None else args.epochs

                    # Train these networks again
                    for net_id, net in iter_nets.items():
                        dataidxs = net_dataidx_map[net_id]
                        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

                        net_train_dl, net_test_dl = get_dataloader(args.dataset, args.datadir, 32, 32, dataidxs)
                        train_net(net_id, net, net_train_dl, net_test_dl, expepochs, lr_iter, reg_iter, net_weights_new[net_id],
                                device=device)

                    lr_iter *= args.lr_decay
                    reg_iter *= args.reg_fac

                
        logging.debug("Trial %d completed" % trial)
        logging.debug("#"*100)

        with open(os.path.join(log_dir, 'trial'+str(trial)+'.json'), 'w') as f:
            json.dump(trials_res[trial], f) 

    with open(os.path.join(log_dir, 'trials_res.json'), 'w') as f:
        json.dump(trials_res, f)

    return trials_res

abli_res = {}

from multiprocessing import Pool
import contextlib

z = lambda x: list(map(int, x.split(',')))
def abli_exp(n=10, dataset="mnist"):

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
    partitions = ["hetero-dir", "homo"]
    #n_layer_list = [i+2 for i in range(n_layers-1)] # don't need layer_num 1
    if args.layers == 1:
        n_list = [i+1 for i in range(n)]
        #n_list = [n]
    else:
        n_list = [i for i in range(0, n+1, 5)]
        n_list[0] = 2
        #n_list = [n]

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
        logdir = os.path.join('eval_pdm_iters',
                              dataset, partition)
        args.loaddir = os.path.join('saved_weights', dataset, partition)
        args.logdir = logdir
        args.partition = partition
        print("Partition type is ", partition)
        abli_res = {}
        
        # set the sig, sig0, gamma
        if dataset == "mnist" and partition == "homo":
            args.pdm_sig0 = 10
            args.pdm_sig = 0.5
            args.pdm_gamma = 10
        elif dataset == "mnist" and partition == "hetero-dir":
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
            #if partition == "hetero-dir":
            #pool_results = po.map_async(run_exp, (n for n in [2, 4, 6]))
            #else:
            pool_results = po.map_async(run_exp, (n for n in n_list))
            results_list = pool_results.get()

        for n, result in zip(n_list, results_list):
            abli_res[n] = result

        with open(os.path.join(logdir, 'abliation_experiment.json'), 'w') as f:
            json.dump(abli_res, f)



if __name__ == "__main__":
    #run_exp()
    parser = get_parser()
    args = parser.parse_args()
    print("Abliation experiment running...")
    abli_exp(n=args.n, dataset=args.dataset)
