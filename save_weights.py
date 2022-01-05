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

from utils import *

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
    parser.add_argument('--init_same', type=bool, required=False, help='All models init with same weight')

    # partitioning setting......
    parser.add_argument('--layers', type=int, required=True, help='do n_nets or n_layers')
    parser.add_argument('--n', nargs='+', type=int, required=True, help='the number of nets or layers')
    parser.add_argument('--n_layers', type=int, required=False, default=1, help="Number of hidden layers")
    parser.add_argument('--n_nets', type=int, required=False, default=10, help="Number of nets to initialize")
    parser.add_argument('--partition', type=str, required=False, help="Partition = homo/hetero/hetero-dir")
    parser.add_argument('--alpha', type=float, required=False, default=0.5,
                        help="Dirichlet distribution constant used for data partitioning")

    # save and load dir
    parser.add_argument('--loaddir', type=str, required=False, help='Load weights directory path')
    parser.add_argument('--logdir', type=str, required=False, help='Log directory path')

    # training setting......
    parser.add_argument('--dataset', type=str, required=False, default="mnist", help="Dataset [mnist/cifar10]")
    parser.add_argument('--datadir', type=str, required=False, default="./data/mnist", help="Data directory")
    parser.add_argument('--model', type=str, required=True, default="fcnet", help="which model to train")
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))),
                        help="the layer config for Fully Connected neural network")
    parser.add_argument('--lr', type=float, required=False, default=0.01, help="Learning rate")
    parser.add_argument('--retrain_lr', type=float, default=0.1,
                        help='learning rate using in specific for local network retrain (default: 0.01)')
    parser.add_argument('--fine_tune_lr', type=float, default=0.1,
                        help='learning rate using in specific for fine tuning the softmax layer on the data center (default: 0.01)')
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Epochs")
    parser.add_argument('--retrain_epochs', type=int, default=1,
                        help='how many epochs will be trained in during the locally retraining process')
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--batch_size', type=int, required=False, default=32, help="the batch size to train")
    parser.add_argument('--reg', type=float, required=False, default=1e-6, help="L2 regularization strength")
    parser.add_argument('--retrain', type=bool, required=True, default=True,
                        help="Do we need retrain the init weights?")

    # randomization setting
    parser.add_argument('--init_seed', type=int, required=False, default=0, help="Random seed")

    # matching experiment setting......
    parser.add_argument('--experiment', required=False, default="None", type=lambda s: s.split(','),
                        help="Type of experiment to run. [none/w-ensemble/u-ensemble/pdm/all]")
    parser.add_argument('--trials', type=int, required=False, default=1, help="Number of trials for each run")
    parser.add_argument('--iter_epochs', type=int, required=False, default=5, help="Epochs for PDM-iterative method")
    parser.add_argument('--reg_fac', type=float, required=False, default=0.0, help="Regularization factor for PDM Iter")
    parser.add_argument('--pdm_sig', type=float, required=False, default=1.0, help="PDM sigma param")
    parser.add_argument('--pdm_sig0', type=float, required=False, default=1.0, help="PDM sigma0 param")
    parser.add_argument('--pdm_gamma', type=float, required=False, default=1.0, help="PDM gamma param")
    parser.add_argument('--communication_rounds', type=int, required=False, default=None,
                        help="How many iterations of PDM matching should be done")
    parser.add_argument('--lr_decay', type=float, required=False, default=1.0,
                        help="Decay LR after every PDM iterative communication")

    # hardware setting (device and multiprocessing)
    parser.add_argument('--device', type=str, required=False, help="Device to run")
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

def train_net(net_id, net, train_dataloader, test_dataloader, args, reg_base_weights=None,
              save_path=None, device="cpu"):

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    net.to(device)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    logger.info('>> Pre-Training Test accuracy: %f' % test_acc)

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

        logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), args.reg*l2_reg))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')
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
    print("Current n  is %d " % n)
    #args.n_nets = n_nets
    #args.logdir = os.path.join(logdir, "n_nets "+str(n_nets))
    #gpu_id = int(n_layer+2 % 4)
    gpu_id = 2 if n%2 == 0 else 0
    device_str = "cuda:" + str(gpu_id)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Current device is :", device)

    if args.layers == 1:
        if args.model == "fcnet":
            if args.dataset == "mnist":
                args.net_config = list(map(int, ("784, " + "100, " * n + "10").split(', ')))
            elif args.dataset == "cifar10":
                args.net_config = list(map(int, ("3072, " + "100, " * n + "10").split(', ')))

        log_dir = os.path.join(args.logdir, "n_layers "+str(n))
    else:
        args.n_nets = n
        log_dir = os.path.join(args.logdir, "n_nets "+str(n))

    if not os.path.exists(log_dir):
        mkdirs(log_dir)

    logger.info("Experiment arguments: %s" % str(args))

    trials_res = {}
    trials_res["Experiment arguments"] = str(args)

    print("The total trials of n_nets %d is " % args.n_nets, args.trials)

    for trial in range(args.trials):
        save_dir = os.path.join(log_dir, "trial "+str(trial))
        if not os.path.exists(save_dir):
            mkdirs(save_dir)

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
        nets, model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_nets, args)

        local_train_accs = []
        local_test_accs = []
        start = datetime.datetime.now()
        for net_id, net in nets.items():
            dataidxs = net_dataidx_map[net_id]
            print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

            save_path = os.path.join(save_dir, "model "+str(net_id)+".pkl")
            train_dl, test_dl = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            trainacc, testacc = train_net(net_id, net, train_dl, test_dl, args, device=device)
            # saving the trained models here
            with open(save_path, "wb") as f_:
                torch.save(net.state_dict(), f_)

            local_train_accs.append(trainacc)
            local_test_accs.append(testacc)

        end = datetime.datetime.now()
        timing = (end - start).seconds
        trials_res[trial]['training time'] = timing

        trials_res[trial]['local_train_accs'] = local_train_accs
        trials_res[trial]['local_test_accs'] = local_test_accs
        train_dl, test_dl = get_dataloader(args.dataset, args.datadir, 32, 32)

        logging.debug("*"*50)
        logging.debug("Running experiments \n")

        nets_list = list(nets.values())

        logging.debug("Trial %d completed" % trial)
        logging.debug("#"*100)

        with open(os.path.join(save_dir, 'trial'+str(trial)+'.json'), 'w') as f:
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
    partitions = ["hetero-dir", "homo"]
    #n_layer_list = [i+2 for i in range(n_layers-1)] # don't need layer_num 1
    if args.layers == 1:
        n_list = n
    else:
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

    for partition in partitions:
        #logdir = os.path.join('log_abli', now.strftime("%Y-%m-%d %H"),
        #                      dataset, partition)
        if args.init_same:
            logdir = os.path.join('saved_weights', dataset,
                                  'init_same', partition, args.model)
        else:
            logdir = os.path.join('saved_weights',
                                  dataset, partition, args.model)
        args.logdir = logdir
        args.partition = partition
        print("Partition type is ", partition)
        abli_res = {}
        
        with contextlib.closing(Pool(args.num_pool_workers)) as po:
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
