import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from itertools import product
import math
import copy
import time
import pickle
from sklearn.metrics import confusion_matrix

from model import *
from datasets import load_mnist_data, load_cifar10_data, MNIST_truncated, CIFAR10_truncated
from datasets import load_fashionmnist_data, FashionMNIST_truncated

# we've changed to a faster solver
#from scipy.optimize import linear_sum_assignment
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
    elif dataset == 'fashionmnist':
        X_train, y_train, X_test, y_test = load_fashionmnist_data(datadir)

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
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)

    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)


def init_nets(net_configs, dropout_p, n_nets, args):
    input_size = net_configs[0]
    output_size = net_configs[-1]
    hidden_sizes = net_configs[1:-1]

    nets = {net_i: None for net_i in range(n_nets)}

    model_meta_data = []
    layer_type = []

    for net_i in range(n_nets):
        if args.model == "fcnet":
            net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif args.model == "simple-cnn":
            net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
        elif args.model == "moderate-cnn":
            net = ModerateCNN()

        nets[net_i] = net

    if args.model != "fcnet":
        for (k, v) in nets[0].state_dict().items():
            model_meta_data.append(v.shape)
            layer_type.append(k)
            # logger.info("{} ::: Layer name: {}, layer shape: {}".format(args.model, k, v.shape))

    return nets, model_meta_data, layer_type

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset == 'mnist':
        dl_obj = MNIST_truncated
    elif dataset == 'cifar10':
        dl_obj = CIFAR10_truncated
    elif dataset == 'fashionmnist':
        dl_obj = FashionMNIST_truncated

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
        return correct / float(total), conf_matrix

    return correct / float(total)
