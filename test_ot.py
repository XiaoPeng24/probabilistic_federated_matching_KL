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
## parameters for optimal transport matching
# for the optimal transport calculation
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

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--layers', type=int, required=False, help='do n_nets or n_layers')
    parser.add_argument('--n', nargs='+', type=int, required=False, help='the number of nets or layers')

    parser.add_argument('--loaddir', type=str, required=False, help='Load weights directory path')
    parser.add_argument('--logdir', type=str, required=False, help='Log directory path')
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--dataset', type=str, required=False, default="mnist", help="Dataset [mnist/cifar10]")
    parser.add_argument('--datadir', type=str, required=False, default="./data/mnist", help="Data directory")
    parser.add_argument('--init_seed', type=int, required=False, default=0, help="Random seed")

    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))

    parser.add_argument('--n_layers', type=int, required=False, default=1, help="Number of hidden layers")

    parser.add_argument('--n_nets', type=int, required=False, default=10, help="Number of nets to initialize")
    parser.add_argument('--model', type=str, required=False, default="fcnet", help="The model of which to train")
    parser.add_argument('--partition', type=str, required=False, help="Partition = homo/hetero/hetero-dir")
    parser.add_argument('--experiment', required=False, default="None", type=lambda s: s.split(','),
                        help="Type of experiment to run. [none/w-ensemble/u-ensemble/pdm/all]")
    parser.add_argument('--trials', type=int, required=False, default=1, help="Number of trials for each run")

    parser.add_argument('--lr', type=float, required=False, default=0.01, help="Learning rate")
    parser.add_argument('--batch_size', type=int, required=False, default=32, help="Learning rate")
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Epochs")
    parser.add_argument('--reg', type=float, required=False, default=1e-6, help="L2 regularization strength")
    parser.add_argument('--retrain', type=bool, default=True, help="Do we need retrain the init weights?")

    parser.add_argument('--alpha', type=float, required=False, default=0.5,
                        help="Dirichlet distribution constant used for data partitioning")

    parser.add_argument('--communication_rounds', type=int, required=False, default=None,
                        help="How many iterations of PDM matching should be done")
    parser.add_argument('--lr_decay', type=float, required=False, default=1.0,
                        help="Decay LR after every PDM iterative communication")
    parser.add_argument('--iter_epochs', type=int, required=False, default=5, help="Epochs for PDM-iterative method")
    parser.add_argument('--reg_fac', type=float, required=False, default=0.0, help="Regularization factor for PDM Iter")

    parser.add_argument('--pdm_sig', type=float, required=False, default=1.0, help="PDM sigma param")
    parser.add_argument('--pdm_sig0', type=float, required=False, default=1.0, help="PDM sigma0 param")
    parser.add_argument('--pdm_gamma', type=float, required=False, default=1.0, help="PDM gamma param")

    parser.add_argument('--device', type=str, required=False, default=1.0, help="Device to run")
    parser.add_argument('--num_pool_workers', type=int, required=False, help='the num of workers')

    return parser

parser = get_parser()
args = parser.parse_args([])
args.layers =1 
args.n =5
args.num_pool_workers = 1
args.net_config = [784, 100, 10]
args.dropout_p = 0.5
args.n_nets = 5
args.dataset = 'mnist'
args.datadir = os.path.join("data", args.dataset)
# args.model = "simple-cnn"
args.model = "fcnet"
nets, model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_nets, args)
train_dl, test_dl = get_dataloader(args.dataset, args.datadir, 32, 32)
nets_list = list(nets.values())
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
aligned_net = nets_list[0].to(device)
for i in range(1, len(nets_list)):
    fusion_models = [nets_list[i].to(device), aligned_net]
    ot_args.input_dim = 784
    ot_args.hidden_dims = [100]
    ot_args.output_dim = 10
    ot_args.model_name = 'fcnet'
    geometric_acc, aligned_net = wasserstein_ensemble.geometric_ensembling_modularized(ot_args, fusion_models,
                                                                                       train_dl, test_dl)