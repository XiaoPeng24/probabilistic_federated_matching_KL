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
    parser.add_argument('--partition', type=str, required=False, help="Partition = homo/hetero/hetero-dir")
    parser.add_argument('--experiment', required=False, default="None", type=lambda s: s.split(','), help="Type of experiment to run. [none/w-ensemble/u-ensemble/pdm/all]")
    parser.add_argument('--trials', type=int, required=False, default=1, help="Number of trials for each run")
    
    parser.add_argument('--lr', type=float, required=False, default=0.01, help="Learning rate")
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Epochs")
    parser.add_argument('--reg', type=float, required=False, default=1e-6, help="L2 regularization strength")
    parser.add_argument('--retrain', type=bool, required=True, default=True, help="Do we need retrain the init weights?")

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
            train_acc, test_acc = train_net(net_id, net, train_dl_local, test_dl_local, args.epochs, args.lr,
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


def local_retrain(local_datasets, weights, args, mode="bottom-up", freezing_index=0, ori_assignments=None,
                  device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    """
    if args.model == "lenet":
        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        kernel_size = 5
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1]]
        output_dim = weights[-1].shape[0]
        logger.info("Num filters: {}, Input dim: {}, hidden_dims: {}, output_dim: {}".format(num_filters, input_dim,
                                                                                             hidden_dims, output_dim))

        matched_cnn = LeNetContainer(
            num_filters=num_filters,
            kernel_size=kernel_size,
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim)
    elif args.model == "vgg":
        matched_shapes = [w.shape for w in weights]
        matched_cnn = matched_vgg11(matched_shapes=matched_shapes)
    elif args.model == "simple-cnn":
        # input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        # [(9, 75), (9,), (19, 225), (19,), (475, 123), (123,), (123, 87), (87,), (87, 10), (10,)]
        if args.dataset in ("cifar10", "cinic10"):
            input_channel = 3
        elif args.dataset == "mnist":
            input_channel = 1

        num_filters = [weights[0].shape[0], weights[2].shape[0]]
        input_dim = weights[4].shape[0]
        hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
        matched_cnn = SimpleCNNContainer(input_channel=input_channel,
                                         num_filters=num_filters,
                                         kernel_size=5,
                                         input_dim=input_dim,
                                         hidden_dims=hidden_dims,
                                         output_dim=10)
    elif args.model == "moderate-cnn":
        # [(35, 27), (35,), (68, 315), (68,), (132, 612), (132,), (132, 1188), (132,),
        # (260, 1188), (260,), (260, 2340), (260,),
        # (4160, 1025), (1025,), (1025, 515), (515,), (515, 10), (10,)]
        if mode not in ("block-wise", "squeezing"):
            num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0],
                           weights[8].shape[0], weights[10].shape[0]]
            input_dim = weights[12].shape[0]
            hidden_dims = [weights[12].shape[1], weights[14].shape[1]]

            input_dim = weights[12].shape[0]
        elif mode == "block-wise":
            # for block-wise retraining the `freezing_index` becomes a range of indices
            # so at here we need to generate a unfreezing list:
            __unfreezing_list = []
            for fi in freezing_index:
                __unfreezing_list.append(2 * fi - 2)
                __unfreezing_list.append(2 * fi - 1)

            # we need to do two changes here:
            # i) switch the number of filters in the freezing indices block to the original size
            # ii) cut the correspoidng color channels
            __fixed_indices = set([i * 2 for i in range(6)])  # 0, 2, 4, 6, 8, 10
            dummy_model = ModerateCNN()

            num_filters = []
            for pi, param in enumerate(dummy_model.parameters()):
                if pi in __fixed_indices:
                    if pi in __unfreezing_list:
                        num_filters.append(param.size()[0])
                    else:
                        num_filters.append(weights[pi].shape[0])
            del dummy_model
            logger.info("################ Num filters for now are : {}".format(num_filters))
            # note that we hard coded index of the last conv layer here to make sure the dimension is compatible
            if freezing_index[0] != 6:
                # if freezing_index[0] not in (6, 7):
                input_dim = weights[12].shape[0]
            else:
                # we need to estimate the output shape here:
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=num_filters)
                dummy_input = torch.rand(1, 3, 32, 32)
                estimated_output = shape_estimator(dummy_input)
                # estimated_shape = (estimated_output[1], estimated_output[2], estimated_output[3])
                input_dim = estimated_output.view(-1).size()[0]

            if (freezing_index[0] <= 6) or (freezing_index[0] > 8):
                hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
            else:
                dummy_model = ModerateCNN()
                for pi, param in enumerate(dummy_model.parameters()):
                    if pi == 2 * freezing_index[0] - 2:
                        _desired_shape = param.size()[0]
                if freezing_index[0] == 7:
                    hidden_dims = [_desired_shape, weights[14].shape[1]]
                elif freezing_index[0] == 8:
                    hidden_dims = [weights[12].shape[1], _desired_shape]
        elif mode == "squeezing":
            pass

        if args.dataset in ("cifar10", "cinic10"):
            if mode == "squeezing":
                matched_cnn = ModerateCNN()
            else:
                matched_cnn = ModerateCNNContainer(3,
                                                   num_filters,
                                                   kernel_size=3,
                                                   input_dim=input_dim,
                                                   hidden_dims=hidden_dims,
                                                   output_dim=10)
        elif args.dataset == "mnist":
            matched_cnn = ModerateCNNContainer(1,
                                               num_filters,
                                               kernel_size=3,
                                               input_dim=input_dim,
                                               hidden_dims=hidden_dims,
                                               output_dim=10)

    new_state_dict = {}
    model_counter = 0
    n_layers = int(len(weights) / 2)

    # we hardcoded this for now: will probably make changes later
    # if mode != "block-wise":
    if mode not in ("block-wise", "squeezing"):
        __non_loading_indices = []
    else:
        if mode == "block-wise":
            if freezing_index[0] != n_layers:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
                __non_loading_indices.append(
                    __unfreezing_list[-1] + 1)  # add the index of the weight connects to the next layer
            else:
                __non_loading_indices = copy.deepcopy(__unfreezing_list)
        elif mode == "squeezing":
            # please note that at here we need to reconstruct the entire local network and retrain it
            __non_loading_indices = [i for i in range(len(weights))]

    def __reconstruct_weights(weight, assignment, layer_ori_shape, matched_num_filters=None, weight_type="conv_weight",
                              slice_dim="filter"):
        # what contains in the param `assignment` is the assignment for a certain layer, a certain worker
        """
        para:: slice_dim: for reconstructing the conv layers, for each of the three consecutive layers, we need to slice the
               filter/kernel to reconstruct the first conv layer; for the third layer in the consecutive block, we need to
               slice the color channel
        """
        if weight_type == "conv_weight":
            if slice_dim == "filter":
                res_weight = weight[assignment, :]
            elif slice_dim == "channel":
                _ori_matched_shape = list(copy.deepcopy(layer_ori_shape))
                _ori_matched_shape[1] = matched_num_filters
                trans_weight = trans_next_conv_layer_forward(weight, _ori_matched_shape)
                sliced_weight = trans_weight[assignment, :]
                res_weight = trans_next_conv_layer_backward(sliced_weight, layer_ori_shape)
        elif weight_type == "bias":
            res_weight = weight[assignment]
        elif weight_type == "first_fc_weight":
            # NOTE: please note that in this case, we pass the `estimated_shape` to `layer_ori_shape`:
            __ori_shape = weight.shape
            res_weight = weight.reshape(matched_num_filters, layer_ori_shape[2] * layer_ori_shape[3] * __ori_shape[1])[
                         assignment, :]
            res_weight = res_weight.reshape((len(assignment) * layer_ori_shape[2] * layer_ori_shape[3], __ori_shape[1]))
        elif weight_type == "fc_weight":
            if slice_dim == "filter":
                res_weight = weight.T[assignment, :]
                # res_weight = res_weight.T
            elif slice_dim == "channel":
                res_weight = weight[assignment, :].T
        return res_weight

    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if (param_idx in __non_loading_indices) and (freezing_index[0] != n_layers):
            # we need to reconstruct the weights here s.t.
            # i) shapes of the weights are euqal to the shapes of the weight in original model (before matching)
            # ii) each neuron comes from the corresponding global neuron
            _matched_weight = weights[param_idx]
            _matched_num_filters = weights[__non_loading_indices[0]].shape[0]
            #
            # we now use this `_slice_dim` for both conv layers and fc layers
            if __non_loading_indices.index(param_idx) != 2:
                _slice_dim = "filter"  # please note that for biases, it doesn't really matter if we're going to use filter or channel
            else:
                _slice_dim = "channel"

            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments,
                                                        layer_ori_shape=param.size(),
                                                        matched_num_filters=_matched_num_filters,
                                                        weight_type="conv_weight", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_weight.reshape(param.size()))}
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments,
                                                      layer_ori_shape=param.size(),
                                                      matched_num_filters=_matched_num_filters,
                                                      weight_type="bias", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    if freezing_index[0] != 6:
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments,
                                                            layer_ori_shape=param.size(),
                                                            matched_num_filters=_matched_num_filters,
                                                            weight_type="fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight)}
                    else:
                        # that's for handling the first fc layer that is connected to the conv blocks
                        _res_weight = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments,
                                                            layer_ori_shape=estimated_output.size(),
                                                            matched_num_filters=_matched_num_filters,
                                                            weight_type="first_fc_weight", slice_dim=_slice_dim)
                        temp_dict = {key_name: torch.from_numpy(_res_weight.T)}
                elif "bias" in key_name:
                    _res_bias = __reconstruct_weights(weight=_matched_weight, assignment=ori_assignments,
                                                      layer_ori_shape=param.size(),
                                                      matched_num_filters=_matched_num_filters,
                                                      weight_type="bias", slice_dim=_slice_dim)
                    temp_dict = {key_name: torch.from_numpy(_res_bias)}
        else:
            if "conv" in key_name or "features" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
            elif "fc" in key_name or "classifier" in key_name:
                if "weight" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
                elif "bias" in key_name:
                    temp_dict = {key_name: torch.from_numpy(weights[param_idx])}

        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    for param_idx, param in enumerate(matched_cnn.parameters()):
        if mode == "bottom-up":
            # for this freezing mode, we freeze the layer before freezing index
            if param_idx < freezing_index:
                param.requires_grad = False
        elif mode == "per-layer":
            # for this freezing mode, we only unfreeze the freezing index
            if param_idx not in (2 * freezing_index - 2, 2 * freezing_index - 1):
                param.requires_grad = False
        elif mode == "block-wise":
            # for block-wise retraining the `freezing_index` becomes a range of indices
            if param_idx not in __non_loading_indices:
                param.requires_grad = False
        elif mode == "squeezing":
            pass

    matched_cnn.to(device).train()
    # start training last fc layers:
    train_dl_local = local_datasets[0]
    test_dl_local = local_datasets[1]

    if mode != "block-wise":
        if freezing_index < (len(weights) - 2):
            optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()),
                                            lr=args.retrain_lr, momentum=0.9)
        else:
            optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()),
                                            lr=(args.retrain_lr / 10), momentum=0.9, weight_decay=0.0001)
    else:
        # optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr, momentum=0.9)
        optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=0.001,
                                         weight_decay=0.0001, amsgrad=True)

    criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    logger.info('n_training: %d' % len(train_dl_local))
    logger.info('n_test: %d' % len(test_dl_local))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    logger.info('>> Pre-Training Test accuracy: %f' % test_acc)

    if mode != "block-wise":
        if freezing_index < (len(weights) - 2):
            retrain_epochs = args.retrain_epochs
        else:
            retrain_epochs = int(args.retrain_epochs * 3)
    else:
        retrain_epochs = args.retrain_epochs

    for epoch in range(retrain_epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dl_local):
            x, target = x.to(device), target.to(device)

            optimizer_fine_tune.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = matched_cnn(x)
            loss = criterion_fine_tune(out, target)
            epoch_loss_collector.append(loss.item())

            loss.backward()
            optimizer_fine_tune.step()

        # logger.debug('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    return matched_cnn

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
    #gpu_id = 3 if n%4 == 0 else 1
    gpu_id = 1
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
    logger.basicConfig(
        filename=filename,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logger.DEBUG, filemode='w')

    logger.debug("Experiment arguments: %s" % str(args))

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
        logger.debug("#" * 100)
        logger.debug("Executing Trial %d with seed %d" % (trial, seed))

        np.random.seed(seed)
        torch.manual_seed(seed)

        print("Partitioning data")
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
            averaging_weights[:, i] = worker_class_counts / total_num_counts

        logger.info("averaging_weights: {}".format(averaging_weights))

        print("Initializing nets")
        nets, model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_nets, args)

        # local_train_accs = []
        # local_test_accs = []
        start = datetime.datetime.now()
        # for net_id, net in nets.items():
        #     dataidxs = net_dataidx_map[net_id]
        #     print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
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
            print("Computing Uniform ensemble accuracy")
            uens_train_acc, _ = compute_ensemble_accuracy(nets_list, train_dl, n_classes,  uniform_weights=True, device=device)
            uens_test_acc, _ = compute_ensemble_accuracy(nets_list, test_dl, n_classes, uniform_weights=True, device=device)

            logger.debug("Uniform ensemble (Train acc): %f" % uens_train_acc)
            logger.debug("Uniform ensemble (Test acc): %f" % uens_test_acc)

            trials_res[trial]["Uniform ensemble (Train acc)"] = uens_train_acc
            trials_res[trial]["Uniform ensemble (Test acc)"] = uens_test_acc

        if ("FedAvg" in args.experiment) or ("all" in args.experiment):
            print("Computing FedAvg accuracy")
            avg_train_acc, avg_test_acc, _, _ = compute_fedavg_accuracy(nets_list, train_dl, test_dl, traindata_cls_counts, n_classes, device=device)

            logger.debug("FedAvg (Train acc): %f" % avg_train_acc)
            logger.debug("FedAvg (Test acc): %f" % avg_test_acc)

            trials_res[trial]["FedAvg (Train acc)"] = avg_train_acc
            trials_res[trial]["FedAvg (Test acc)"] = avg_test_acc

        if ("pdm_KL" in args.experiment) or ("all" in args.experiment):
            print("Computing hungarian matching")
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

                if best_test_acc > comp:
                    best_reg = KL_reg
                    comp = best_test_acc

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["pdm_KL_reg "+str(KL_reg)]["timing"] = timing

            trials_res[trial]["best_reg"] = best_reg

        if ("pdm_I" in args.experiment) or ("all" in args.experiment):
            
            if trial == 0:
                log_dir = os.path.join(log_dir, 'self_info_cost1')
                if not os.path.exists(log_dir):
                    mkdirs(log_dir)

            print("Computing hungarian matching")
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

                print("Trial %d I_reg %f finished!!!" % (trial, I_reg))

            trials_res[trial]["best_reg"] = best_reg

        if ("pdm_neg_I" in args.experiment) or ("all" in args.experiment):
            
            if trial == 0:
                log_dir = os.path.join(log_dir, 'self_info_neg_cost')
                if not os.path.exists(log_dir):
                    mkdirs(log_dir)

            print("Computing hungarian matching")
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

                print("Trial %d I_neg_reg %f finished!!!" % (trial, I_reg))

            trials_res[trial]["best_reg"] = best_reg

        if ("pdm_coff" in args.experiment) or ("all" in args.experiment):

            if trial == 0:
                log_dir = os.path.join(log_dir, 'coff_cost')
                if not os.path.exists(log_dir):
                    mkdirs(log_dir)

            print("Computing hungarian matching")
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

                print("Trial %d coff %f finished!!!" % (trial, coff))

            trials_res[trial]["best_coff"] = str(best_coff)

        if ("pdm_fix" in args.experiment) or ("all" in args.experiment):

            if trial == 0:
                log_dir = os.path.join(log_dir, 'fix_cost')
                if not os.path.exists(log_dir):
                    mkdirs(log_dir)

            print("Computing hungarian matching")
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

                print("Trial %d coff %f finished!!!" % (trial, coff))

            trials_res[trial]["best_coff"] = str(best_coff)

        if ("pdm_iterative" in args.experiment) or ("all" in args.experiment):
            print("Running Iterative PDM matching procedure")
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
                        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))

                        net_train_dl, net_test_dl = get_dataloader(args.dataset, args.datadir, 32, 32, dataidxs)
                        train_net(net_id, net, net_train_dl, net_test_dl, expepochs, lr_iter, reg_iter, net_weights_new[net_id],
                                device=device)

                    lr_iter *= args.lr_decay
                    reg_iter *= args.reg_fac

                
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
    partitions = ["hetero-dir", "homo"]
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
        logdir = os.path.join('KL_regs_bst_hyp_unlimi',
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
