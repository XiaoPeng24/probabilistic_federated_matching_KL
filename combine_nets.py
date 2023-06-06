import torch
import torch.nn.functional as F
import numpy as np
import  copy
from model import FcNet, cat_w_b

from matching.pfnm import layer_group_descent as pdm_multilayer_group_descent
from matching.pfnm import layer_skip_group_descent as skip_multilayer_group_descent
from matching.pfnm import layer_wise_group_descent
from matching.pfnm import block_patching, patch_weights
from matching.pfnm_communication import layer_group_descent as pdm_iterative_layer_group_descent
from matching.pfnm_communication import build_init as pdm_build_init

from itertools import product
from sklearn.metrics import confusion_matrix
from utils import *
# from KL_reg_unlimi import local_retrain
import pdb

def prepare_weight_matrix(n_classes, weights: dict):
    weights_list = {}

    for net_i, cls_cnts in weights.items():
        cls = np.array(list(cls_cnts.keys()))
        cnts = np.array(list(cls_cnts.values()))
        weights_list[net_i] = np.array([0] * n_classes, dtype=np.float32)
        weights_list[net_i][cls] = cnts
        weights_list[net_i] = torch.from_numpy(weights_list[net_i]).view(1, -1)

    return weights_list


def prepare_uniform_weights(n_classes, net_cnt, fill_val=1):
    weights_list = {}

    for net_i in range(net_cnt):
        temp = np.array([fill_val] * n_classes, dtype=np.float32)
        weights_list[net_i] = torch.from_numpy(temp).view(1, -1)

    return weights_list


def prepare_sanity_weights(n_classes, net_cnt):
    return prepare_uniform_weights(n_classes, net_cnt, fill_val=0)


def normalize_weights(weights):
    Z = np.array([])
    eps = 1e-6
    weights_norm = {}

    for _, weight in weights.items():
        if len(Z) == 0:
            Z = weight.data.numpy()
        else:
            Z = Z + weight.data.numpy()

    for mi, weight in weights.items():
        weights_norm[mi] = weight / torch.from_numpy(Z + eps)

    return weights_norm


def get_weighted_average_pred(models: list, weights: dict, x, device="cpu"):
    out_weighted = None

    # Compute the predictions
    for model_i, model in enumerate(models):
        #logger.info("Model: {}".format(next(model.parameters()).device))
        #logger.info("data device: {}".format(x.device))
        out = F.softmax(model(x), dim=-1)  # (N, C)

        weight = weights[model_i].to(device)

        if out_weighted is None:
            weight = weight.to(device)
            out_weighted = (out * weight)
        else:
            out_weighted += (out * weight)

    return out_weighted


def compute_ensemble_accuracy(models: list, dataloader, n_classes, train_cls_counts=None, uniform_weights=False, sanity_weights=False, device="cpu"):

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            model.to(device)
            was_training[i] = True
            model.eval()

    if uniform_weights is True:
        weights_list = prepare_uniform_weights(n_classes, len(models))
    elif sanity_weights is True:
        weights_list = prepare_sanity_weights(n_classes, len(models))
    else:
        weights_list = prepare_weight_matrix(n_classes, train_cls_counts)

    weights_norm = normalize_weights(weights_list)

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            #pdb.set_trace()
            x, target = x.to(device), target.to(device)
            target = target.long()
            out = get_weighted_average_pred(models, weights_norm, x, device=device)

            _, pred_label = torch.max(out, 1)

            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()

            if device == "cpu":
                pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                true_labels_list = np.append(true_labels_list, target.data.numpy())
            else:
                pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    #logger.info(correct, total)

    conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    for i, model in enumerate(models):
        if was_training[i]:
            model.train()

    return correct / float(total), conf_matrix


def prepare_fedavg_weights(batch_freqs, nets, device="cpu"):

    total_num = sum(sum(batch_freqs))
    #pdb.set_trace()
    net_i = 0
    weights = []
    for freqs, net in zip(batch_freqs, nets):
        layer_i = 0
        #print("net_id: ", net_i, " layer_id: ", layer_i)
        statedict = net.state_dict()
        #print(statedict)
        ratio = sum(freqs) / total_num
        while True:

            if ('layers.%d.weight' % layer_i) not in statedict.keys():
                break

            if device == "cpu":
                layer_weight = statedict['layers.%d.weight' % layer_i].numpy().T
                layer_bias = statedict['layers.%d.bias' % layer_i].numpy()
            else:
                layer_weight = statedict['layers.%d.weight' % layer_i].cpu().numpy().T
                layer_bias = statedict['layers.%d.bias' % layer_i].cpu().numpy()

            if net_i == 0:
                weights.extend([layer_weight*ratio, layer_bias*ratio])
            else:
                weights[layer_i*2] += layer_weight*ratio
                weights[layer_i*2+1] += layer_bias*ratio
            
            layer_i += 1

        net_i += 1

    return weights


def compute_fedavg_accuracy(models: list, train_dl, test_dl, cls_freqs, n_classes, device="cpu"):
    
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    avg_weights = prepare_fedavg_weights(batch_freqs, models, device=device)

    dims = []
    dims.append(avg_weights[0].shape[0])

    for i in range(0, len(avg_weights), 2):
        dims.append(avg_weights[i].shape[1])

    ip_dim = dims[0]
    op_dim = dims[-1]
    hidden_dims = dims[1:-1]

    pdm_net = FcNet(ip_dim, hidden_dims, op_dim)
    statedict = pdm_net.state_dict()

    # print(pdm_net)

    i = 0
    layer_i = 0
    while i < len(avg_weights):
        weight = avg_weights[i]
        i += 1
        bias = avg_weights[i]
        i += 1

        statedict['layers.%d.weight' % layer_i] = torch.from_numpy(weight.T)
        statedict['layers.%d.bias' % layer_i] = torch.from_numpy(bias)
        layer_i += 1

    pdm_net.load_state_dict(statedict)

    train_acc, conf_matrix_train = compute_ensemble_accuracy([pdm_net], train_dl, n_classes, uniform_weights=True, device=device)
    test_acc, conf_matrix_test = compute_ensemble_accuracy([pdm_net], test_dl, n_classes, uniform_weights=True, device=device)

    return train_acc, test_acc, conf_matrix_train, conf_matrix_test


def pdm_prepare_weights(nets, device="cpu"):
    weights = []

    for net_i, net in enumerate(nets):
        layer_i = 0
        statedict = net.state_dict()
        net_weights = []

        for param_id, (k, v) in enumerate(statedict.items()):
            if device == "cpu":
                if 'conv' in k or 'features' in k:
                    if 'weight' in k:
                        _weight_shape = v.size()
                        if len(_weight_shape) == 4:
                            net_weights.append(v.numpy().reshape(_weight_shape[0],
                                                                 _weight_shape[1] * _weight_shape[2] * _weight_shape[
                                                                     3]))
                        else:
                            pass
                    else:
                        net_weights.append(v.numpy())
                else:
                    if 'weight' in k:
                        net_weights.append(v.cpu().numpy().T)
                    else:
                        net_weights.append(v.cpu().numpy())
            else:
                if 'conv' in k or 'features' in k:
                    if 'weight' in k:
                        _weight_shape = v.size()
                        if len(_weight_shape) == 4:
                            net_weights.append(v.cpu().numpy().reshape(_weight_shape[0],
                                                                       _weight_shape[1] * _weight_shape[2] *
                                                                       _weight_shape[3]))
                        else:
                            pass
                    else:
                        net_weights.append(v.cpu().numpy())
                else:
                    if 'weight' in k:
                        net_weights.append(v.cpu().numpy().T)
                    else:
                        net_weights.append(v.cpu().numpy())

        weights.append(net_weights)

    return weights

def skip_prepare_weights(nets, device="cpu"):
    weights = []
    meta_data = []

    for net_i, net in enumerate(nets):
        layer_i = 0
        cat_w_b(net, device=device)
        statedict = net.state_dict()
        net_weights = []
        while True:

            if ('layers.%d.weight' % layer_i) not in statedict.keys():
                break

            if device == "cpu":
                layer_weight = statedict['layers.%d.weight' % layer_i].numpy()
            else:
                layer_weight = statedict['layers.%d.weight' % layer_i].cpu().numpy()

            net_weights.extend([layer_weight])
            layer_i += 1

        weights.append(net_weights)

    for w in weights[0]:
        meta_data.append(w.shape)

    return weights, meta_data

def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []

    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs

from scipy.stats import multivariate_normal

def weights_prob_selfI_stats(weights, layer_type, sigma0, args):
    """
    Stats the prior probability and self information of weights in each layer
    """

    # get the weight_bias
    n_layers = int(len(weights) / 2)

    stats_layers = {}
    stats_layers['probability'] = []
    stats_layers['self information'] = []

    for layer_index in range(1, n_layers):
        self_information = []

        if args.model == 'fcnet':
            if layer_index == 1:
                weight_bias = np.hstack((weights[0].T, weights[layer_index * 2 - 1].reshape(-1, 1),
                                          weights[layer_index * 2]))
            else:
                weight_bias = np.hstack((weights[layer_index * 2 - 1].reshape(-1, 1),
                                          weights[layer_index * 2]))
        else:
            if 'conv' in layer_type or 'features' in layer_type:
                weight_bias = np.hstack((weights[layer_index * 2 - 2], weights[layer_index * 2 - 1].reshape(-1, 1)))
            elif 'fc' in layer_type or 'classifier' in layer_type:
                weight_bias = np.hstack((weights[layer_index * 2 - 2].T, weights[layer_index * 2 - 1].reshape(-1, 1)))

        dim = weight_bias.shape[-1]
        mask_convari_mat = np.eye(dim) * sigma0
        center = np.zeros(dim)

        prior_probab = multivariate_normal.pdf(weight_bias, mean=center, cov=mask_convari_mat)
        self_info = - np.log(prior_probab)

        stats_layers['probability'].append(prior_probab)
        stats_layers['self information'].append(self_info)

    return  stats_layers

def compute_pdm_net_accuracy(weights, train_dl, test_dl, n_classes, device="cpu"):
    # pdb.set_trace()
    dims = []
    dims.append(weights[0].shape[0])

    for i in range(0, len(weights), 2):
        dims.append(weights[i].shape[1])

    ip_dim = dims[0]
    op_dim = dims[-1]
    hidden_dims = dims[1:-1]

    pdm_net = FcNet(ip_dim, hidden_dims, op_dim)
    statedict = pdm_net.state_dict()

    # print(pdm_net)

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

    pdm_net.load_state_dict(statedict)

    train_acc, conf_matrix_train = compute_ensemble_accuracy([pdm_net], train_dl, n_classes, uniform_weights=True, device=device)
    test_acc, conf_matrix_test = compute_ensemble_accuracy([pdm_net], test_dl, n_classes, uniform_weights=True, device=device)

    return train_acc, test_acc, conf_matrix_train, conf_matrix_test

def compute_full_cnn_accuracy(models, weights, train_dl, test_dl, n_classes, device, args):
    """Note that we only handle the FC weights for now"""
    # we need to figure out the FC dims first

    # LeNetContainer
    # def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10)

    # this should be safe to be hard-coded since most of the modern image classification dataset are in RGB format
    # args_n_nets = len(models)

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
        num_filters = [weights[0].shape[0], weights[2].shape[0], weights[4].shape[0], weights[6].shape[0],
                       weights[8].shape[0], weights[10].shape[0]]
        input_dim = weights[12].shape[0]
        hidden_dims = [weights[12].shape[1], weights[14].shape[1]]
        if args.dataset in ("cifar10", "cinic10"):
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

    # logger.info("Keys of layers of convblock ...")
    new_state_dict = {}
    model_counter = 0
    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        # print("&"*30)
        # print("Key: {}, Weight Shape: {}, Matched weight shape: {}".format(key_name, param.size(), weights[param_idx].shape))
        # print("&"*30)
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

    train_acc, conf_matrix_train = compute_ensemble_accuracy([matched_cnn], train_dl, n_classes, uniform_weights=True,
                                                             device=device)
    test_acc, conf_matrix_test = compute_ensemble_accuracy([matched_cnn], test_dl, n_classes, uniform_weights=True,
                                                           device=device)

    return train_acc, test_acc, conf_matrix_train, conf_matrix_test

def compute_skip_net_accuracy(weights, train_dl, test_dl, n_classes, device="cpu"):

    dims = []
    dims.append(weights[0].shape[1]-1)

    for i in range(0, len(weights)):
        dims.append(weights[i].shape[0]-1)

    ip_dim = dims[0]
    op_dim = dims[-1]+1
    hidden_dims = dims[1:-1]

    skip_net = FcNet(ip_dim, hidden_dims, op_dim)
    cat_w_b(skip_net)
    statedict = skip_net.state_dict()

    # print(pdm_net)

    layer_i = 0
    while layer_i < len(weights):
        weight = weights[layer_i]

        statedict['layers.%d.weight' % layer_i] = torch.from_numpy(weight)
        layer_i += 1

    skip_net.load_state_dict(statedict)

    train_acc, conf_matrix_train = compute_ensemble_accuracy([skip_net], train_dl, n_classes, uniform_weights=True, device=device)
    test_acc, conf_matrix_test = compute_ensemble_accuracy([skip_net], test_dl, n_classes, uniform_weights=True, device=device)

    return train_acc, test_acc, conf_matrix_train, conf_matrix_test

def compute_pdm_matching_multilayer(models, train_dl, test_dl, cls_freqs, n_classes, sigma0=None, it=0, sigma=None, gamma=None, 
                                    device="cpu", KL_reg=0, unlimi=False, use_freq=False, SPAHM=False, nafi=False):
    print("The iteration number of matching is ", it)
    batch_weights = pdm_prepare_weights(models, device=device)
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    gammas = [1.0, 10.0, 50.0] if gamma is None else [gamma]
    sigmas = [1.0, 0.1, 0.5] if sigma is None else [sigma]
    sigma0s = [1.0, 10.0] if sigma0 is None else [sigma0]

    for gamma, sigma, sigma0 in product(gammas, sigmas, sigma0s):
        print("Gamma: ", gamma, "Sigma: ", sigma, "Sigma0: ", sigma0)

        hungarian_weights, assignments = pdm_multilayer_group_descent(
            batch_weights, sigma0_layers=sigma0, sigma_layers=sigma, batch_frequencies=batch_freqs, it=it, gamma_layers=gamma, 
            KL_reg=KL_reg, unlimi=unlimi, use_freq=use_freq, SPAHM=SPAHM, nafi=nafi
        )
        #hungarian_weights.to(device)
        train_acc, test_acc, _, _ = compute_pdm_net_accuracy(hungarian_weights, train_dl, test_dl, n_classes, device=device)

        res = {}
        if train_acc > best_train_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_weights = hungarian_weights
            best_sigma = sigma
            best_gamma = gamma
            best_sigma0 = sigma0
            res['shapes'] = list(map(lambda x: x.shape, best_weights))
            res['train_accuracy'] = best_train_acc
            res['test_accuracy'] = best_test_acc
            res['sigma0'] = best_sigma0
            res['sigma'] = best_sigma
            res['gamma'] = best_gamma
            res['weights'] = best_weights

    print('Best sigma0: %f, Best sigma: %f, Best Gamma: %f, Best accuracy (Test): %f. Training acc: %f' % (
    best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc))

    return res

def BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map, traindata_cls_counts,
            averaging_weights, args, n_classes, sigma0=None, it=0, sigma=None, gamma=None,
            device="cpu", KL_reg=0, unlimi=False):
    # starting the neural matching
    models = nets_list
    cls_freqs = traindata_cls_counts
    assignments_list = []

    batch_weights = pdm_prepare_weights(models, device=device)
    raw_batch_weights = copy.deepcopy(batch_weights)

    logger.info("==" * 15)
    logger.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(nets_list)
    matching_shapes = []

    first_fc_index = None

    for layer_index in range(1, n_layers):
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
            batch_weights=batch_weights,
            layer_index=layer_index,
            sigma0_layers=sigma0,
            sigma_layers=sigma,
            batch_frequencies=batch_freqs,
            it=it,
            gamma_layers=gamma,
            model_meta_data=model_meta_data,
            model_layer_type=layer_type,
            n_layers=n_layers,
            matching_shapes=matching_shapes,
            args=args, KL_reg=KL_reg, unlimi=unlimi
        )
        assignments_list.append(assignment)

        # iii) load weights to the model and train the whole thing
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2]
        if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
            l_type = "conv"
        elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
            l_type = "fc"

        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and (
                    'conv' in type_of_prev_layer or 'features' in type_of_this_layer))

        if first_fc_identifier:
            first_fc_index = layer_index

        matching_shapes.append(L_next)
        tempt_weights = [
            ([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) for w
            in range(num_workers)]

        # i) permutate the next layer wrt matching result
        for worker_index in range(num_workers):
            if first_fc_index is None:
                if l_type == "conv":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2],
                                                    L_next, assignment[worker_index],
                                                    layer_index + 1, model_meta_data,
                                                    matching_shapes=matching_shapes, layer_type=l_type,
                                                    dataset=args.dataset, network_name=args.model)
                elif l_type == "fc":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T,
                                                    L_next, assignment[worker_index],
                                                    layer_index + 1, model_meta_data,
                                                    matching_shapes=matching_shapes, layer_type=l_type,
                                                    dataset=args.dataset, network_name=args.model).T

            elif layer_index >= first_fc_index:
                patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, L_next,
                                               assignment[worker_index]).T

            tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights
        for worker_index in range(num_workers):
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])

        retrained_nets = []
        for worker_index in range(num_workers):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)

            logger.info("Re-training on local worker: {}, starting from layer: {}".format(worker_index,
                                                                                          2 * (layer_index + 1) - 2))
            retrained_cnn = local_retrain((train_dl_local, test_dl_local), tempt_weights[worker_index], args,
                                          freezing_index=(2 * (layer_index + 1) - 2), device=device)
            retrained_nets.append(retrained_cnn)
        batch_weights = pdm_prepare_weights(retrained_nets, device=device)

    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

    with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
        pickle.dump(batch_weights, weights_file)

    last_layer_weights_collector = []

    for i in range(num_workers):
        # firstly we combine last layer's weight and bias
        bias_shape = batch_weights[i][-1].shape
        last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
        last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)

        # the directed normalization doesn't work well, let's try weighted averaging
        last_layer_weights_collector.append(last_layer_weights)

    last_layer_weights_collector = np.array(last_layer_weights_collector)

    avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

    for i in range(n_classes):
        avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
        for j in range(num_workers):
            avg_weight_collector += averaging_weights[j][i] * last_layer_weights_collector[j][:, i]
        avg_last_layer_weight[:, i] = avg_weight_collector

    # avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
    for i in range(num_layers):
        if i < (num_layers - 2):
            matched_weights.append(batch_weights[0][i])

    matched_weights.append(avg_last_layer_weight[0:-1, :])
    matched_weights.append(avg_last_layer_weight[-1, :])

    train_dl, test_dl = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
    train_acc, test_acc, _, _ = compute_full_cnn_accuracy(nets_list, matched_weights,
                                                          train_dl, test_dl, n_classes, device=device)

    res = {}
    res['shapes'] = list(map(lambda x: x.shape, matched_weights))
    res['train_accuracy'] = train_acc
    res['test_accuracy'] = test_acc
    res['sigma0'] = sigma0
    res['sigma'] = best_sigma
    res['gamma'] = best_gamma
    res['weights'] = best_weights

    return res

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

def compute_skip_matching_multilayer(models, train_dl, test_dl, cls_freqs, n_classes, sigma0=None, it=0, sigma=None, gamma=None, 
                                    device="cpu", KL_reg=0, unlimi=False):
    print("The iteration number of matching is ", it)
    batch_weights, layer_meta_data = skip_prepare_weights(models, device=device)
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    gammas = [1.0, 10.0, 50.0] if gamma is None else [gamma]
    sigmas = [1.0, 0.1, 0.5] if sigma is None else [sigma]
    sigma0s = [1.0, 10.0] if sigma0 is None else [sigma0]

    for gamma, sigma, sigma0 in product(gammas, sigmas, sigma0s):
        print("Gamma: ", gamma, "Sigma: ", sigma, "Sigma0: ", sigma0)

        hungarian_weights = skip_multilayer_group_descent(
            batch_weights, sigma0_layers=sigma0, sigma_layers=sigma, batch_frequencies=batch_freqs, 
            layer_meta_data=layer_meta_data, it=it, gamma_layers=gamma, 
            KL_reg=KL_reg, unlimi=unlimi
        )
        #hungarian_weights.to(device)
        train_acc, test_acc, _, _ = compute_skip_net_accuracy(hungarian_weights, train_dl, test_dl, n_classes, device=device)

        key = (sigma0, sigma, gamma)
        res[key] = {}
        res[key]['shapes'] = list(map(lambda x: x.shape, hungarian_weights))
        res[key]['train_accuracy'] = train_acc
        res[key]['test_accuracy'] = test_acc

        print('Sigma0: %s. Sigma: %s. Shapes: %s, Accuracy: %f' % (
        str(sigma0), str(sigma), str(res[key]['shapes']), test_acc))

        if train_acc > best_train_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_weights = hungarian_weights
            best_sigma = sigma
            best_gamma = gamma
            best_sigma0 = sigma0

    print('Best sigma0: %f, Best sigma: %f, Best Gamma: %f, Best accuracy (Test): %f. Training acc: %f' % (
    best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc))

    return (best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, res)


def compute_iterative_pdm_matching(models, train_dl, test_dl, cls_freqs, n_classes, sigma, sigma0, gamma, it, old_assignment=None,
                                   device="cpu", KL_reg=0):

    batch_weights = pdm_prepare_weights(models, device=device)
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)

    hungarian_weights, assignments = pdm_iterative_layer_group_descent(
        batch_weights, batch_freqs, sigma_layers=sigma, sigma0_layers=sigma0, gamma_layers=gamma, it=it, assignments_old=old_assignment,
        KL_reg=KL_reg
    )

    train_acc, test_acc, conf_matrix_train, conf_matrix_test = compute_pdm_net_accuracy(hungarian_weights, train_dl, test_dl, n_classes,
                                                                                        device=device)

    batch_weights_new = [pdm_build_init(hungarian_weights, assignments, j) for j in range(len(models))]
    matched_net_shapes = list(map(lambda x: x.shape, hungarian_weights))

    return batch_weights_new, train_acc, test_acc, matched_net_shapes, assignments, hungarian_weights, conf_matrix_train, conf_matrix_test

    
def flatten_weights(weights_j):
    flat_weights = np.hstack((weights_j[0].T, weights_j[1].reshape(-1,1), weights_j[2]))
    return flat_weights


def build_network(clusters, batch_weights, D):
    cluster_network = [clusters[:,:D].T, clusters[:,D].T, clusters[:,(D+1):]]
    bias = np.mean(batch_weights, axis=0)[-1]
    cluster_network += [bias]
    return cluster_network