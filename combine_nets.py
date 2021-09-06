import torch
import torch.nn.functional as F
import numpy as np
from model import FcNet, cat_w_b

from matching.pfnm import layer_group_descent as pdm_multilayer_group_descent
from matching.pfnm import layer_skip_group_descent as skip_multilayer_group_descent
from matching.pfnm_communication import layer_group_descent as pdm_iterative_layer_group_descent
from matching.pfnm_communication import build_init as pdm_build_init

from itertools import product
from sklearn.metrics import confusion_matrix
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
        while True:

            if ('layers.%d.weight' % layer_i) not in statedict.keys():
                break

            if device == "cpu":
                layer_weight = statedict['layers.%d.weight' % layer_i].numpy().T
                layer_bias = statedict['layers.%d.bias' % layer_i].numpy()
            else:
                layer_weight = statedict['layers.%d.weight' % layer_i].cpu().numpy().T
                layer_bias = statedict['layers.%d.bias' % layer_i].cpu().numpy()

            net_weights.extend([layer_weight, layer_bias])
            layer_i += 1

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


def compute_pdm_net_accuracy(weights, train_dl, test_dl, n_classes, device="cpu"):

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
                                    device="cpu", KL_reg=0, unlimi=False, use_freq=False):
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

        hungarian_weights = pdm_multilayer_group_descent(
            batch_weights, sigma0_layers=sigma0, sigma_layers=sigma, batch_frequencies=batch_freqs, it=it, gamma_layers=gamma, 
            KL_reg=KL_reg, unlimi=unlimi, use_freq=use_freq
        )
        #hungarian_weights.to(device)
        train_acc, test_acc, _, _ = compute_pdm_net_accuracy(hungarian_weights, train_dl, test_dl, n_classes, device=device)

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