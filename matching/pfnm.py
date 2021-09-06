import numpy as np
from scipy.optimize import linear_sum_assignment
import pdb

from matching.kl_cost import compute_KL3_cost, compute_KL2_cost
from matching.kl_cost import compute_cost as unlimi_cost, compute_KL4_cost as unlimi_KL_cost


def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j):

    match_norms = ((weights_j_l + global_weights) ** 2 / (sigma_inv_j + global_sigmas)).sum(axis=1) - (
                global_weights ** 2 / global_sigmas).sum(axis=1)

    return match_norms


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    #pdb.set_trace()

    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts), 10)
    param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    param_cost += np.log(counts / (J - counts))

    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost


def matching_upd_j(weights_j, global_weights, sigma_inv_j, global_sigmas, prior_mean_norm, prior_inv_sigma,
                   popularity_counts, gamma, J, KL_reg=0, unlimi=False):

    L = global_weights.shape[0]

    if unlimi:
        full_cost = unlimi_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)
    else:
        full_cost = compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)

    KL_cost = 0
    
    #if KL == 1:
    #	KL_cost = compute_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
    #                         popularity_counts, gamma, J)
    #elif KL == 2:
    #    KL_cost = compute_KL2_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
    #                         popularity_counts, gamma, J)
    #elif KL == 3:
    #    KL_cost = compute_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
    #                         popularity_counts, gamma, J)
    #    full_cost = 0
    #elif KL == 4:
    #    KL_cost = compute_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
    #                         popularity_counts, gamma, J)
    #    full_cost = 0

    if KL_reg!=0:
        if unlimi:
            KL_cost = KL_reg*unlimi_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)
        else:
            KL_cost = KL_reg*compute_KL2_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)

    row_ind, col_ind = linear_sum_assignment(-full_cost+KL_cost)

    assignment_j = []

    new_L = L

    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_weights[i] += weights_j[l]
            global_sigmas[i] += sigma_inv_j
        else:  # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_weights = np.vstack((global_weights, prior_mean_norm + weights_j[l]))
            global_sigmas = np.vstack((global_sigmas, prior_inv_sigma + sigma_inv_j))

    return global_weights, global_sigmas, popularity_counts, assignment_j


def objective(global_weights, global_sigmas):
    obj = ((global_weights) ** 2 / global_sigmas).sum()
    return obj


def patch_weights(w_j, L_next, assignment_j_c):
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    return new_w_j


def process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0):
    J = len(batch_weights)
    sigma_bias = sigma
    sigma0_bias = sigma0
    mu0_bias = 0.1
    softmax_bias = [batch_weights[j][-1] for j in range(J)]
    softmax_inv_sigma = [s / sigma_bias for s in last_layer_const]
    softmax_bias = sum([b * s for b, s in zip(softmax_bias, softmax_inv_sigma)]) + mu0_bias / sigma0_bias
    softmax_inv_sigma = 1 / sigma0_bias + sum(softmax_inv_sigma)
    return softmax_bias, softmax_inv_sigma


def match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, it, KL_reg=0, 
                unlimi=False):
    """
    weight_bias: [J, np.array(n_neurons, dim)]
    """
    #pdb.set_trace()
    
    J = len(weights_bias)

    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])

    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior

    global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
    global_sigmas = np.outer(np.ones(global_weights.shape[0]), sigma_inv_prior + sigma_inv_layer[group_order[0]])

    popularity_counts = [1] * global_weights.shape[0]

    assignment = [[] for _ in range(J)]

    assignment[group_order[0]] = list(range(global_weights.shape[0]))

    ## Initialize
    for j in group_order[1:]:
        global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                        global_weights,
                                                                                        sigma_inv_layer[j],
                                                                                        global_sigmas, prior_mean_norm,
                                                                                        sigma_inv_prior,
                                                                                        popularity_counts, gamma, J, KL_reg,
                                                                                        unlimi)
        assignment[j] = assignment_j

    ## Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order:  # random_order:
            to_delete = []
            ## Remove j
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj), assignment[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                print('Warning - weird unmatching')
                else:
                    global_weights[i] = global_weights[i] - batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]

            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)

            ## Match j
            global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                            global_weights,
                                                                                            sigma_inv_layer[j],
                                                                                            global_sigmas,
                                                                                            prior_mean_norm,
                                                                                            sigma_inv_prior,
                                                                                            popularity_counts, gamma, J, KL_reg,
                                                                                            unlimi)
            assignment[j] = assignment_j

    print('Number of global neurons is %d, gamma %f' % (global_weights.shape[0], gamma))

    return assignment, global_weights, global_sigmas


def layer_group_descent(batch_weights, batch_frequencies, sigma_layers, sigma0_layers, gamma_layers, it, 
                        KL_reg=0, unlimi=False, use_freq=False):
    """
    batch_frequencies: [n_nets, n_classes, freqs]
    batch_weights: [n_nets, n_layers(weight,bias), dim(D,1)]
    it: the number of matching iterations
    """
    #pdb.set_trace()

    n_layers = int(len(batch_weights[0]) / 2)

    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    last_layer_const = []
    total_freq = sum(batch_frequencies)
    for f in batch_frequencies:
        last_layer_const.append(f / total_freq)

    J = len(batch_weights)
    D = batch_weights[0][0].shape[0]
    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None

    ## Group descent for layer
    for c in range(1, n_layers)[::-1]:
        sigma = sigma_layers[c - 1]
        sigma_bias = sigma_bias_layers[c - 1]
        gamma = gamma_layers[c - 1]
        sigma0 = sigma0_layers[c - 1]
        sigma0_bias = sigma0_bias_layers[c - 1]
        if c == (n_layers - 1) and n_layers > 2:
            # weights_bias: [J, (bias+weight)]
            weights_bias = [np.hstack((batch_weights[j][c * 2 - 1].reshape(-1, 1), batch_weights[j][c * 2])) for j in
                            range(J)]
            sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
            mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            if use_freq:
                sigma_inv_layer = [np.array([1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
            else:
                sigma_inv_layer = [np.array([1 / sigma_bias] + [1 / sigma for y in last_layer_const[j]]) for j in range(J)]
        elif c > 1:
            weights_bias = [np.hstack((batch_weights[j][c * 2 - 1].reshape(-1, 1),
                                       patch_weights(batch_weights[j][c * 2], L_next, assignment_c[j]))) for j in
                            range(J)]
            sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
            mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in
                               range(J)]
        else:
            weights_bias = [np.hstack((batch_weights[j][0].T, batch_weights[j][c * 2 - 1].reshape(-1, 1),
                                       patch_weights(batch_weights[j][c * 2], L_next, assignment_c[j]))) for j in
                            range(J)]
            sigma_inv_prior = np.array(
                D * [1 / sigma0] + [1 / sigma0_bias] + (weights_bias[0].shape[1] - 1 - D) * [1 / sigma0])
            mean_prior = np.array(D * [mu0] + [mu0_bias] + (weights_bias[0].shape[1] - 1 - D) * [mu0])
            if n_layers == 2:
                if use_freq:
                    sigma_inv_layer = [
                        np.array(D * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in
                        range(J)]
                else:
                    sigma_inv_layer = [
                        np.array(D * [1 / sigma] + [1 / sigma_bias] + [1 / sigma for y in last_layer_const[j]]) for j in
                        range(J)]
            else:
                sigma_inv_layer = [
                    np.array(D * [1 / sigma] + [1 / sigma_bias] + (weights_bias[j].shape[1] - 1 - D) * [1 / sigma]) for
                    j in range(J)]

        assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                      sigma_inv_prior, gamma, it, KL_reg, unlimi)
        L_next = global_weights_c.shape[0]

        if c == (n_layers - 1) and n_layers > 2:
            softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
            global_weights_out = [global_weights_c[:, 0], global_weights_c[:, 1:], softmax_bias]
            global_inv_sigmas_out = [global_sigmas_c[:, 0], global_sigmas_c[:, 1:], softmax_inv_sigma]
        elif c > 1:
            global_weights_out = [global_weights_c[:, 0], global_weights_c[:, 1:]] + global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, 0], global_sigmas_c[:, 1:]] + global_inv_sigmas_out
        else:
            if n_layers == 2:
                softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
                global_weights_out = [softmax_bias]
                global_inv_sigmas_out = [softmax_inv_sigma]
            global_weights_out = [global_weights_c[:, :D].T, global_weights_c[:, D],
                                  global_weights_c[:, (D + 1):]] + global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, :D].T, global_sigmas_c[:, D],
                                     global_sigmas_c[:, (D + 1):]] + global_inv_sigmas_out

    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]

    return map_out

def layer_skip_group_descent(batch_weights, batch_frequencies, layer_meta_data, sigma_layers, sigma0_layers, gamma_layers, it, 
                        KL_reg=0, unlimi=False):
    """
    batch_frequencies: [n_nets, n_classes(freqs)]
    batch_weights: [n_nets, n_layers(weight,bias), dim(D,1)]
    it: the number of matching iterations

    the batch_weights has been processed by catenate weight and bias
    """
    #pdb.set_trace()

    n_layers = int(len(batch_weights[0]))

    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    last_layer_const = []
    total_freq = sum(batch_frequencies)
    for f in batch_frequencies:
        last_layer_const.append(f / total_freq)

    J = len(batch_weights)
    D = batch_weights[0][0].shape[0]
    n_classes = batch_weights[0][-1].shape[0]
    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None

    global_weights_out = []
    global_inv_sigmas_out = []
    ## Group descent for layer
    for c in range(1, n_layers):
        sigma = sigma_layers[c - 1]
        sigma_bias = sigma_bias_layers[c - 1]
        gamma = gamma_layers[c - 1]
        sigma0 = sigma0_layers[c - 1]
        sigma0_bias = sigma0_bias_layers[c - 1]

        if c%2 == 0:
            continue

        weights = [np.hstack((batch_weights[j][c-1], batch_weights[j][c].T)) for j in
                    range(J)]
        sigma_inv_prior = np.array(weights[0].shape[1] * [1 / sigma0])
        mean_prior = np.array(weights[0].shape[1] * [mu0])
        sigma_inv_layer = [np.array((weights[j].shape[1]) * [1 / sigma]) for j in
                            range(J)]

        assignment_c, global_weights_c, global_sigmas_c = match_layer(weights, sigma_inv_layer, mean_prior,
                                                                      sigma_inv_prior, gamma, it, KL_reg, unlimi)
        L_next = global_weights_c.shape[0]

        split = layer_meta_data[c-1][1]
        global_weights_out = global_weights_out + [global_weights_c[:, :split], global_weights_c[:, split:].T] 
        global_inv_sigmas_out = global_inv_sigmas_out + [global_sigmas_c[:, :split], global_sigmas_c[:, split:].T]

    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]

    # handle the last layer if it not be cancatenated by it's previous layer
    if n_layers%2 != 0:
        avg_last_layer_weight = np.zeros(batch_weights[0][n_layers-1].shape, dtype=np.float32)
        for i in range(n_classes):
            avg_weight_collector = np.zeros(batch_weights[0][n_layers-1][i].shape, dtype=np.float32)
            for j in range(J):
                avg_weight_collector += last_layer_const[j][i]*batch_weights[j][n_layers-1][i]
            avg_last_layer_weight[i] = avg_weight_collector
        #pdb.set_trace()
        map_out = map_out + [avg_last_layer_weight]

    return map_out