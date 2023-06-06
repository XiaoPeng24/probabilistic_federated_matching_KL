import numpy as np
from scipy.optimize import linear_sum_assignment
# from lapsolver import solve_dense
import pdb

from matching.kl_cost import compute_KL3_cost, compute_KL2_cost
from matching.kl_cost import compute_cost as unlimi_cost, compute_KL4_cost as unlimi_KL_cost
from matching.kl_cost import compute_KL5_cost as unlimi_KL_inverse_cost
from matching.kl_cost import SPAHM_cost, SPAHM_KL_cost, hyperparameters
from matching.kl_cost import nafi_KL_cost
from matching.self_info_cost import self_info_cost, self_info2_cost as unlimi_self_info_cost


def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j, fix_coff=0):

    fix_norms =  - (weights_j_l ** 2).sum(axis=1)

    match_norms = ((weights_j_l + global_weights) ** 2 / (sigma_inv_j + global_sigmas)).sum(axis=1) - (
                global_weights ** 2 / global_sigmas).sum(axis=1) + fix_coff * fix_norms

    return match_norms


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J, coff=1, fix_coff=0):

    #pdb.set_trace()

    Lj = weights_j.shape[0]
    counts = np.minimum(np.array(popularity_counts), 10)
    param_cost = coff * np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, 
                                 sigma_inv_j, fix_coff=fix_coff) for l in range(Lj)])
    param_cost += np.log(counts / (J - counts))

    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = coff * np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum() - fix_coff*(weights_j ** 2).sum(axis=1)), np.ones(max_added))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost


def matching_upd_j(weights_j, global_weights,
                   sigma_inv_j, global_sigmas, prior_mean_norm, prior_inv_sigma,
                   popularity_counts, gamma, J, KL_reg=0, I_reg=0, coff=1, fix_coff=0, unlimi=False, SPAHM=False, nafi=False):

    L = global_weights.shape[0]

    if unlimi:
        if SPAHM:
            full_cost = SPAHM_cost(global_weights, weights_j, sigma_inv_j, prior_mean_norm,
                                    prior_inv_sigma,
                                    popularity_counts, gamma, J)
        else:
            full_cost = unlimi_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J, coff, fix_coff)
    else:
        full_cost = compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J, coff, fix_coff)

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
            if SPAHM:
                KL_cost = KL_reg*SPAHM_KL_cost(global_weights, weights_j, sigma_inv_j, prior_mean_norm,
                                       prior_inv_sigma,
                                       popularity_counts, gamma, J)
            else:
                KL_cost = KL_reg*unlimi_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)
        else:
            KL_cost = KL_reg*compute_KL2_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)

    nafi_cost = 0

    if KL_reg!=0 and nafi:
        nafi_cost = KL_reg*nafi_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                                        popularity_counts, gamma, J)

    I_cost = 0

    if I_reg!=0:
        if unlimi:
            I_cost = I_reg*unlimi_self_info_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)
        else:
            I_cost = I_reg*self_info_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                             popularity_counts, gamma, J)

    row_ind, col_ind = linear_sum_assignment(-full_cost+KL_cost+I_cost+nafi_cost)
    # row_ind, col_ind = solve_dense(-full_cost+KL_cost+I_cost)

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


def matching_upd_j_SPAHM(atoms_j, global_atoms, global_atoms_squared, sigma, sigma0, mu0, popularity_counts, gamma, J,
                   KL_reg=0, unlimi=False):
    L = global_atoms.shape[0]

    full_cost = SPAHM_cost(global_atoms, atoms_j, sigma, mu0, sigma0, popularity_counts, gamma, J)

    KL_cost = 0

    if KL_reg != 0:
        KL_cost = KL_reg * SPAHM_KL_cost(global_atoms, atoms_j, sigma, mu0,
                    sigma0, popularity_counts, gamma, J)

    row_ind, col_ind = linear_sum_assignment(-full_cost + KL_cost)
    # row_ind, col_ind = solve_dense(-full_cost+KL_cost+I_cost)

    assignment_j = []

    new_L = L

    for l, i in zip(row_ind, col_ind):
        if i < L:
            popularity_counts[i] += 1
            assignment_j.append(i)
            global_atoms[i] += atoms_j[l]
            global_atoms_squared[i] += atoms_j[l] ** 2
        else:  # new neuron
            popularity_counts += [1]
            assignment_j.append(new_L)
            new_L += 1
            global_atoms = np.vstack((global_atoms, atoms_j[l]))
            global_atoms_squared = np.vstack((global_atoms_squared, atoms_j[l] ** 2))

    return global_atoms, global_atoms_squared, popularity_counts, assignment_j

def objective(global_weights, global_sigmas):
    obj = ((global_weights) ** 2 / global_sigmas).sum()
    return obj


def patch_weights(w_j, L_next, assignment_j_c):
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    return new_w_j


def patch_weights(w_j, L_next, assignment_j_c):
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    return new_w_j


def block_patching(w_j, L_next, assignment_j_c, layer_index, model_meta_data,
                   matching_shapes=None,
                   layer_type="fc",
                   dataset="cifar10",
                   network_name="lenet"):
    """
    In CNN, weights patching needs to be handled block-wisely
    We handle all conv layers and the first fc layer connected with the output of conv layers here
    """
    # logger.info('--'*15)
    # logger.info("ori w_j shape: {}".format(w_j.shape))
    # logger.info("L_next: {}".format(L_next))
    # logger.info("assignment_j_c: {}, length of assignment: {}".format(assignment_j_c, len(assignment_j_c)))
    # logger.info("correspoding meta data: {}".format(model_meta_data[2 * layer_index - 2]))
    # logger.info("layer index: {}".format(layer_index))
    # logger.info('--'*15)
    if assignment_j_c is None:
        return w_j

    layer_meta_data = model_meta_data[2 * layer_index - 2]
    prev_layer_meta_data = model_meta_data[2 * layer_index - 2 - 2]

    if layer_type == "conv":
        new_w_j = np.zeros((w_j.shape[0], L_next * (layer_meta_data[-1] ** 2)))

        # we generate a sequence of block indices
        block_indices = [np.arange(i * layer_meta_data[-1] ** 2, (i + 1) * layer_meta_data[-1] ** 2) for i in
                         range(L_next)]
        ori_block_indices = [np.arange(i * layer_meta_data[-1] ** 2, (i + 1) * layer_meta_data[-1] ** 2) for i in
                             range(layer_meta_data[1])]
        for ori_id in range(layer_meta_data[1]):
            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]

    elif layer_type == "fc":
        # we need to estimate the output shape here:
        if network_name == "simple-cnn":
            if dataset in ("cifar10", "cinic10"):
                shape_estimator = SimpleCNNContainerConvBlocks(input_channel=3, num_filters=matching_shapes,
                                                               kernel_size=5)
            elif dataset == "mnist":
                shape_estimator = SimpleCNNContainerConvBlocks(input_channel=1, num_filters=matching_shapes,
                                                               kernel_size=5)
        elif network_name == "moderate-cnn":
            if dataset in ("cifar10", "cinic10"):
                shape_estimator = ModerateCNNContainerConvBlocks(num_filters=matching_shapes)
            elif dataset == "mnist":
                shape_estimator = ModerateCNNContainerConvBlocksMNIST(num_filters=matching_shapes)
        elif network_name == "lenet":
            shape_estimator = LeNetContainer(num_filters=matching_shapes, kernel_size=5)

        if dataset in ("cifar10", "cinic10"):
            dummy_input = torch.rand(1, 3, 32, 32)
        elif dataset == "mnist":
            dummy_input = torch.rand(1, 1, 28, 28)
        estimated_output = shape_estimator(dummy_input)
        new_w_j = np.zeros((w_j.shape[0], estimated_output.view(-1).size()[0]))
        logger.info("estimated_output shape : {}".format(estimated_output.size()))
        logger.info("meta data of previous layer: {}".format(prev_layer_meta_data))

        block_indices = [np.arange(i * estimated_output.size()[-1] ** 2, (i + 1) * estimated_output.size()[-1] ** 2) for
                         i in range(L_next)]
        # for i, bid in enumerate(block_indices):
        #    logger.info("{}, {}".format(i, bid))
        # logger.info("**"*20)
        ori_block_indices = [np.arange(i * estimated_output.size()[-1] ** 2, (i + 1) * estimated_output.size()[-1] ** 2)
                             for i in range(prev_layer_meta_data[0])]
        # for i, obid in enumerate(ori_block_indices):
        #    logger.info("{}, {}".format(i, obid))
        # logger.info("assignment c: {}".format(assignment_j_c))
        for ori_id in range(prev_layer_meta_data[0]):
            # logger.info("{} ------------ to ------------ {}".format(block_indices[assignment_j_c[ori_id]], ori_block_indices[ori_id]))
            new_w_j[:, block_indices[assignment_j_c[ori_id]]] = w_j[:, ori_block_indices[ori_id]]

        # logger.info("mapped block id: {}, ori block id: {}".format(block_indices[assignment_j_c[ori_id]], ori_block_indices[ori_id]))
    # do a double check logger.infoing here:
    # logger.info("{}".format(np.array_equal(new_w_j[:, block_indices[4]], w_j[:, ori_block_indices[0]])))
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
                I_reg=0, coff=1, fix_coff=0, unlimi=False, SPAHM=False, nafi=False):
    """
    weight_bias: [J, np.array(n_neurons, dim)]
    sigma_inv_layer: [J, dim]
    mean_prior: [dim]
    sigma_inv_prior: [dim]
    Modified in April 28th, 2022: combine it with the SPAHM
    """
    #pdb.set_trace()
    
    J = len(weights_bias)
    D = weights_bias[0].shape[1]

    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])

    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    assignment = [[] for _ in range(J)]
    # if SPAHM:
    #     sigma = np.ones(D) *  (1 / sigma_inv_layer[0][0])
    #     sigma0 = np.ones(D) * (1 / sigma_inv_prior[0])
    #     total_atoms = sum([weight_j.shape[0] for weight_j in weights_bias])
    #     mu0 = sum([weight_j.sum(axis=0) for weight_j in weights_bias]) / total_atoms
    #     print('Init mu0 estimate mean is %f' % (mu0.mean()))
    #     global_atoms = np.copy(weights_bias[group_order[0]])
    #     global_atoms_squared = np.copy(weights_bias[group_order[0]] ** 2)
    #
    #     popularity_counts = [1] * global_atoms.shape[0]
    #     assignment[group_order[0]] = list(range(global_atoms.shape[0]))
    # else:
    prior_mean_norm = mean_prior * sigma_inv_prior
    global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
    global_sigmas = np.outer(np.ones(global_weights.shape[0]), sigma_inv_prior + sigma_inv_layer[group_order[0]])

    popularity_counts = [1] * global_weights.shape[0]

    assignment[group_order[0]] = list(range(global_weights.shape[0]))

    ## Initialize
    for j in group_order[1:]:
        # if SPAHM:
        #     global_atoms, global_atoms_squared, popularity_counts, assignment_j = matching_upd_j_SPAHM(weights_bias[j],
        #                                                                                                global_atoms,
        #                                                                                                global_atoms_squared,
        #                                                                                                sigma,
        #                                                                                                sigma0,
        #                                                                                                mu0,
        #                                                                                                popularity_counts, gamma, J,
        #                                                                                                KL_reg, unlimi)
        #     assignment[j] = assignment_j
        #
        #     # mu0, sigma, sigma0 = hyperparameters(global_atoms, global_atoms_squared, popularity_counts)
        #     print('Init Sigma mean estimate is %f; sigma0 is %f; mu0 is %f' % (sigma.mean(), sigma0.mean(), mu0.mean()))
        # else:
        global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                        global_weights,
                                                                                        sigma_inv_layer[j],
                                                                                        global_sigmas, prior_mean_norm,
                                                                                        sigma_inv_prior,
                                                                                        popularity_counts, gamma, J, KL_reg,
                                                                                        I_reg, coff, fix_coff, unlimi, SPAHM, nafi)
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
                    # if SPAHM:
                    #     global_atoms[i] = global_atoms[i] - weights_bias[j][l]
                    #     global_atoms_squared[i] = global_atoms_squared[i] - weights_bias[j][l] ** 2
                    # else:
                    global_weights[i] = global_weights[i] - batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]

            # if SPAHM:
            #     global_atoms = np.delete(global_atoms, to_delete, axis=0)
            #     global_atoms_squared = np.delete(global_atoms_squared, to_delete, axis=0)
            # else:
            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)

            ## Match j
            # if SPAHM:
            #     global_atoms, global_atoms_squared, popularity_counts, assignment_j = matching_upd_j_SPAHM(weights_bias[j],
            #                                                                                          global_atoms,
            #                                                                                          global_atoms_squared,
            #                                                                                          sigma, sigma0,
            #                                                                                          mu0,
            #                                                                                          popularity_counts,
            #                                                                                          gamma, J,
            #                                                                                          KL_reg,unlimi)
            #     assignment[j] = assignment_j
            #
            #     # mu0, sigma, sigma0 = hyperparameters(global_atoms, global_atoms_squared, popularity_counts)
            #     print('Sigma mean estimate is %f; sigma0 is %f; mu0 is %f' % (sigma.mean(), sigma0.mean(), mu0.mean()))
            # else:
            global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                            global_weights,
                                                                                            sigma_inv_layer[j],
                                                                                            global_sigmas,
                                                                                            prior_mean_norm,
                                                                                            sigma_inv_prior,
                                                                                            popularity_counts, gamma, J, KL_reg,
                                                                                            I_reg, coff, fix_coff, unlimi, SPAHM, nafi)
            assignment[j] = assignment_j

    # if SPAHM:
    #     print('Number of global neurons is %d, gamma %f' % (global_atoms.shape[0], gamma))
    #     return  assignment, mu0 * sigma + global_atoms * sigma0, np.outer(popularity_counts, sigma0) + sigma
    # else:
    print('Number of global neurons is %d, gamma %f' % (global_weights.shape[0], gamma))

    return assignment, global_weights, global_sigmas


def layer_group_descent(batch_weights, batch_frequencies, sigma_layers, sigma0_layers, gamma_layers, it, 
                        KL_reg=0, I_reg=0, coff=1, fix_coff=0, unlimi=False, use_freq=False, SPAHM=False, nafi=False):
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
    assignments = [None for c in range(n_layers)]
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
                                                                      sigma_inv_prior, gamma, it, KL_reg, I_reg, 
                                                                      coff, fix_coff, unlimi, SPAHM, nafi)
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

        assignments[c] = assignment_c

    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]

    return map_out, assignments


def layer_wise_group_descent(batch_weights, layer_index, batch_frequencies, sigma_layers,
                             sigma0_layers, gamma_layers, it,
                             model_meta_data,
                             model_layer_type,
                             n_layers,
                             matching_shapes,
                             args, KL_reg=0, I_reg=0, coff=1, fix_coff=0, unlimi=False):
    """
    We implement a layer-wise matching here:
    """
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

    # J: number of workers
    J = len(batch_weights)
    # init_num_kernel: the number of conv filters in the first conv layer
    init_num_kernel = batch_weights[0][0].shape[0]

    # for saving (#channel * k * k)
    init_channel_kernel_dims = []
    for bw in batch_weights[0]:
        if len(bw.shape) > 1:
            init_channel_kernel_dims.append(bw.shape[1])
    logger.info("init_channel_kernel_dims: {}".format(init_channel_kernel_dims))

    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None

    sigma = sigma_layers[layer_index - 1]
    sigma_bias = sigma_bias_layers[layer_index - 1]
    gamma = gamma_layers[layer_index - 1]
    sigma0 = sigma0_layers[layer_index - 1]
    sigma0_bias = sigma0_bias_layers[layer_index - 1]

    if layer_index <= 1:
        weights_bias = [np.hstack((batch_weights[j][0], batch_weights[j][layer_index * 2 - 1].reshape(-1, 1))) for j in
                        range(J)]

        sigma_inv_prior = np.array(
            init_channel_kernel_dims[layer_index - 1] * [1 / sigma0] + [1 / sigma0_bias])
        mean_prior = np.array(init_channel_kernel_dims[layer_index - 1] * [mu0] + [mu0_bias])

        # handling 2-layer neural network
        if n_layers == 2:
            sigma_inv_layer = [
                np.array(D * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in
                range(J)]
        else:
            sigma_inv_layer = [np.array(init_channel_kernel_dims[layer_index - 1] * [1 / sigma] + [1 / sigma_bias]) for
                               j in range(J)]

    elif layer_index == (n_layers - 1) and n_layers > 2:
        # our assumption is that this branch will consistently handle the last fc layers
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and (
                    'conv' in prev_layer_type or 'features' in layer_type))

        # if first_fc_identifier:
        #     weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T,
        #                                 batch_weights[j][2 * layer_index - 1].reshape(-1, 1),
        #                                 batch_weights[j][2 * layer_index])) for j in range(J)]
        # else:
        #     weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T,
        #                                 batch_weights[j][2 * layer_index - 1].reshape(-1, 1),
        #                                 batch_weights[j][2 * layer_index])) for j in range(J)]

        # we switch to ignore the last layer here:
        if first_fc_identifier:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T,
                                       batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]
        else:
            weights_bias = [np.hstack((batch_weights[j][2 * layer_index - 2].T,
                                       batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for j in range(J)]

        sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
        mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])

        # hwang: this needs to be handled carefully
        # sigma_inv_layer = [np.array([1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
        # sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in range(J)]

        # sigma_inv_layer = [np.array((matching_shapes[layer_index - 2]) * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]

        # sigma_inv_layer = [np.array((matching_shapes[layer_index - 2]) * [1 / sigma] + [1 / sigma_bias]) for j in range(J)]
        sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in range(J)]

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]

        if 'conv' in layer_type or 'features' in layer_type:
            weights_bias = [
                np.hstack((batch_weights[j][2 * layer_index - 2], batch_weights[j][2 * layer_index - 1].reshape(-1, 1)))
                for j in range(J)]

        elif 'fc' in layer_type or 'classifier' in layer_type:
            # we need to determine if the type of the current layer is the same as it's previous layer
            # i.e. we need to identify if the fully connected layer we're working on is the first fc layer after the conv block
            # first_fc_identifier = ('fc' in layer_type and 'conv' in prev_layer_type)
            first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and (
                        'conv' in prev_layer_type or 'features' in layer_type))
            # logger.info("first_fc_identifier: {}".format(first_fc_identifier))
            if first_fc_identifier:
                weights_bias = [np.hstack(
                    (batch_weights[j][2 * layer_index - 2].T, batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for
                                j in range(J)]
            else:
                weights_bias = [np.hstack(
                    (batch_weights[j][2 * layer_index - 2].T, batch_weights[j][2 * layer_index - 1].reshape(-1, 1))) for
                                j in range(J)]

        sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
        mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
        sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in range(J)]

    logger.info("Layer index: {}, init_num_kernel: {}".format(layer_index, init_num_kernel))
    logger.info("weights bias: {}".format(weights_bias[0].shape))
    logger.info("sigma_inv_prior shape: {}".format(sigma_inv_prior.shape))
    logger.info("mean_prior shape: {}".format(mean_prior.shape))
    logger.info("sigma_inv_layer shape: {}".format(sigma_inv_layer[0].shape))

    assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                      sigma_inv_prior, gamma, it, KL_reg, I_reg,
                                                                      coff, fix_coff, unlimi)


    L_next = global_weights_c.shape[0]

    if layer_index <= 1:
        if n_layers == 2:
            softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
            global_weights_out = [softmax_bias]
            global_inv_sigmas_out = [softmax_inv_sigma]

        global_weights_out = [global_weights_c[:, :init_channel_kernel_dims[int(layer_index / 2)]],
                              global_weights_c[:, init_channel_kernel_dims[int(layer_index / 2)]]]
        global_inv_sigmas_out = [global_sigmas_c[:, :init_channel_kernel_dims[int(layer_index / 2)]],
                                 global_sigmas_c[:, init_channel_kernel_dims[int(layer_index / 2)]]]

        logger.info("Branch A, Layer index: {}, Global weights out shapes: {}".format(layer_index, [gwo.shape for gwo in
                                                                                                    global_weights_out]))

    elif layer_index == (n_layers - 1) and n_layers > 2:
        softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)

        layer_type = model_layer_type[2 * layer_index - 2]
        prev_layer_type = model_layer_type[2 * layer_index - 2 - 2]
        # first_fc_identifier = ('fc' in layer_type and 'conv' in prev_layer_type)
        first_fc_identifier = (('fc' in layer_type or 'classifier' in layer_type) and (
                    'conv' in prev_layer_type or 'features' in layer_type))

        # if first_fc_identifier:
        #     global_weights_out = [global_weights_c[:, 0:-softmax_bias.shape[0]-1].T,
        #                             global_weights_c[:, -softmax_bias.shape[0]-1],
        #                             global_weights_c[:, -softmax_bias.shape[0]:],
        #                             softmax_bias]

        #     global_inv_sigmas_out = [global_sigmas_c[:, 0:-softmax_bias.shape[0]-1].T,
        #                                 global_sigmas_c[:, -softmax_bias.shape[0]-1],
        #                                 global_sigmas_c[:, -softmax_bias.shape[0]:],
        #                                 softmax_inv_sigma]
        # else:
        #     global_weights_out = [global_weights_c[:, 0:matching_shapes[layer_index - 1 - 1]].T,
        #                             global_weights_c[:, matching_shapes[layer_index - 1 - 1]],
        #                             global_weights_c[:, matching_shapes[layer_index - 1 - 1]+1:],
        #                             softmax_bias]

        #     global_inv_sigmas_out = [global_sigmas_c[:, 0:matching_shapes[layer_index - 1 - 1]].T,
        #                                 global_sigmas_c[:, matching_shapes[layer_index - 1 - 1]],
        #                                 global_sigmas_c[:, matching_shapes[layer_index - 1 - 1]+1:],
        #                                 softmax_inv_sigma]

        # remove fitting the last layer
        # if first_fc_identifier:
        #     global_weights_out = [global_weights_c[:, 0:-softmax_bias.shape[0]-1].T,
        #                             global_weights_c[:, -softmax_bias.shape[0]-1]]

        #     global_inv_sigmas_out = [global_sigmas_c[:, 0:-softmax_bias.shape[0]-1].T,
        #                                 global_sigmas_c[:, -softmax_bias.shape[0]-1]]
        # else:
        #     global_weights_out = [global_weights_c[:, 0:matching_shapes[layer_index - 1 - 1]].T,
        #                             global_weights_c[:, matching_shapes[layer_index - 1 - 1]]]

        #     global_inv_sigmas_out = [global_sigmas_c[:, 0:matching_shapes[layer_index - 1 - 1]].T,
        #                                 global_sigmas_c[:, matching_shapes[layer_index - 1 - 1]]]
        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape
        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1] - 1], global_weights_c[:, gwc_shape[1] - 1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1] - 1], global_sigmas_c[:, gwc_shape[1] - 1]]
        elif "fc" in layer_type or 'classifier' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1] - 1].T, global_weights_c[:, gwc_shape[1] - 1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1] - 1].T, global_sigmas_c[:, gwc_shape[1] - 1]]

        logger.info("#### Branch B, Layer index: {}, Global weights out shapes: {}".format(layer_index,
                                                                                           [gwo.shape for gwo in
                                                                                            global_weights_out]))

    elif (layer_index > 1 and layer_index < (n_layers - 1)):
        layer_type = model_layer_type[2 * layer_index - 2]
        gwc_shape = global_weights_c.shape

        if "conv" in layer_type or 'features' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1] - 1], global_weights_c[:, gwc_shape[1] - 1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1] - 1], global_sigmas_c[:, gwc_shape[1] - 1]]
        elif "fc" in layer_type or 'classifier' in layer_type:
            global_weights_out = [global_weights_c[:, 0:gwc_shape[1] - 1].T, global_weights_c[:, gwc_shape[1] - 1]]
            global_inv_sigmas_out = [global_sigmas_c[:, 0:gwc_shape[1] - 1].T, global_sigmas_c[:, gwc_shape[1] - 1]]
        logger.info("Branch layer index, Layer index: {}, Global weights out shapes: {}".format(layer_index,
                                                                                                [gwo.shape for gwo in
                                                                                                 global_weights_out]))

    logger.info("global inv sigma out shape: {}".format([giso.shape for giso in global_inv_sigmas_out]))
    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]
    return map_out, assignment_c, L_next

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