import numpy as np
from scipy.optimize import linear_sum_assignment
import pdb


def row_param_KL_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                       sum_prior_mean_norm, sigma_factor, popularity_counts):

    counts = np.minimum(np.array(popularity_counts), 10)

    match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * sigma_factor) ** 2 /
                   (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) * (counts + 1) - (
                          ((global_weights - sum_prior_mean_norm) * sigma_factor) ** 2 / global_sigmas ** 2).sum(axis=1) * counts  # minist_log5

    # print("The prior_inv_sigma: ", prior_inv_sigma)

    return match_norms


def compute_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                     sigma_factor, popularity_counts, gamma, J):
    """
    :param global_weights:
    :param weights_j:
    :param global_sigmas:
    :param sigma_inv_j:
    :param prior_mean_norm:
    :param prior_inv_sigma:
    :param sigma_inv_factor: the factor of sigma inv in the norm
    :param popularity_counts:
    :param gamma:
    :param J:
    :return:
    """

    # pdb.set_trace()

    sum_prior_mean_norm = np.array([k * prior_mean_norm for k in popularity_counts])

    Lj = weights_j.shape[0]

    param_cost = np.array([row_param_KL_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j, prior_mean_norm,
                                              prior_inv_sigma, sum_prior_mean_norm, sigma_factor, popularity_counts)
                           for l in range(Lj)])

    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))

    nonparam_cost = np.outer((((weights_j - prior_mean_norm) * sigma_factor) ** 2 / (
            prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1), np.ones(max_added))  # minist_log5

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def row_param_KL2_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                       sum_prior_mean_norm, sigma_factor, popularity_counts):

    counts = np.array(popularity_counts)

    #pdb.set_trace()

    match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * sigma_factor) ** 2 /
                   (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) * (counts + 1) - (
                          ((global_weights - sum_prior_mean_norm) * sigma_factor) ** 2 / global_sigmas ** 2).sum(axis=1) * counts  # minist_log5

    # print("The prior_inv_sigma: ", prior_inv_sigma)

    return match_norms


def compute_KL2_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                     sigma_factor, popularity_counts, gamma, J):
    """
    :param global_weights:
    :param weights_j:
    :param global_sigmas:
    :param sigma_inv_j:
    :param prior_mean_norm:
    :param prior_inv_sigma:
    :param sigma_inv_factor: the factor of sigma inv in the norm
    :param popularity_counts:
    :param gamma:
    :param J:
    :return:
    """

    #pdb.set_trace()

    if isinstance(prior_mean_norm, int):
        sum_prior_mean_norm = 0
    else:
        sum_prior_mean_norm = np.array([k * prior_mean_norm for k in popularity_counts])

    Lj = weights_j.shape[0]

    param_cost = np.array([row_param_KL2_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j, prior_mean_norm,
                                              prior_inv_sigma, sum_prior_mean_norm, sigma_factor, popularity_counts)
                           for l in range(Lj)])

    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = Lj

    nonparam_cost = np.outer((((weights_j - prior_mean_norm) * sigma_factor) ** 2 / (
            prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1), np.ones(max_added))  # minist_log5

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j):

    match_norms = ((weights_j_l + global_weights) ** 2 / (sigma_inv_j + global_sigmas)).sum(axis=1) - (
                global_weights ** 2 / global_sigmas).sum(axis=1)

    return match_norms


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    #pdb.set_trace()

    Lj = weights_j.shape[0]
    param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def compute_cost2(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    #pdb.set_trace()

    Lj = weights_j.shape[0]
    param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = Lj
    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def self_info_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    # part 1: two KL cost part, the first part is negative and sigma_inv_factor is np.sqrt(prior_inv_sigma), the second
    #  is positive and sigma_inv_factor is np.sqrt(sigma_inv_j), and the seconde part has no mu_0

    part1_KL_1 = -compute_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                    np.sqrt(prior_inv_sigma), popularity_counts, gamma, J)
    part1_KL_2 = compute_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, 0,
                    prior_inv_sigma, np.sqrt(sigma_inv_j), popularity_counts, gamma, J)


    part1_cost = part1_KL_1 + part1_KL_2

    # part 2: the negative of origin cost multiply by 2

    part2_cost = -2 * compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J)

    full_cost = part1_cost + part2_cost

    return full_cost

def self_info2_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    # part 1: two KL cost part, the first part is negative and sigma_inv_factor is np.sqrt(prior_inv_sigma), the second
    #  is positive and sigma_inv_factor is np.sqrt(sigma_inv_j), and the seconde part has no mu_0

    part1_KL_1 = -compute_KL2_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                    np.sqrt(prior_inv_sigma), popularity_counts, gamma, J)
    part1_KL_2 = compute_KL2_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, 0,
                    prior_inv_sigma, np.sqrt(sigma_inv_j), popularity_counts, gamma, J)


    part1_cost = part1_KL_1 + part1_KL_2

    # part 2: the negative of origin cost multiply by 2

    part2_cost = -2 * compute_cost2(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J)

    full_cost = part1_cost + part2_cost

    return full_cost