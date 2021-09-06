import numpy as np
from scipy.optimize import linear_sum_assignment
import pdb


def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j):

    match_norms = ((weights_j_l + global_weights) ** 2 / (sigma_inv_j + global_sigmas)).sum(axis=1) - (
                global_weights ** 2 / global_sigmas).sum(axis=1)

    return match_norms


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J):

    Lj = weights_j.shape[0]
    #counts = np.minimum(np.array(popularity_counts), 10)
    counts = np.array(popularity_counts)

    param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    param_cost += np.log(counts / (J - counts))

    ## Nonparametric cost
    L = global_weights.shape[0]
    #max_added = min(Lj, max(700 - L, 1))
    max_added = Lj

    nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    full_cost = np.hstack((param_cost, nonparam_cost))

    return full_cost

def row_param_KL_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                      sum_prior_mean_norm):

    #D = weights_j_l.shape[0]
    #L = global_weights.shape[0]

    #match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / 
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log2

    #match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / 
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log3, 4

    match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / 
                  (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
                  ((global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + (
                  sigma_inv_j / prior_inv_sigma - 1 - np.log(sigma_inv_j / prior_inv_sigma)).sum() # minist_log5


    return match_norms

def compute_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                    popularity_counts, gamma, J):

    #pdb.set_trace()

    D = weights_j.shape[1]
    sum_prior_mean_norm = np.array([k * prior_mean_norm for k in popularity_counts])

    Lj = weights_j.shape[0]
    #counts = np.minimum(np.array(popularity_counts), 10)
    #param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    #param_cost += np.log(counts / (J - counts))

    param_cost = np.array([row_param_KL_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j, prior_mean_norm, 
                          prior_inv_sigma, sum_prior_mean_norm)
                 for l in range(Lj)])

    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    #nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                #prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))
    #cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    #nonparam_cost -= cost_pois
    #nonparam_cost += 2 * np.log(gamma / J)

    #nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log2

    #nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log3, 4

    nonparam_cost = np.outer((((weights_j - prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / (
                            prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + (sigma_inv_j / prior_inv_sigma - 1 - np.log(
                            sigma_inv_j / prior_inv_sigma)).sum(), np.ones(max_added)) #minist_log5


    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def row_param_KL2_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                      sum_prior_mean_norm, popularity_counts):

    #D = weights_j_l.shape[0]
    #L = global_weights.shape[0]

    #match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / 
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log2

    #match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / 
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log3, 4
    counts = np.minimum(np.array(popularity_counts), 10)
    
    match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / 
                  (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) * (counts+1) - (
                  ((global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) * counts + (
                  sigma_inv_j / prior_inv_sigma - 1 - np.log(sigma_inv_j / prior_inv_sigma)).sum() # minist_log5

    #print("The prior_inv_sigma: ", prior_inv_sigma)

    return match_norms

def compute_KL2_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                    popularity_counts, gamma, J):

    pdb.set_trace()

    D = weights_j.shape[1]
    sum_prior_mean_norm = np.array([k * prior_mean_norm for k in popularity_counts])

    Lj = weights_j.shape[0]
    #counts = np.minimum(np.array(popularity_counts), 10)
    #param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    #param_cost += np.log(counts / (J - counts))

    param_cost = np.array([row_param_KL2_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j, prior_mean_norm, 
                          prior_inv_sigma, sum_prior_mean_norm, popularity_counts)
                 for l in range(Lj)])

    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    #nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                #prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))
    #cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    #nonparam_cost -= cost_pois
    #nonparam_cost += 2 * np.log(gamma / J)

    #nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log2

    #nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log3, 4

    nonparam_cost = np.outer((((weights_j - prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / (
                            prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + (sigma_inv_j / prior_inv_sigma - 1 - np.log(
                            sigma_inv_j / prior_inv_sigma)).sum(), np.ones(max_added)) #minist_log5


    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def row_param_KL3_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                      sum_prior_mean_norm, popularity_counts):

    #D = weights_j_l.shape[0]
    #L = global_weights.shape[0]

    #match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / 
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log2

    #match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / 
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log3, 4
    counts = np.minimum(np.array(popularity_counts), 10)
    
    match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / 
                  (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) * (counts+1) - (
                  ((global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) * counts # minist_log5

    #print("The prior_inv_sigma: ", prior_inv_sigma)

    return match_norms

def compute_KL3_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                    popularity_counts, gamma, J):

    #pdb.set_trace()

    D = weights_j.shape[1]
    sum_prior_mean_norm = np.array([k * prior_mean_norm for k in popularity_counts])

    Lj = weights_j.shape[0]
    #counts = np.minimum(np.array(popularity_counts), 10)
    #param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    #param_cost += np.log(counts / (J - counts))

    param_cost = np.array([row_param_KL3_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j, prior_mean_norm, 
                          prior_inv_sigma, sum_prior_mean_norm, popularity_counts)
                 for l in range(Lj)])

    ## Nonparametric cost
    L = global_weights.shape[0]
    max_added = min(Lj, max(700 - L, 1))
    #nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                #prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))
    #cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    #nonparam_cost -= cost_pois
    #nonparam_cost += 2 * np.log(gamma / J)

    #nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log2

    #nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log3, 4

    nonparam_cost = np.outer((((weights_j - prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / (
                            prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1), np.ones(max_added)) #minist_log5


    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def row_param_KL4_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                      sum_prior_mean_norm, popularity_counts):

    #D = weights_j_l.shape[0]
    #L = global_weights.shape[0]

    #match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / 
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log2

    #match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / 
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log3, 4
    #counts = np.minimum(np.array(popularity_counts), 10)
    counts = np.array(popularity_counts)
    
    match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / 
                  (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) * (counts+1) - (
                  ((global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) * counts # minist_log5

    #print("The prior_inv_sigma: ", prior_inv_sigma)

    return match_norms

def compute_KL4_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                    popularity_counts, gamma, J):

    #pdb.set_trace()

    D = weights_j.shape[1]
    sum_prior_mean_norm = np.array([k * prior_mean_norm for k in popularity_counts])

    Lj = weights_j.shape[0]
    #counts = np.minimum(np.array(popularity_counts), 10)
    #param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    #param_cost += np.log(counts / (J - counts))

    param_cost = np.array([row_param_KL3_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j, prior_mean_norm, 
                          prior_inv_sigma, sum_prior_mean_norm, popularity_counts)
                 for l in range(Lj)])

    ## Nonparametric cost
    L = global_weights.shape[0]
    #max_added = min(Lj, max(700 - L, 1))
    max_added = Lj
    #nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                #prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))
    #cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    #nonparam_cost -= cost_pois
    #nonparam_cost += 2 * np.log(gamma / J)

    #nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log2

    #nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log3, 4

    nonparam_cost = np.outer((((weights_j - prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / (
                            prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1), np.ones(max_added)) #minist_log5


    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost