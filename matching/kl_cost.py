import numpy as np
from scipy.optimize import linear_sum_assignment
import pdb


def row_param_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j, fix_coff=0):

    fix_norms =  - (weights_j_l ** 2).sum()

    match_norms = ((weights_j_l + global_weights) ** 2 / (sigma_inv_j + global_sigmas)).sum(axis=1) - (
                global_weights ** 2 / global_sigmas).sum(axis=1) + fix_coff * fix_norms

    return match_norms


def compute_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J, coff=1, fix_coff=0):

    '''
    coff: adjust the norm and the log
    '''

    Lj = weights_j.shape[0]
    #counts = np.minimum(np.array(popularity_counts), 10)
    counts = np.array(popularity_counts)

    param_cost = coff * np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    param_cost += np.log(counts / (J - counts))

    ## Nonparametric cost
    L = global_weights.shape[0]
    #max_added = min(Lj, max(700 - L, 1))
    max_added = Lj

    nonparam_cost = coff * np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
                prior_mean_norm ** 2 / prior_inv_sigma).sum() - fix_coff*(weights_j ** 2).sum(axis=1)), np.ones(max_added))
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


def row_param_KL5_cost(global_weights, weights_j_l, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                       sum_prior_mean_norm, popularity_counts):
    """
    This is the inverse KL cost: KL(p(w) || p(\theta))
    original is: KL(p(\theta) || p(w))
    """
    # D = weights_j_l.shape[0]
    # L = global_weights.shape[0]

    # match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 /
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log2

    # match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 /
    #              (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) - (
    #              ((global_weights - sum_prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / global_sigmas ** 2).sum(axis=1) + D * (
    #              sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(prior_inv_sigma[0] / sigma_inv_j[0])) # minist_log3, 4
    counts = np.minimum(np.array(popularity_counts), 10)
    # counts = np.array(popularity_counts)

    match_norms = (((weights_j_l - prior_mean_norm + global_weights - sum_prior_mean_norm) * np.sqrt(
        prior_inv_sigma)) ** 2 /
                   (sigma_inv_j + global_sigmas) ** 2).sum(axis=1) * (counts + 1) - (
                          ((global_weights - sum_prior_mean_norm) * np.sqrt(
                              prior_inv_sigma)) ** 2 / global_sigmas ** 2).sum(axis=1) * counts  # minist_log5

    # print("The prior_inv_sigma: ", prior_inv_sigma)

    return match_norms


def compute_KL5_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                     popularity_counts, gamma, J):
    """
        This is the inverse KL cost: KL(p(w) || p(\theta))
        original is: KL(p(\theta) || p(w))
        """
    # pdb.set_trace()

    D = weights_j.shape[1]
    sum_prior_mean_norm = np.array([k * prior_mean_norm for k in popularity_counts])

    Lj = weights_j.shape[0]
    # counts = np.minimum(np.array(popularity_counts), 10)
    # param_cost = np.array([row_param_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j) for l in range(Lj)])
    # param_cost += np.log(counts / (J - counts))

    param_cost = np.array([row_param_KL5_cost(global_weights, weights_j[l], global_sigmas, sigma_inv_j, prior_mean_norm,
                                              prior_inv_sigma, sum_prior_mean_norm, popularity_counts)
                           for l in range(Lj)])

    ## Nonparametric cost
    L = global_weights.shape[0]
    # max_added = min(Lj, max(700 - L, 1))
    max_added = Lj
    # nonparam_cost = np.outer((((weights_j + prior_mean_norm) ** 2 / (prior_inv_sigma + sigma_inv_j)).sum(axis=1) - (
    # prior_mean_norm ** 2 / prior_inv_sigma).sum()), np.ones(max_added))
    # cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    # nonparam_cost -= cost_pois
    # nonparam_cost += 2 * np.log(gamma / J)

    # nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log2

    # nonparam_cost = np.outer((((weights_j - prior_mean_norm) / np.sqrt(sigma_inv_j)) ** 2 / (
    #                        prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1) + D * (sigma_inv_j[0] / prior_inv_sigma[0] - 1 + np.log(
    #                        prior_inv_sigma[0] / sigma_inv_j[0])), np.ones(max_added)) #minist_log3, 4

    nonparam_cost = np.outer((((weights_j - prior_mean_norm) * np.sqrt(sigma_inv_j)) ** 2 / (
            prior_inv_sigma + sigma_inv_j) ** 2).sum(axis=1), np.ones(max_added))  # minist_log5

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost


def hyperparameters(global_atoms, global_atoms_squared, popularity_counts):
    popularity_counts = np.copy(popularity_counts)
    mean_atoms = global_atoms / popularity_counts.reshape(-1, 1)
    mu0 = mean_atoms.mean(axis=0)
    sigma = global_atoms_squared - (global_atoms ** 2) / popularity_counts.reshape(-1, 1)
    sigma = sigma.sum(axis=0) / (popularity_counts.sum() - len(popularity_counts))
    sigma = np.maximum(sigma, 1e-10)
    sigma0 = ((mean_atoms - mu0) ** 2).mean(axis=0)
    sigma0 = sigma0 - sigma * ((1 / popularity_counts).sum()) / len(popularity_counts)
    sigma0 = np.maximum(sigma0, 1e-10)
    return mu0, sigma, sigma0

def SPAHM_cost(global_atoms, atoms_j, sigma, mu0, sigma0, popularity_counts, gamma, J):
    """
    There are some mistakes in original compute_cost function, so I here to modify it
    """
    D = sigma.shape[0]
    Lj = atoms_j.shape[0]
    counts = np.array(popularity_counts)
    sigma_ratio = sigma0 / sigma
    denum_match = np.outer(counts + 1, sigma0) + sigma
    param_cost = []
    for l in range(Lj):
        cost_match = ((sigma_ratio * (atoms_j[l] + global_atoms) ** 2 + 2 * mu0 * (
                    atoms_j[l] + global_atoms)) / denum_match).sum(axis=1)
        param_cost.append(cost_match)
    denum_no_match = np.outer(counts, sigma0) + sigma
    cost_no_match = ((sigma_ratio * (global_atoms) ** 2 + 2 * mu0 * (global_atoms)) / denum_no_match).sum(axis=1)
    # here we shouldn't sum them, only obtain one dimension in axis-1 because the sigma have been extended by dimension D
    # @@doubts: from the cost form in the paper, the sigma_cost should be in [:, 0], but this will cause the number of atoms in
    # global model explodes, when I convert into the .sum(axis=1) form, this will not happen
    # sigma_cost = (np.log(denum_no_match) - np.log(denum_match))[:, 0]
    sigma_cost = (np.log(denum_no_match) - np.log(denum_match)).sum(axis=1)
    # here the mu_cost needs multiple sigma_ratio inverse, and the order is match - no_match
    mu_cost = (np.outer(counts + 1, mu0 ** 2 / sigma_ratio) / denum_match - np.outer(counts, mu0 ** 2 / sigma_ratio) / denum_no_match).sum(axis=1)
    # counts = np.minimum(counts, 10)  # truncation of prior counts influence
    counts = counts
    param_cost = np.array(param_cost) - cost_no_match + sigma_cost + mu_cost + 2 * np.log(counts / (J - counts))

    ## Nonparametric cost

    L = global_atoms.shape[0]
    #     max_added = min(Lj, max(700 - L, 1))
    max_added = Lj
    # here there are some sigma or sigma0 factor needs to multiply
    nonparam_cost = ((sigma0 * sigma_ratio * atoms_j ** 2 + 2 * mu0 * atoms_j * sigma0 - mu0 ** 2 * sigma0) / ((sigma0 + sigma)*sigma0)).sum(axis=1)
    nonparam_cost = np.outer(nonparam_cost, np.ones(max_added))
    cost_pois = 2 * np.log(np.arange(1, max_added + 1))
    nonparam_cost -= cost_pois
    nonparam_cost += 2 * np.log(gamma / J)

    ## sigma penalty
    # here is the same like above, don't need to sum
    # nonparam_cost += np.log(sigma)[0] - np.log(sigma0 + sigma)[0]
    nonparam_cost += np.log(sigma).sum()- np.log(sigma0 + sigma).sum()

    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def SPAHM_KL_cost(global_atoms, atoms_j, sigma, mu0,
                    sigma0, popularity_counts, gamma, J):
    """
    This is the Kullback-leibler divergence between prior distribution and posterior distribution (by adding
    current atoms_j): KL(Prior || Posterior) = F(Posterior) - F(Prior) - <Posterior-Prior, delta(F(Prior))>
    """

    ## parametric cost
    Lj = atoms_j.shape[0]
    counts = np.array(popularity_counts)
    sigma_ratio = sigma0 / sigma
    denum_match = np.outer(counts + 1, sigma0) + sigma
    pos_param_cost = []
    for l in range(Lj):
        cost_match = ((sigma_ratio * (atoms_j[l] + global_atoms) ** 2 + 2 * mu0 * (
                    atoms_j[l] + global_atoms)) / denum_match).sum(axis=1)
        pos_param_cost.append(cost_match)
    pos_param_cost = np.array(pos_param_cost) + (np.outer(counts + 1, mu0 ** 2 / sigma_ratio) / denum_match).sum(axis=1)
    pos_param_cost = pos_param_cost - np.log(denum_match).sum(axis=1)

    denum_no_match = np.outer(counts, sigma0) + sigma
    pri_param_cost = ((sigma_ratio * (global_atoms) ** 2 + 2 * mu0 * (global_atoms)) / denum_no_match).sum(axis=1)
    pri_param_cost = pri_param_cost + (np.outer(counts, mu0 ** 2 / sigma_ratio) / denum_no_match - np.log(denum_no_match)).sum(axis=1)
    param_cost = pos_param_cost - pri_param_cost

    # calculate the product param cost
    prod_param_cost = []
    for l in range(Lj):
        cost_match_1 = ((sigma_ratio * (atoms_j[l] + global_atoms) * atoms_j[l] + mu0 * atoms_j[l]) / denum_match).sum(axis=1)
        cost_match_2 = ((sigma0 * sigma_ratio * (atoms_j[l] + global_atoms) ** 2 + 2 * sigma0 * mu0 * (
                         atoms_j[l] + global_atoms) + mu0 ** 2 * sigma) / (2 * denum_match ** 2)).sum(axis=1)
        cost_match = cost_match_1 - cost_match_2
        prod_param_cost.append(cost_match)
    prod_param_cost -= (1 / denum_match).sum(axis=1)

    param_cost -= 2 * prod_param_cost

    ## Nonparametric cost
    L = global_atoms.shape[0]
    #     max_added = min(Lj, max(700 - L, 1))
    max_added = Lj
    pos_nonparam_cost = ((sigma0 * sigma_ratio * atoms_j ** 2 + 2 * mu0 * atoms_j + mu0 ** 2 / sigma_ratio) /
                                (sigma0 + sigma)).sum(axis=1) - np.log(sigma0 + sigma)[0]
    pri_nonparam_cost = (mu0 ** 2 / sigma0).sum() - np.log(sigma).sum()
    nonparam_cost = pos_nonparam_cost - pri_nonparam_cost
    nonparam_cost = np.outer(nonparam_cost, np.ones(max_added))

    # calculate the product nonparm cost
    prod_nonparam_cost = ((sigma_ratio * atoms_j ** 2 + mu0 * atoms_j) / (sigma0 + sigma)).sum(axis=1)
    prod_nonparam_cost -= ((sigma0 * sigma_ratio * atoms_j ** 2 + 2 * sigma0 * mu0 * atoms_j + mu0 ** 2 * sigma) / (2 * (sigma0 + sigma) ** 2)).sum(axis=1)
    prod_nonparam_cost = np.outer(prod_nonparam_cost, np.ones(max_added))
    prod_nonparam_cost -= (1 / (sigma0 + sigma)).sum()

    nonparam_cost -= 2 * prod_nonparam_cost

    ##  the full cost
    full_cost = np.hstack((param_cost, nonparam_cost))
    return full_cost

def nafi_KL_cost(global_weights, weights_j, global_sigmas, sigma_inv_j, prior_mean_norm, prior_inv_sigma,
                 popularity_counts, gamma, J, coff=1, fix_coff=0):

    '''
    nafi_KL_cost: calculate the kullback-leibler divergence between two multi-gaussian distributions N_x(mu_x, Sigma_x)
    and N_y(mu_y, Sigma_y):
    KL(N_x, N_y) = 1/2[tr(Sigma_y^-1 Sigma_x) + (mu_y - mu_x)^T Sigma_y^-1 (mu_y - mu_x) - (D+K) + log(\frac{\det(Sigma_y)}{\det(Sigma_x)})]
    '''

    ## Parametric cost
    Lj = weights_j.shape[0]
    D = weights_j.shape[1]
    #counts = np.minimum(np.array(popularity_counts), 10)

    # param_cost = np.array([(global_sigmas / (global_sigmas+sigma_inv_j)).sum(axis=1) + \
    #             (((global_weights + weights_j[l])/(global_sigmas + sigma_inv_j) - global_weights/global_sigmas)**2 / (global_sigmas + sigma_inv_j)).sum(axis=1) +\
    #             np.log(np.prod(global_sigmas+sigma_inv_j, axis=1)/np.prod(global_sigmas, axis=1)) for l in range(Lj)])
    param_cost = np.array([(((global_weights + weights_j[l]) / (
                                       global_sigmas + sigma_inv_j) - global_weights / global_sigmas) ** 2 / (
                                        global_sigmas + sigma_inv_j)).sum(axis=1) for l
                           in range(Lj)])

    ## Nonparametric cost
    #max_added = min(Lj, max(700 - L, 1))
    max_added = Lj

    # nonparam_cost = np.outer(((1 / prior_inv_sigma / (prior_inv_sigma + sigma_inv_j)).sum() + \
    #                       (((prior_mean_norm + weights_j) / (
    #                                   prior_inv_sigma + sigma_inv_j) - prior_mean_norm / prior_inv_sigma) ** 2 / (
    #                                    prior_inv_sigma + sigma_inv_j)).sum(axis=1) + \
    #                       np.log(np.prod(prior_inv_sigma + sigma_inv_j) / np.prod(1 / prior_inv_sigma))), np.ones(max_added))
    nonparam_cost = np.outer(((((prior_mean_norm + weights_j) / (
                                      prior_inv_sigma + sigma_inv_j) - prior_mean_norm / prior_inv_sigma) ** 2 / (
                                       prior_inv_sigma + sigma_inv_j)).sum(axis=1)),
                             np.ones(max_added))
    full_cost = np.hstack((param_cost, nonparam_cost))

    return full_cost

