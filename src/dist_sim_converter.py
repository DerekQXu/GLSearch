import numpy as np


def get_ds_metric_config(ds_metric):
    if ds_metric == 'ged':
        dist_or_sim = 'dist'
        true_algo = 'astar'
    elif ds_metric == 'mcs':
        dist_or_sim = 'sim'
        true_algo = 'mccreesh2017'
    elif ds_metric == 'random':
        dist_or_sim = 'random'
        true_algo = 'random'
    else:
        raise ValueError('Unknown graph dist/sim metric {}'.format(ds_metric))
    return dist_or_sim, true_algo


def normalize_ds_score(ds_score, g1, g2):
    g1_size = g1.get_nxgraph().number_of_nodes()
    g2_size = g2.get_nxgraph().number_of_nodes()
    return 2 * ds_score / (g1_size + g2_size)


def dist_to_sim(dist, kernel):
    if 'exp' in kernel:
        scale = _get_exp_kernel_scale(kernel)
        return np.exp(-scale * dist)
    else:
        raise NotImplementedError()


def sim_to_dist(sim, kernel):
    if kernel == 'exp':
        scale = _get_exp_kernel_scale(kernel)
        return -np.log(sim) / scale
    else:
        raise NotImplementedError()


def _get_exp_kernel_scale(kernel):
    scale = float(kernel.split('exp_')[1])  # TODO: check
    return scale
