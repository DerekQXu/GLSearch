from config import FLAGS
import torch
import torch.nn as nn

import sys
from os.path import dirname, abspath, join
import numpy as np
import networkx as nx

cur_folder = dirname(abspath(__file__))
print(join(dirname(dirname(cur_folder)), 'model', 's-gwl'))
sys.path.insert(0, join(dirname(dirname(cur_folder)), 'model', 's-gwl'))

try:
    from methods.GromovWassersteinGraphToolkit import recursive_direct_graph_matching
except:
    print('Note: Gromov Wasserstein not supported. Please obtain Gromov Wasserstein Graph Toolkit')

num_iter = 2000
ot_dict = {'loss_type': 'L2',  # the key hyperparameters of GW distance
           'ot_method': 'proximal',
           'beta': 0.025,
           'outer_iteration': num_iter,
           # outer, inner iteration, error bound of optimal transport
           'iter_bound': 1e-30,
           'inner_iteration': 2,
           'outer_iteration': num_iter,
           'sk_bound': 1e-30,
           'node_prior': 1e3,
           'max_iter': 4,  # iteration and error bound for calcuating barycenter
           'cost_bound': 1e-26,
           'update_p': False,  # optional updates of source distribution
           'lr': 0,
           'alpha': 0}


class SGW(torch.nn.Module):
    def __init__(self):
        super(SGW, self).__init__()

    def forward(self, ins, batch_data, model):
        pair_list = batch_data.split_into_pair_list(ins, 'x')
        for pair in pair_list:
            g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
            mask = torch.matmul(torch.tensor(g1.init_x), torch.tensor(g2.init_x.T)).type(
                torch.FloatTensor).to(FLAGS.device)
            soft_Y = sgw(g1, g2)
            np.set_printoptions(precision=4)
            np.set_printoptions(suppress=True)
            print(soft_Y)
            if np.isnan(soft_Y).any():
                soft_Y = np.zeros_like(soft_Y)
            # TODO 1: masking with init features
            pair.assign_y_pred_list(  # used by evaluation
                [mask *torch.tensor(soft_Y).type(torch.FloatTensor).to(FLAGS.device)],
                format='torch_{}'.format(FLAGS.device))  # multiple predictions
        return torch.FloatTensor([0.0]) # indicating 0 loss


def sgw(g1, g2):
    g1_nodes = sorted(g1.nodes)
    g2_nodes = sorted(g2.nodes)
    probs1 = [g1.degree[x] for x in g1_nodes]
    p_s = np.array(probs1 / np.sum(probs1)).reshape((len(probs1), 1))
    probs2 = [g2.degree[x] for x in g2_nodes]
    p_t = np.array(probs2 / np.sum(probs2)).reshape((len(probs2), 1))
    cost_s = nx.adjacency_matrix(g1, nodelist=g1_nodes)
    cost_t = nx.adjacency_matrix(g2, nodelist=g2_nodes)
    idx2node_s = dict(zip(range(len(g1_nodes)), sorted(g1_nodes)))
    idx2node_t = dict(zip(range(len(g2_nodes)), sorted(g2_nodes)))

    pairs_idx, pairs_name, pairs_confidence, soft_Y = recursive_direct_graph_matching(
        0.5 * (cost_s + cost_s.T), 0.5 * (cost_t + cost_t.T), p_s, p_t, idx2node_s, idx2node_t,
        ot_dict,
        weights=None, predefine_barycenter=False, cluster_num=2,
        partition_level=3, max_node_num=0)

    return soft_Y


if __name__ == '__main__':
    from load_data import load_dataset
    name = 'syn:np_tr=2,np_te=2,nn_core=10,nn_tot=20,ed=2,gen_type=BA'
    # name = 'syn:np_tr=1000,np_te=100,nn_core=24,nn_tot=64,ed=0.2,gen_type=ER'
    # name = 'syn:np_tr=1000,np_te=100,nn_core=30,nn_tot=64,ed=0.2|4,gen_type=WS'
    # dataset = load_dataset(name, 'train', 'random', 'bfs')
    dataset = load_dataset(name, 'test', 'random', 'bfs')
    print(dataset)
    g1, g2 = dataset.gs[0].nxgraph, dataset.gs[1].nxgraph
    soft_Y = sgw(g1, g2)
    print(soft_Y)
    # dataset.save_graphs_as_gexf()




