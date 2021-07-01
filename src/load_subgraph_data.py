from dataset import OurDataset
from graph import RegularGraph
from graph_pair import GraphPair

import random
import networkx as nx
from os.path import join

def get_random_new_nids(g, seed):
    new_nids = list(range(g.number_of_nodes()))
    seed.shuffle(new_nids)
    return new_nids

def apply_node_relabel(g, seed):
    new_nids = get_random_new_nids(g, seed)
    assert len(new_nids) == g.number_of_nodes()
    mapping = {}
    i = 0
    for nid in g.nodes():
        mapping[nid] = new_nids[i]
        i += 1
    return nx.relabel_nodes(g, mapping)

def load_nxgraph(g):
    # TODO: what format?
    return nx.Graph()

def grab_subgraph(g, num_nodes, seed):
    # TODO: bfs? dfs? random walk? connectivity?
    return g

def load_subgraph_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert natts == [] and eatts == [] and tvt in ['train', 'test']

    random.seed(123)  # just to be safe... NOTE this fixes seed for all random fn!
    seed = random.Random(123)

    gid = 0
    pairs = {}
    graph_list = []
    # TODO: what format?
    for g in ___:
        # get the graphs
        g_nx = load_nxgraph(g)
        g1, g2 = grab_subgraph(g_nx, g_nx.number_of_nodes()), grab_subgraph(g_nx, g_nx.number_of_nodes())

        # apply post processing
        gid1, gid2, mapping = gid, gid+1, {}
        g1.graph['gid'] = gid1
        g2.graph['gid'] = gid2
        apply_node_relabel(g1, seed)
        apply_node_relabel(g2, seed)

        # append to data structs
        graph_list.append(RegularGraph(g1))
        graph_list.append(RegularGraph(g2))
        pairs[(gid1, gid2)] = GraphPair(y_true_dict_list=[mapping], ds_true=len(mapping),
                                        running_time=0)
        gid += 2

    glabel = None
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)

