from dataset import OurDataset
from graph import RegularGraph
from graph_pair import GraphPair

from os.path import join
import os
import csv
import networkx as nx
from networkx import empty_graph
import itertools
import random

from tqdm import tqdm

LABEL = 'type' # TODO merge with dataset_config

def load_pdb_data_mcs(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert natts == ['type', 'x', 'y', 'z', 'ss_type'] and eatts == ['dist', 'id'] and tvt in ['train', 'test'] and \
           align_metric == 'random' and glabel == 'random'

    path_graphs = None
    path_pairs = None
    pair_key_list = load_pair_key_list(path_pairs)
    graph_dict = load_graph_dict(path_graphs) # gid -> nxgraph

    graph_list = []
    for g in graph_dict.values():
        graph_list.append(RegularGraph(g))

    pairs = {}
    for pair_key in pair_key_list:
        mapping = {}
        gid0, gid1 = pair_key
        # g0, g1 = graph_dict[gid0], graph_dict[gid1]
        pairs[(gid0, gid1)] = GraphPair(y_true_dict_list=[mapping], ds_true=len(mapping),
                                        running_time=0)

    glabel = None  # TODO: why is this always None?
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)

def load_pair_key_list(path):
    fp = open(path)
    reader = csv.reader(fp, delimiter=',')

    pair_key_list = []
    for row in reader:
        assert len(row) == 2
        pair_key = tuple([int(gid) for gid in row])
        pair_key_list.append(pair_key)
    return pair_key_list

def load_graph_dict(path):
    graph_dict = {}
    print('loading pdb graphs (might take a while)...')
    for graph_file in tqdm(os.listdir(path)):
        gid = int(graph_file.split('.')[0])
        g = nx.read_gexf(join(path, graph_file))
        g = relabel_nodes_str2int(g)
        add_pdb_edges(g)
        rm_node_attr(g)
        g.graph['gid'] = gid
        graph_dict[gid] = g
    return graph_dict

def relabel_nodes_str2int(g):
    mapping = {}
    for v in g.nodes:
        mapping[v] = int(v)
    return nx.relabel_nodes(g, mapping)

def rm_node_attr(g):
    for node in g.nodes:
        del g.nodes[node]['label']

import numpy as np
def compute_dist(p1, p2):
    return np.sqrt(np.sum((np.array(p1)-np.array(p2))**2))

def add_pdb_edges(g):
    return g
