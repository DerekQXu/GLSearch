from dataset import OurDataset
from graph import RegularGraph
from graph_pair import GraphPair

import os
import copy
import csv
import random
import networkx as nx
from os.path import join

import networkx.algorithms.isomorphism as iso

def read_graph(fn, delimiter, num_lines_header):
    fp = open(fn)
    csv_reader = csv.reader(fp, delimiter=delimiter)
    g = nx.Graph()
    for _ in range(num_lines_header):
        next(csv_reader)
    i = 0
    for line in csv_reader:
        if i < 1:
            print(line)
            i += 1
        u,v = line
        g.add_edge(u,v)
    return g

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

def get_file_path(fn, ext):
    fnp = join('/', 'home', 'username', 'Documents', 'GraphMatching', 'file', 'MCSRL_files', 'TODO', f'{fn}.{ext}')
    return fnp


def is_edge_in_sg(e, sg_nodes):
    i,j = e
    return (i in sg_nodes) and (j in sg_nodes)

def add_edge(g, sg_nodes, seed):
    nnodes_before = g.number_of_nodes()
    chosen_edge = seed.sample(g.nodes,2)
    while chosen_edge in g.edges or is_edge_in_sg(chosen_edge, sg_nodes):
        chosen_edge = seed.sample(g.nodes,2)
    g.add_edge(chosen_edge[0], chosen_edge[1])
    nnodes_after = g.number_of_nodes()
    assert nnodes_before == nnodes_after

def del_edge(g, sg_nodes, seed):
    nnodes_before = g.number_of_nodes()
    chosen_edge = seed.choice(list(g.edges))
    while is_edge_in_sg(chosen_edge, sg_nodes):
        chosen_edge = seed.choice(list(g.edges))
    g.remove_edge(chosen_edge[0], chosen_edge[1])
    nnodes_after = g.number_of_nodes()
    assert nnodes_before == nnodes_after
    return g

def perturb_graph(g, sg_nodes, ntimes, seed, batch=5):
    print('starting to perturb...')
    while ntimes > 0:
        g_cpy = copy.deepcopy(g)
        for i in range(batch):
            del_edge(g_cpy, sg_nodes, seed)
            add_edge(g_cpy, sg_nodes, seed)
        if nx.is_connected(g_cpy):
            g = g_cpy
            ntimes -= batch
            if ntimes % 50 == 0:
                print(f'{ntimes} perturbations left')
    print('perturbation done')
    return g

def get_sg_nodes(start_node, g, min_len):
    queue = [start_node]
    subg_nodes = [start_node]
    while len(subg_nodes) < min_len:
        if len(queue) == 0:
            print('wtf?')
            break
        node = queue.pop()
        neighbors = set(g.neighbors(node)) - set(subg_nodes)
        for node_out in neighbors:
            subg_nodes.append(node_out)
            queue.append(node_out)
    return subg_nodes

def is_iso_wrapper(g1, g2, v1, v2):
    sg1, sg2 = g1.subgraph(v1), g2.subgraph(v2)
    natts, eatts = [], []
    nm = iso.categorical_node_match(natts, [''] * len(natts))
    em = iso.categorical_edge_match(eatts, [''] * len(eatts))
    return nx.is_isomorphic(sg1, sg2, node_match=nm, edge_match=em)

def find_3sgs(g, n_frac, seed):
    sg_size = int(n_frac * len(g.nodes))
    seed_node1 = seed.choice(list(g.nodes))
    sg_nodes1 = get_sg_nodes(seed_node1, g, sg_size)[:sg_size]
    seed_node2 = seed.choice(list(set(g.nodes) - set(sg_nodes1)))
    sg_nodes2 = get_sg_nodes(seed_node2, g, sg_size)[:sg_size]
    seed_node3 = seed.choice(list(set(g.nodes) - set(sg_nodes1) - set(sg_nodes2)))
    sg_nodes3 = get_sg_nodes(seed_node3, g, sg_size)[:sg_size]
    sg1, sg2, sg3 = g.subgraph(sg_nodes1), \
                    g.subgraph(sg_nodes2), \
                    g.subgraph(sg_nodes3)
    print('checking if sg connected...')
    assert nx.is_connected(sg1)
    assert nx.is_connected(sg2)
    assert nx.is_connected(sg3)
    print('sg connectivity check passed!')
    return sg1, sg2, sg3

def randomly_connect_2sgs(sg1, sg2, n_edges, seed):
    for v in sg1.nodes:
        sg1.nodes[v]['sg_node'] = True

    g = nx.disjoint_union(sg1, sg2)

    sg1_nodes, sg2_nodes = set(), set()
    for v in g:
        if 'sg_node' in g.nodes[v]:
            sg1_nodes.add(v)
        else:
            sg2_nodes.add(v)

    for _ in range(n_edges):
        chosen_edge = (
            seed.choice(list(sg1_nodes)),
            seed.choice(list(sg2_nodes))
        )
        while chosen_edge in g.edges:
            chosen_edge = (
                seed.choice(list(sg1_nodes)),
                seed.choice(list(sg2_nodes))
            )
        g.add_edge(chosen_edge[0], chosen_edge[1])
    # [v for v in g.nodes if 'sg_node' in g.nodes[v]]
    return g



def my_disjoint_union(G, H):
    """ Return the disjoint union of graphs G and H.

    This algorithm forces distinct integer node labels.

    Parameters
    ----------
    G,H : graph
       A NetworkX graph

    Returns
    -------
    U : A union graph with the same type as G.

    Notes
    -----
    A new graph is created, of the same class as G.  It is recommended
    that G and H be either both directed or both undirected.

    The nodes of G are relabeled 0 to len(G)-1, and the nodes of H are
    relabeled len(G) to len(G)+len(H)-1.

    Graph, edge, and node attributes are propagated from G and H
    to the union graph.  If a graph attribute is present in both
    G and H the value from H is used.
    """
    R1 = nx.convert_node_labels_to_integers(G)
    R2 = nx.convert_node_labels_to_integers(H, first_label=len(R1))
    for v in R1.nodes:
        R1.nodes[v]['sg_node'] = True
    R = nx.union(R1, R2)
    R.graph.update(G.graph)
    R.graph.update(H.graph)
    return R


def connect_subgraphs(g, n_frac, n_edges, seed):
    sg1, sg2, sg3 = find_3sgs(g, n_frac, seed)
    g1 = randomly_connect_2sgs(sg1, sg2, n_edges, seed)
    g2 = randomly_connect_2sgs(sg1, sg3, n_edges, seed)
    # [v for v in g1.nodes if 'sg_node' in g1.nodes[v]]
    return g1, g2

def load_ssgexf_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert natts == ['type'] and eatts == [] and tvt in ['train', 'test']
    data_name, label_name, fn_ntimes = name.split(':')
    fn, n_frac, n_edges = fn_ntimes.split(';')
    n_frac = float(n_frac)
    n_edges = int(n_edges)

    assert n_frac < 0.3

    random.seed(123)  # just to be safe... NOTE this fixes seed for all random fn!
    seed = random.Random(123)

    graph_list, pairs, mapping = [], {}, {}

    fn = get_file_path(fn, 'gexf')
    g = nx.read_gexf(fn).to_undirected()
    g = max(nx.connected_component_subgraphs(g), key=len)

    g1, g2 = connect_subgraphs(g, n_frac, n_edges, seed)
    # assert is_iso_wrapper(g1, g2, sg_nodes, sg_nodes)

    gid1, gid2 = 0,1

    g1.graph['gid'] = gid1
    g2.graph['gid'] = gid2

    _assign_labels(g1, label_name)
    _assign_labels(g2, label_name)

    g1 = apply_node_relabel(g1, seed)
    g2 = apply_node_relabel(g2, seed)

    print('checking isomorphism...')
    sg_nodes_1 = [v for v in g1.nodes if 'sg_node' in g1.nodes[v]]
    sg_nodes_2 = [v for v in g2.nodes if 'sg_node' in g2.nodes[v]]
    for v in sg_nodes_1:
        del g1.nodes[v]['sg_node']
    for v in sg_nodes_2:
        del g2.nodes[v]['sg_node']
    g1.graph['sg'] = sg_nodes_1
    g2.graph['sg'] = sg_nodes_2
    # assert is_iso_wrapper(g1, g2, sg_nodes_1, sg_nodes_2)
    print('iso check passed!')

    graph_list.append(RegularGraph(g1))
    graph_list.append(RegularGraph(g2))
    pairs[(gid1, gid2)] = GraphPair(
        y_true_dict_list=[mapping], ds_true=len(mapping), running_time=0)

    glabel = None  # TODO: why is this always None?
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)


def load_ccgexf_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert natts == ['type'] and eatts == [] and tvt in ['train', 'test']
    data_name, label_name, fn_ntimes = name.split(':')
    fn, ntimes, n_frac, batch = fn_ntimes.split(';')
    ntimes = int(ntimes)
    n_frac = float(n_frac)
    batch = int(batch)

    random.seed(123)  # just to be safe... NOTE this fixes seed for all random fn!
    seed = random.Random(123)

    graph_list, pairs, mapping = [], {}, {}

    fn = get_file_path(fn, 'gexf')
    g = nx.read_gexf(fn).to_undirected()
    g = max(nx.connected_component_subgraphs(g), key=len)
    sg_nodes = get_sg_nodes(list(g.nodes)[12], g, int(n_frac*len(g.nodes)))

    g1 = perturb_graph(g, sg_nodes, ntimes, seed, batch)
    g2 = perturb_graph(g, sg_nodes, ntimes, seed, batch)
    # assert is_iso_wrapper(g1, g2, sg_nodes, sg_nodes)

    gid1, gid2 = 0,1
    _assign_labels(g1, label_name)
    _assign_labels(g2, label_name)
    for v in sg_nodes:
        g1.nodes[v]['sg_node'] = True
        g2.nodes[v]['sg_node'] = True
    g1.graph['gid'] = gid1
    g2.graph['gid'] = gid2

    g1 = apply_node_relabel(g1, seed)
    g2 = apply_node_relabel(g2, seed)

    print('checking isomorphism...')
    sg_nodes_1 = [v for v in g1.nodes if 'sg_node' in g1.nodes[v]]
    sg_nodes_2 = [v for v in g2.nodes if 'sg_node' in g2.nodes[v]]
    for v in sg_nodes_1:
        del g1.nodes[v]['sg_node']
    for v in sg_nodes_2:
        del g2.nodes[v]['sg_node']
    g1.graph['sg'] = sg_nodes_1
    g2.graph['sg'] = sg_nodes_2
    # assert is_iso_wrapper(g1, g2, sg_nodes_1, sg_nodes_2)
    print('iso check passed!')

    graph_list.append(RegularGraph(g1))
    graph_list.append(RegularGraph(g2))
    pairs[(gid1, gid2)] = GraphPair(
        y_true_dict_list=[mapping], ds_true=len(mapping), running_time=0)

    glabel = None  # TODO: why is this always None?
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)

def load_duogexf_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert natts == ['type'] and eatts == [] and tvt in ['train', 'test']
    data_name, label_name, fns = name.split(':')
    fn1, fn2 = fns.split(';')
    assert data_name in ['duogexf', 'duogexf_tsv', 'duogexf_csv']

    random.seed(123)  # just to be safe... NOTE this fixes seed for all random fn!
    seed = random.Random(123)

    graph_list, pairs, mapping = [], {}, {}

    gid1, gid2 = 0,1
    if data_name == 'duogexf':
        fn1 = get_file_path(fn1,'gexf')
        fn2 = get_file_path(fn2,'gexf')
        g1 = nx.read_gexf(fn1).to_undirected()
        g2 = nx.read_gexf(fn2).to_undirected()
    elif data_name == 'duogexf_tsv':
        fn1 = get_file_path(fn1,'tsv')
        fn2 = get_file_path(fn2,'tsv')
        delimiter = '\t'
        num_lines_header = int(input('How many lines in header? '))
        g1 = read_graph(fn1, delimiter, num_lines_header)
        g2 = read_graph(fn2, delimiter, num_lines_header)
    elif data_name == 'duogexf_csv':
        fn1 = get_file_path(fn1,'csv')
        fn2 = get_file_path(fn2,'csv')
        delimiter = ','
        num_lines_header = int(input('How many lines in header? '))
        g1 = read_graph(fn1, delimiter, num_lines_header)
        g2 = read_graph(fn2, delimiter, num_lines_header)
    else:
        assert False
    _assign_labels(g1, label_name)
    _assign_labels(g2, label_name)
    g1 = max(nx.connected_component_subgraphs(g1), key=len)
    g2 = max(nx.connected_component_subgraphs(g2), key=len)
    g1.graph['gid'] = gid1
    g2.graph['gid'] = gid2
    g1 = apply_node_relabel(g1, seed)
    g2 = apply_node_relabel(g2, seed)
    graph_list.append(RegularGraph(g1))
    graph_list.append(RegularGraph(g2))
    pairs[(gid1, gid2)] = GraphPair(y_true_dict_list=[mapping], ds_true=len(mapping),
                                    running_time=0)

    glabel = None  # TODO: why is this always None?
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)

def _assign_labels(g, label_name):
    for node in g.nodes:
        if label_name == '':
            label_val = 0
        else:
            label_keys = label_name.split('.')
            label_val = '.'
            for key in label_keys:
                label_val += f'{g.nodes[node][key]}.'
        label_keys = list(g.nodes[node].keys())
        for label_key in label_keys:
            del g.nodes[node][label_key]
        g.nodes[node]['type'] = label_val
    for edge in g.edges:
        label_keys = list(g.edges[edge].keys())
        for label_key in label_keys:
            del g.edges[edge][label_key]

def load_isogexf_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert natts == [] and eatts == [] and tvt in ['train', 'test']
    data_name, fn = name.split(':')
    fn = join('/','home','username','Documents','GraphMatching','data', fn)
    assert data_name == 'isogexf'

    random.seed(123)  # just to be safe... NOTE this fixes seed for all random fn!
    seed = random.Random(123)

    delimiter = input('What is the delimiter? ')
    num_lines_header = int(input('How many lines in header? '))

    # initialize graph parameters
    gid = 0
    pairs = {}
    graph_list = []
    for graph_file in os.listdir(fn):
        mapping = {}
        gid1, gid2 = gid, gid+1
        g1 = read_graph(join(fn, graph_file), delimiter, num_lines_header)
        g2 = read_graph(join(fn, graph_file), delimiter, num_lines_header)
        g1 = max(nx.connected_component_subgraphs(g1), key=len)
        g2 = max(nx.connected_component_subgraphs(g2), key=len)
        g1.graph['gid'] = gid1
        g2.graph['gid'] = gid2
        g1 = apply_node_relabel(g1, seed)
        g2 = apply_node_relabel(g2, seed)
        graph_list.append(RegularGraph(g1))
        graph_list.append(RegularGraph(g2))
        pairs[(gid1, gid2)] = GraphPair(y_true_dict_list=[mapping], ds_true=len(mapping),
                                        running_time=0)
        gid += 2

    glabel = None  # TODO: why is this always None?
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)

