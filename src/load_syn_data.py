from dataset import OurDataset
from graph import RegularGraph
from graph_pair import GraphPair

import random
import itertools
import numpy as np
import networkx as nx
from networkx import empty_graph
from networkx.generators import barabasi_albert_graph, erdos_renyi_graph, watts_strogatz_graph

LABEL = 'type'  # TODO merge with dataset_config


def get_new_nid_mapping(bias, g):
    mapping = {}
    for nid in range(g.number_of_nodes()):
        mapping[nid] = nid + bias
    return mapping


def get_linking_edges(single_mapping, prev_single_mapping, l, seed):
    linking_edges1 = []
    linking_edges2 = []
    k = int(min(len(single_mapping), len(prev_single_mapping)) * l)
    sampled_nid1 = seed.sample(list(prev_single_mapping.keys()), k)
    sampled_nid2 = seed.sample(list(single_mapping.keys()), k)
    for i in range(k):
        nid1 = sampled_nid1[i]
        nid2 = sampled_nid2[i]
        linking_edges1.append((nid1, nid2))
        linking_edges2.append((prev_single_mapping[nid1], single_mapping[nid2]))
    return linking_edges1, linking_edges2


def relabel_g(g, bias):
    new_nid_mapping = get_new_nid_mapping(bias, g)
    g = nx.relabel_nodes(g, new_nid_mapping)
    return g


def expand_mapping(src_mapping, bias):
    new_mapping = {}
    for v in src_mapping:
        w = src_mapping[v]
        new_mapping[v + bias] = w + bias
    return new_mapping


def update_prev_mapping(prev_mapping, single_mapping):
    for v, w in single_mapping.items():
        prev_mapping[v] = w


def link_graphs(gs1, gs2, mappings, seed):
    assert len(gs1) >= 2
    assert len(gs1) == len(gs2) == len(mappings)
    prev_g1 = gs1[0]
    prev_g2 = gs2[0]
    prev_mapping = mappings[0]
    prev_single_mapping = mappings[0]
    for i in range(1, len(gs1)):
        # assumes all gs have same #nodes
        bias = prev_g1.number_of_nodes()
        new_g1 = relabel_g(gs1[i], bias)
        new_g2 = relabel_g(gs2[i], bias)
        single_mapping = expand_mapping(mappings[i], bias)
        linking_edges1, linking_edges2 = get_linking_edges(single_mapping, prev_single_mapping, 0.8,
                                                           seed)
        prev_single_mapping = single_mapping
        update_prev_mapping(prev_mapping, single_mapping)
        prev_g1 = nx.compose(prev_g1, new_g1)
        prev_g2 = nx.compose(prev_g2, new_g2)
        prev_g1.add_edges_from(linking_edges1)
        prev_g2.add_edges_from(linking_edges2)
    return prev_g1, prev_g2, prev_mapping


def load_syn_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert 'syn' in name
    assert natts == [LABEL] and eatts == [] and tvt in ['train', 'test'] and \
           align_metric in ['random', 'mcs'] and glabel == 'random'

    random.seed(123)  # just to be safe... NOTE this fixes seed for all random fn!
    seed = random.Random(123)

    # initialize graph parameters
    ps_raw = _parse_graphs_info(name, 'syn')
    ps_static, ps_dynamic = parse_configs(ps_raw, tvt)

    # get static configs
    num_pairs = ps_static['num_pairs']
    gen_type = ps_static['gen_type']
    num_feat = ps_static['num_feat']
    label_type = ps_static['label_type']

    # precompute which dynamic configs we will be using
    indices_dynamic = make_evenly_distributed_li(ps_static['num_pairs'],
                                                 ps_dynamic['num_dynamic_param'],
                                                 seed)

    # construct pairs
    pairs = {}
    graph_list = []
    num_graphs = 0
    for i in range(num_pairs):
        # sample dynamic config
        idx_dynamic = indices_dynamic[i]

        # get dynamic configs
        nn_core = ps_dynamic['nn_core'][idx_dynamic]
        nn_tot = ps_dynamic['nn_tot'][idx_dynamic]
        ed = ps_dynamic['ed'][idx_dynamic]
        assert nn_tot >= nn_core  # TODO: put this into the parse functions

        # generate the 2 graphs
        if nn_core == -1:
            g0, g1, mapping = _gen_2_indep_graphs(nn_tot, ed, gen_type, num_feat, label_type, seed)
        else:
            g0, g1, mapping = _gen_2_shared_graphs(nn_core, nn_tot, ed, gen_type, num_feat, label_type, seed)
        # g2, g3, mapping_23 = _gen_2_graphs(nn_core, nn_tot, ed, gen_type, num_feat, seed)
        # g4, g5, mapping_45 = _gen_2_graphs(nn_core, nn_tot, ed, gen_type, num_feat, seed)
        # g6, g7, mapping_67 = _gen_2_graphs(nn_core, nn_tot, ed, gen_type, num_feat, seed)
        # g8, g9, mapping_89 = _gen_2_graphs(nn_core, nn_tot, ed, gen_type, num_feat, seed)
        #
        # g0, g1, mapping = link_graphs(
        #     [g0, g2, g4, g6, g8], [g1, g3, g5, g7, g9],
        #     [mapping_01, mapping_23, mapping_45, mapping_67, mapping_89], seed)

        assert len(mapping) <= g0.number_of_nodes()
        assert len(mapping) <= g1.number_of_nodes()
        assert g0.number_of_nodes() > 0
        assert g1.number_of_nodes() > 0

        g0, g1, num_graphs = assign_gid_to_graphs(g0, g1, num_graphs)

        # append graphs to graph_list and pair objects
        graph_list.append(RegularGraph(g0))
        graph_list.append(RegularGraph(g1))
        gid0, gid1 = g0.graph['gid'], g1.graph['gid']
        pairs[(gid0, gid1)] = GraphPair(y_true_dict_list=[mapping], ds_true=len(mapping),
                                        running_time=0)

    glabel = None  # TODO: why is this always None?
    return OurDataset(name, graph_list, natts, eatts, pairs, tvt, align_metric, node_ordering,
                      glabel, None)


def parse_dynamic_config(ps_raw):
    ps_dynamic = {}

    nn_core_str_li = ps_raw['nn_core'].split(';')
    nn_tot_str_li = ps_raw['nn_tot'].split(';')
    ed_str_li = ps_raw['ed'].split(';')

    ps_dynamic['nn_core'] = cvt_li_elt_to_type(nn_core_str_li, int)
    ps_dynamic['nn_tot'] = cvt_li_elt_to_type(nn_tot_str_li, int)
    ps_dynamic['ed'] = cvt_li_elt_to_type(ed_str_li, str)

    assert len(ps_dynamic['nn_core']) == len(ps_dynamic['nn_tot']) == len(ps_dynamic['ed'])
    assert len(ps_dynamic['nn_core']) >= 1
    ps_dynamic['num_dynamic_param'] = len(ps_dynamic['nn_core'])

    return ps_dynamic


def parse_static_config(ps_raw, tvt):
    ps_static = {}

    get_what = 'np_tr' if tvt == 'train' else 'np_te'
    num_pairs_str = ps_raw[get_what]
    gen_type_str = ps_raw['gen_type'] if 'gen_type' in ps_raw.keys() else 'BA'
    num_feat_str = ps_raw['num_feat'] if 'num_feat' in ps_raw.keys() else 1
    label_type = ps_raw['label_type'] if 'label_type' in ps_raw.keys() else 'random'

    ps_static['num_pairs'] = int(num_pairs_str)
    ps_static['gen_type'] = str(gen_type_str)
    ps_static['num_feat'] = int(num_feat_str)
    ps_static['label_type'] = str(label_type)

    assert ps_static['num_pairs'] >= 1
    assert ps_static['num_feat'] >= 1
    assert ps_static['label_type'] in ['random', 'degree_bin']
    return ps_static


def parse_configs(ps_raw, tvt):
    ps_static = parse_static_config(ps_raw, tvt)
    ps_dynamic = parse_dynamic_config(ps_raw)
    return ps_static, ps_dynamic


def cvt_li_elt_to_type(li, type):
    return [type(elt) for elt in li]

# below code is quite tricky; possibly refactor?
def make_bin_distributed_li(nodes_list, g, num_feat, seed):
    # get degree distribution
    degree_dist = []
    for node in nodes_list:
        degree_dist.append(int(g.degree[node]))
    degree_argsort = np.argsort(degree_dist)

    # bin'd label assignment by degree
    li = [-1 for _ in range(len(nodes_list))]
    remaining_feat_li = list(range(num_feat))
    remaining_li_entries = len(li)
    start_idx = 0
    while len(remaining_feat_li) > 0:
        num_extra = int(remaining_li_entries/len(remaining_feat_li))

        label = remaining_feat_li.pop()
        indices = degree_argsort[start_idx:start_idx+num_extra]
        for idx in indices:
            assert li[idx] == -1
            li[idx] = label

        remaining_li_entries -= num_extra
        start_idx = start_idx + num_extra
    assert -1 not in li

    return li

def make_evenly_distributed_li(n, k, seed):
    li = []
    for i in range(n):
        elt = i % k
        li.append(elt)
    if seed is not None:
        seed.shuffle(li)
    return li


def _gen_2_indep_graphs(nn_tot, edge_density, gen_type, num_feat, label_type, seed):
    mapping = {}
    if gen_type == 'BA':
        edge_density = int(edge_density)
        G1 = barabasi_albert_graph(n=nn_tot, m=edge_density, seed=seed)
        G2 = barabasi_albert_graph(n=nn_tot, m=edge_density, seed=seed)
    elif gen_type == 'ER':
        edge_density = float(edge_density)

        finished = False
        gen_attempt = 0
        while not finished:
            print('gen attempt: {}'.format(gen_attempt))
            G1 = erdos_renyi_graph(n=nn_tot, p=edge_density, seed=seed)
            finished = nx.is_connected(G1)
            gen_attempt += 1

        finished = False
        gen_attempt = 0
        while not finished:
            print('gen attempt: {}'.format(gen_attempt))
            G2 = erdos_renyi_graph(n=nn_tot, p=edge_density, seed=seed)
            finished = nx.is_connected(G2)
            gen_attempt += 1
    elif gen_type == 'WS':
        p, k = edge_density.split('|')
        p, k = float(p), int(k)

        finished = False
        gen_attempt = 0
        while not finished:
            print('gen attempt: {}'.format(gen_attempt))
            G1 = watts_strogatz_graph(n=nn_tot, k=k, p=p, seed=seed)
            finished = nx.is_connected(G1)
            gen_attempt += 1

        finished = False
        gen_attempt = 0
        while not finished:
            print('gen attempt: {}'.format(gen_attempt))
            G2 = watts_strogatz_graph(n=nn_tot, k=k, p=p, seed=seed)
            finished = nx.is_connected(G2)
            gen_attempt += 1
    else:
        assert False

    import networkx.algorithms.isomorphism as iso
    natts = [LABEL]
    nm = iso.categorical_node_match(natts, [''] * len(natts))
    if nx.is_isomorphic(G1, G2, node_match=nm):
        print(f'{gen_type} isomorphic pair generated')

    # add node labels
    G1, G2 = label_pair(G1, G2, mapping, num_feat, label_type, seed)
    return G1, G2, mapping


def _gen_2_shared_graphs(nn_core, nn_tot, edge_density, gen_type, num_feat, label_type, seed):
    # generate the 2 graphs (generation code from networkx documentation)
    if gen_type == 'BA':
        edge_density = int(edge_density)
        G1, G2 = gen_BA_pair(nn_tot, edge_density, nn_core, seed)
    elif gen_type == 'ER':
        edge_density = float(edge_density)
        G1, G2 = gen_ER_pair(nn_tot, edge_density, nn_core, seed)
    elif gen_type == 'WS':
        p, k = edge_density.split('|')
        p, k = float(p), int(k)
        G1, G2 = gen_WS_pair(nn_tot, nn_core, p, k, seed)
    else:
        assert False

    # shuffle the node ids
    nids = list(range(nn_tot))
    nids1, nids2 = nids.copy(), nids.copy()
    seed.shuffle(nids1)
    seed.shuffle(nids2)
    relabel_dict1, relabel_dict2 = {}, {}
    for i, nid1 in enumerate(nids1):
        relabel_dict1[i] = nid1
    for i, nid2 in enumerate(nids2):
        relabel_dict2[i] = nid2
    G1 = nx.relabel_nodes(G1, relabel_dict1)
    G2 = nx.relabel_nodes(G2, relabel_dict2)

    # fix mapping to reflect shuffled node ids
    mapping = {}
    for i in range(int(nn_core)):
        mapping[relabel_dict1[i]] = relabel_dict2[i]

    # add node labels
    G1, G2 = label_pair(G1, G2, mapping, num_feat, label_type, seed)

    # import networkx.algorithms.isomorphism as iso
    # natts = [LABEL]
    # nm = iso.categorical_node_match(natts, [''] * len(natts))
    # assert nx.is_isomorphic(G1.subgraph(list(mapping.keys())),
    #                         G2.subgraph(list(mapping.values())),
    #                         node_match=nm)

    return G1, G2, mapping


def gen_BA_pair(n, m, k, seed):  # TODO: clean up this function!
    ############## FROM networkx ##############
    if m < 1 or m >= n:
        raise nx.NetworkXError("Barabási–Albert network must have m >= 1"
                               " and m < n, m = %d, n = %d" % (m, n))

    # Add m initial nodes (m0 in barabasi-speak)
    G = empty_graph(m)
    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < k:
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m, seed)
        source += 1

    source_1, source_2 = k, k
    repeated_nodes_1, repeated_nodes_2 = repeated_nodes.copy(), repeated_nodes.copy()
    targets_1, targets_2 = targets.copy(), targets.copy()
    G1, G2 = G.copy(), G.copy()

    while source_1 < n:
        assert source_1 == source_2
        G1.add_edges_from(zip([source_1] * m, targets_1))
        repeated_nodes_1.extend(targets_1)
        repeated_nodes_1.extend([source_1] * m)
        targets_1 = _random_subset(repeated_nodes_1, m, seed)
        source_1 += 1

        G2.add_edges_from(zip([source_2] * m, targets_2))
        repeated_nodes_2.extend(targets_2)
        repeated_nodes_2.extend([source_2] * m)
        targets_2 = _random_subset(repeated_nodes_2, m, seed)
        source_2 += 1

    return G1, G2


def gen_ER_pair(n, p, k, seed):
    common_nids = list(range(k))
    finished = False
    gen_attempt = 0
    while not finished:
        print('gen attempt: {}'.format(gen_attempt))
        edges = itertools.combinations(range(n), 2)
        G1 = nx.Graph()
        G2 = nx.Graph()
        G3 = nx.Graph()
        G1.add_nodes_from(range(n))
        G2.add_nodes_from(range(n))
        G3.add_nodes_from(range(k))
        for e in edges:
            nid1, nid2 = e
            if nid1 in common_nids and nid2 in common_nids:
                opt = seed.random() < p
                if opt:
                    G1.add_edge(*e)
                    G2.add_edge(*e)
                    G3.add_edge(*e)
            else:
                opt1 = seed.random() < p
                opt2 = seed.random() < p
                if opt1:
                    G1.add_edge(*e)
                if opt2:
                    G2.add_edge(*e)

        if nx.is_connected(G1) and nx.is_connected(G2) and nx.is_connected(G3):
            finished = True
        gen_attempt += 1
    return G1, G2


def gen_WS_pair(n, l, p, k, seed):
    common_nids = list(range(l))
    finished = False
    gen_attempt = 0
    while not finished:
        print('gen attempt: {}'.format(gen_attempt))
        G1 = nx.Graph()
        G2 = nx.Graph()
        G3 = nx.Graph()

        nodes = list(range(n))
        for j in range(1, k // 2 + 1):
            targets = nodes[j:] + nodes[0:j]
            G1.add_edges_from(zip(nodes, targets))
            G2.add_edges_from(zip(nodes, targets))
            G3.add_edges_from(zip(nodes, targets))
        for j in range(1, k // 2 + 1):
            targets = nodes[j:] + nodes[0:j]
            for u, v in zip(nodes, targets):
                if seed.random() < p:
                    w1 = seed.choice(nodes)
                    w2 = seed.choice(nodes)
                    w3 = seed.choice(nodes)
                    skip_flag1, skip_flag2, skip_flag3 = False, False, False
                    # Enforce no self-loops or multiple edges
                    while w1 == u or G1.has_edge(u, w1):
                        w1 = seed.choice(nodes)
                        if G1.degree(u) >= n - 1:
                            skip_flag1 = True
                            break  # skip this rewiring
                    while w2 == u or G2.has_edge(u, w2):
                        w2 = seed.choice(nodes)
                        if G2.degree(u) >= n - 1:
                            skip_flag2 = True
                            break  # skip this rewiring
                    while w3 == u or G1.has_edge(u, w3) or G2.has_edge(u, w3) or G3.has_edge(u, w3):
                        w3 = seed.choice(nodes)
                        if G1.degree(u) >= n - 1 or G2.degree(u) >= n - 1 or G3.degree(u) >= n - 1:
                            skip_flag3 = True
                            break  # skip this rewiring
                    is_common_nt = (u in common_nids) and (v in common_nids)
                    is_common_1 = is_common_nt or (u in common_nids and w1 in common_nids)
                    is_common_2 = is_common_nt or (u in common_nids and w2 in common_nids)
                    is_common_3 = is_common_nt or (u in common_nids and w3 in common_nids)
                    if is_common_3 and not skip_flag3:
                        G1.remove_edge(u, v)
                        G1.add_edge(u, w3)
                        G2.remove_edge(u, v)
                        G2.add_edge(u, w3)
                        G3.remove_edge(u, v)
                        G3.add_edge(u, w3)
                    else:
                        if not is_common_1 and not skip_flag1:
                            G1.remove_edge(u, v)
                            G1.add_edge(u, w1)
                        if not is_common_2 and not skip_flag2:
                            G2.remove_edge(u, v)
                            G2.add_edge(u, w2)
        G3 = G3.subgraph(common_nids)
        if nx.is_connected(G1) and nx.is_connected(G2) and nx.is_connected(G3):
            finished = True
        gen_attempt += 1
    return G1, G2


# will only label unlabelled nodes
def label_pair(G1, G2, mapping, num_feat, label_type, seed):
    # get shared and unshared nodes
    nodes_same_g1 = list(mapping.keys())
    nodes_same_g2 = list(mapping.values())
    nodes_diff_g1 = list(set(G1.nodes) - set(nodes_same_g1))
    nodes_diff_g2 = list(set(G2.nodes) - set(nodes_same_g2))

    # precompute which labels we will be using
    if label_type == 'random':
        assert len(nodes_same_g1) == len(nodes_same_g2)
        indices_same = make_evenly_distributed_li(len(nodes_same_g1), num_feat, seed)
        indices_diff_g1 = make_evenly_distributed_li(len(nodes_diff_g1), num_feat, seed)
        indices_diff_g2 = make_evenly_distributed_li(len(nodes_diff_g2), num_feat, seed)
    elif label_type == 'degree_bin':
        # NOTE: if the degree distribution of core and not core are different -> weirdly distributed
        assert len(nodes_same_g1) == len(nodes_same_g2) == 0
        indices_same = []
        indices_diff_g1 = make_bin_distributed_li(nodes_diff_g1, G1, num_feat, seed)
        indices_diff_g2 = make_bin_distributed_li(nodes_diff_g2, G2, num_feat, seed)
    else:
        assert False

    G1 = label_graph(G1, nodes_same_g1, indices_same)
    G1 = label_graph(G1, nodes_diff_g1, indices_diff_g1)
    G2 = label_graph(G2, nodes_same_g2, indices_same)
    G2 = label_graph(G2, nodes_diff_g2, indices_diff_g2)

    return G1, G2


def label_graph(G, nodes_li, indices_labels):
    # assign labels
    for i, node in enumerate(nodes_li):
        label = indices_labels[i]  # sample label
        G.nodes[node][LABEL] = label
    return G


def _random_subset(seq, m, rng):
    """ Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.

    Note: rng is a random.Random or numpy.random.RandomState instance.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)
    return targets


def _parse_graphs_info(name, zeroth_word):
    # syn:np=200,nn_core=32,nn_tot=64,ed=1
    sp = name.split(':')
    assert len(sp) == 2
    assert sp[0] == zeroth_word
    graphs_info = {}
    for spec in sp[1].split(','):
        ssp = spec.split('=')
        graphs_info[ssp[0]] = '='.join(ssp[1:])
    return graphs_info


def assign_gid_to_graphs(g0, g1, num_graphs):
    g0.graph['gid'] = num_graphs
    g1.graph['gid'] = num_graphs + 1
    num_graphs += 2
    return g0, g1, num_graphs
