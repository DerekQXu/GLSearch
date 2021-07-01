from dataset import OurDataset
from graph import RegularGraph
from graph_pair import GraphPair
from graph_label import add_glabel_to_each_graph
import copy
from utils import sorted_nicely, get_data_path, assert_valid_nid
from os.path import join, basename
from glob import glob

from ast import literal_eval
from pandas import read_csv
import os
import networkx as nx


def gen_BA(n_nodes, m_edges_density, query_size):
    assert n_nodes >= m_edges_density
    assert n_nodes >= query_size

    from networkx.generators.random_graphs import barabasi_albert_graph as BA
    import networkx as nx
    import random

    g1 = BA(n_nodes, m_edges_density)
    g2 = nx.Graph(g1.subgraph(list(range(query_size)))).copy()
    g2 = max(nx.connected_component_subgraphs(g2), key=len)

    nid_dict_layer1 = {}
    relabel_dict = {}
    for i, nid in enumerate(g2.node):
        nid_dict_layer1[i] = nid
        relabel_dict[nid] = i
    g2 = nx.relabel_nodes(g2, relabel_dict)

    nid_dict_layer2 = {}
    relabel_dict = {}
    nids = list(range(g2.number_of_nodes()))
    random.shuffle(nids)
    for i, nid in enumerate(nids):
        nid_dict_layer2[i] = nid
        relabel_dict[i] = nid
    g2 = nx.relabel_nodes(g2, relabel_dict)

    mapping = {}
    for i in range(g2.number_of_nodes()):
        mapping[nid_dict_layer1[i]] = nid_dict_layer2[i]

    return g1, g2, mapping


def load_debug_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert 'debug' in name
    assert tvt == 'all' or 'BA' in name  # TODO: change this
    assert align_metric == 'random'
    dir_name = 'AIDS700nef'
    graphs = []
    natts = ['type']
    eatts = []
    if name in ['debug', 'debug_no-1']:
        pairs = {(30, 21): GraphPair([{0: 1, 2: -1}, {0: 1, 2: -1}], 1, running_time=0),
                 (21, 21): GraphPair([{0: 1, 2: 0}, {0: 1, 2: -1}], 2, running_time=0),
                 (21, 37): GraphPair([{1: 0, 0: 1}, {0: 2, 2: 0}], 3, running_time=0),
                 (37, 40): GraphPair([{}], 0, running_time=0),
                 (30, 37): GraphPair([{}], 0, running_time=0),
                 (6, 6): GraphPair([{0: 5, 1: 1, 2: 2, 3: 3}, {0: 1}], 1, running_time=0),
                 (6, 39): GraphPair([{5: -1}, {5: -1}], 2, running_time=0),
                 (21, 21): GraphPair([{3: 2}, {3: 2}], 3, running_time=0),
                 (29, 29): GraphPair([{1: 1}], 4, running_time=0)}
    elif name in ['mini_debug', 'mini_debug_no-1']:
        pairs = {(30, 21): GraphPair([{0: -1, 2: -1}, {0: -1, 2: -1}], 2, running_time=0),
                 (29, 29): GraphPair([{0: -1, 2: -1}, {0: -1, 2: -1}], 2, running_time=0)}
    elif name == 'debug_single_iso':
        g = nx.Graph(gid=0)
        g.add_node(0, type='red')
        g.add_node(1, type='green')
        g.add_node(2, type='blue')
        g.add_node(3, type='orange')
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        g_copy = g.copy()
        g_copy.graph['gid'] = 1
        graphs = [RegularGraph(g), RegularGraph(g_copy)]
        pairs = {(0, 0): GraphPair([{0: 0, 1: 1, 2: 2, 3: 3}], 4, running_time=0),
                 (1, 1): GraphPair([{0: 0, 1: 1, 2: 2, 3: 3}], 4, running_time=0)}
    elif 'debug_BA' in name:
        natts = []
        params = _parse_graphs_info(name, 'debug_BA')
        train_size, test_size, num_nodes_training, num_nodes_testing = \
            int(params['train_size']), int(params['test_size']), \
            int(params['num_nodes_training']), int(params['num_nodes_testing'])

        num_graphs = 0
        train_pairs, test_pairs = {}, {}
        train_graphs, test_graphs = [], []
        for _ in range(train_size):
            g0, g1, mapping_train = gen_BA(num_nodes_training * 2, 1, num_nodes_training)
            assert len(mapping_train) == g1.number_of_nodes()  # g1 is contained in g2
            assert g0.number_of_nodes() > 0
            assert g1.number_of_nodes() > 0
            g0.graph['gid'] = num_graphs
            g1.graph['gid'] = num_graphs + 1
            train_graphs.append(RegularGraph(g0))
            train_graphs.append(RegularGraph(g1))
            train_pairs[(num_graphs, num_graphs + 1)] = GraphPair([mapping_train],
                                                                  ds_true=g1.number_of_nodes(),
                                                                  running_time=0)
            num_graphs += 2

        for _ in range(test_size):
            g0, g1, mapping_test = gen_BA(num_nodes_testing * 2, 1, num_nodes_testing)
            assert len(mapping_test) == g1.number_of_nodes()
            assert g0.number_of_nodes() > 0
            assert g1.number_of_nodes() > 0
            g0.graph['gid'] = num_graphs
            g1.graph['gid'] = num_graphs + 1
            test_graphs.append(RegularGraph(g0))
            test_graphs.append(RegularGraph(g1))
            test_pairs[(num_graphs, num_graphs + 1)] = GraphPair([mapping_test],
                                                                 ds_true=g1.number_of_nodes(),
                                                                 running_time=0)
            num_graphs += 2

        glabel = None
        # graphs = add_glabel_to_each_graph(graphs, None, glabel)
        if tvt == 'train':
            return OurDataset(name, train_graphs, natts, eatts, train_pairs, tvt, align_metric,
                              node_ordering, glabel, None)
        if tvt == 'test':
            return OurDataset(name, test_graphs, natts, eatts, test_pairs, tvt, align_metric,
                              node_ordering, glabel, None)
    else:
        raise NotImplementedError()
    # if name != 'debug_single_iso':
    #     for tvt_gs in ['train', 'test']:
    #         graphs.extend(
    #             iterate_get_graphs(join(get_data_path(), dir_name, tvt_gs),
    #                                natts=natts, eatts=eatts))
    graphs = add_glabel_to_each_graph(graphs, None, glabel)
    rtn = OurDataset(name, graphs, natts, eatts, pairs, tvt, align_metric,
                     node_ordering, glabel, None)
    if '_no-1' in name:
        for (gid1, gid2), pair in pairs.items():
            nn = min(rtn.look_up_graph_by_gid(gid1).get_nxgraph().number_of_nodes(),
                     rtn.look_up_graph_by_gid(gid2).get_nxgraph().number_of_nodes())
            pairs[(gid1, gid2)] = GraphPair([{i: i for i in range(nn)}],
                                            pair.get_ds_true())
    return rtn


def load_classic_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    node_labels = False
    if name == 'aids700nef':
        dir_name = 'AIDS700nef'
        assert glabel is None
        node_labels = True
    elif name == 'DD':
        dir_name = 'DD'
        node_labels = True

    elif name == 'imdbmulti':
        dir_name = 'IMDBMulti'
        assert glabel == 'discrete'
    elif name == 'linux':
        dir_name = 'LINUX'
        assert glabel is None
    elif name == 'ptc':
        dir_name = 'PTC'
        node_labels = True
        assert glabel is None
    elif name == 'webeasy':
        dir_name = 'WEBEASY'
        node_labels = True
        assert glabel is None
    elif name == 'nci109':
        dir_name = 'NCI109'
        node_labels = True
        assert glabel is None
    elif name == 'nci1':
        dir_name = 'NCI1'
        node_labels = True
        assert glabel is None
    elif name == 'redditmulti10k':
        dir_name = 'RedditMulti10k'
        assert glabel is None
    elif name == "mutag":
        dir_name = "MUTAG"
        node_labels = True
        assert glabel is None
    elif name == "alchemy":
        dir_name = "ALCHEMY"
        node_labels = True
        assert glabel is None
    else:
        raise NotImplementedError()

    dir_name = join(get_data_path(), dir_name)

    train_gs = iterate_get_graphs(join(dir_name, 'train'),
                                  natts=natts, eatts=eatts)
    test_gs = iterate_get_graphs(join(dir_name, 'test'),
                                 natts=natts, eatts=eatts)
    all_gs = train_gs + test_gs

    # Load graph labels.
    all_gs = add_glabel_to_each_graph(all_gs, dir_name, glabel)

    if tvt == 'train':
        graphs = train_gs
    elif tvt == 'test':
        graphs = test_gs
    elif tvt == 'all':
        graphs = all_gs
    else:
        raise NotImplementedError()

    if skip_pairs:
        graph_pairs = {}
    else:
        graph_pairs = load_classic_data_pairs(natts, graphs, dir_name)

    return OurDataset(name, graphs, natts, eatts, graph_pairs, tvt, align_metric,
                      node_ordering, glabel, None)

def load_classic_data_pairs(natts, graphs, dir_name):
    # some files may be name ...processed_node_mapping_duplicate_mappings.csv
    search_str_preprocessed = '*' + '*'.join(["mccreesh2017", "preprocessed_node_mapping", ".csv"])
    search_str_unprocessed = '*' + '*'.join(["mccreesh2017", ".csv"])
    search_str_bad_mappings = '*' + '*'.join(["mccreesh2017", "_bad_mappings.csv"])

    all_files = set(glob(join(dir_name, 'csv', search_str_unprocessed)))
    processed_files = set(glob(join(dir_name, 'csv', search_str_preprocessed)))
    bad_mapping_files = set(glob(join(dir_name, 'csv', search_str_bad_mappings)))
    files_to_process = all_files - processed_files - bad_mapping_files


    graphs_gid_dict = {nxgraph.graph['gid']: nxgraph for nxgraph in
                       map(lambda g: g.get_nxgraph(), graphs)}

    # graph_pairs = {}

    for file_path in files_to_process:
        file_size = os.stat(file_path).st_size
        res_file_name = os.path.basename(os.path.splitext(file_path)[0])
        res_path = os.path.join(dir_name, 'csv',
                                res_file_name + "_preprocessed_node_mapping.csv")
        # only process unprocessed files that are non empty
        if not os.path.exists(res_path) and not (file_size < 100):
            print("preprocessing {}".format(file_path))
            preprocess_node_mapping(file_path, graphs_gid_dict, node_labels, natts=natts)

    search_str_preprocessed_only = '*' + '*'.join(["mccreesh2017", "preprocessed_node_mapping.csv"])
    preprocessed_files = glob(join(dir_name, 'csv', search_str_preprocessed_only))
    print("Read {} preprocessed files".format(len(preprocessed_files)))
    print("Creating GraphPair objects...")

    graph_pairs = {}
    for fnum, file_name in enumerate(preprocessed_files):
        for index, chunk in enumerate(read_csv(file_name, sep=',', chunksize=1)):
            if index % 2000 == 0:
                if index % 20000 == 0:
                    print('file: {}. file {} of {}'.format(file_name, fnum + 1,
                                                           len(preprocessed_files)))
                print(index)

            gid_pair = (chunk['i_gid'][index], chunk['j_gid'][index])

            # if gid_pair in graph_pairs.keys() or (gid_pair[1], gid_pair[0]) in \
            #         graph_pairs.keys():
            #     chunk.to_csv('{}.csv'.format(join(file_name[:-4] +
            #                                       '_duplicate_mappings')),
            #                  header=False, mode='a', chunksize=1)
            # else:
            node_mappings = literal_eval(chunk['node_mapping'][index])
            running_time = float(chunk['time(msec)'][index])
            if running_time == 0.0:
                running_time = -1.0
            ds_score = int(chunk['mcs'][index])
            if gid_pair in graph_pairs.keys():
                if graph_pairs[gid_pair].running_time == -1.0:
                    graph_pairs[gid_pair].running_time = running_time
            else:
                graph_pairs[gid_pair] = \
                    GraphPair(y_true_dict_list=node_mappings,
                              ds_true=ds_score, running_time=running_time)

    return graph_pairs



def iterate_get_graphs(dir, check_connected=True, natts=(), eatts=()):
    graphs = []
    for file in sorted_nicely(glob(join(dir, '*.gexf'))):
        gid = int(basename(file).split('.')[0])
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        if not nx.is_connected(g):
            msg = '{} not connected'.format(gid)
            if check_connected:
                raise ValueError(msg)
            else:
                print(msg)
        # assumes default node mapping to convert_node_labels_to_integers

        nlist = sorted(g.nodes())
        g.graph['node_label_mapping'] = dict(zip(nlist,
                                                 range(0, g.number_of_nodes())))
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        # lnids = sorted_nicely(g.nodes()) # list of (sorted) node ids
        # # Must use sorted_nicely because otherwise may result in:
        # # ['0', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9'].
        # # Be very cautious on sorting a list of strings
        # # which are supposed to be integers.
        for i, (n, ndata) in enumerate(sorted(g.nodes(data=True))):
            assert_valid_nid(n, g)
            assert i == n
            # print(ndata)
            _remove_entries_from_dict(ndata, natts)
            # print(ndata)
        for i, (n1, n2, edata) in enumerate(sorted(g.edges(data=True))):
            assert_valid_nid(n1, g)
            assert_valid_nid(n2, g)
            # print(i, n1, n2, edata)
            _remove_entries_from_dict(edata, eatts)
            # print(i, n1, n2, edata)
        graphs.append(RegularGraph(g))
    if not graphs:
        raise ValueError('Loaded 0 graphs from {}\n'
                         'Please download the gexf-formated dataset'
                         ' from Google Drive and extract under:\n{}'.
                         format(dir, get_data_path()))
    return graphs


def preprocess_node_mapping(dataset_csv_path, graphs_dict, node_labels=False, natts=None,
                            node_ordering_helper_dict=None):
    if not os.path.exists(dataset_csv_path):
        print("{} ....is not a valid path".format(dataset_csv_path))
        return

    res_folder_path = os.path.dirname(dataset_csv_path)
    res_file_name = os.path.basename(os.path.splitext(dataset_csv_path)[0])
    res_path = os.path.join(res_folder_path, res_file_name + "_preprocessed_node_mapping")

    if not os.path.isdir(res_folder_path):
        os.mkdir(res_folder_path)

    graphs_skipped_mistmatched_nodes = 0
    graphs_skipped_mismatched_node_translation = 0
    for index, chunk in enumerate(read_csv('{}'.format(dataset_csv_path), sep=',', chunksize=1)):
        if (index - 1) % 5000 == 0:
            print("Processed {} lines".format(index))

        edge_mappings = chunk['node_mapping'][index]
        if not edge_mappings:
            continue

        node_mappings = []

        graph_i = graphs_dict[chunk['i_gid'][index]]
        graph_j = graphs_dict[chunk['j_gid'][index]]
        graph_i_node_mapping = graph_i.graph['node_label_mapping']
        graph_j_node_mapping = graph_j.graph['node_label_mapping']

        if node_labels:
            graph_attrs_i = []
            graph_attrs_j = []
            assert type(natts) is list
            for natt in natts:
                graph_attrs_i.append(nx.get_node_attributes(graph_i, natt))
                graph_attrs_j.append(nx.get_node_attributes(graph_j, natt))
        try:
            edge_mappings = literal_eval(edge_mappings)
            chunk['edge_mapping'] = [copy.deepcopy(edge_mappings)]
            for edge_mapping in edge_mappings:
                node_mapping = {}
                edge_mapping = {(graph_i_node_mapping[k[0]], graph_i_node_mapping[k[1]]):
                                    (graph_j_node_mapping[v[0]], graph_j_node_mapping[v[1]])
                                for k, v in edge_mapping.items()}

                # one node with one node mapping
                if len(edge_mapping) == 0:
                    pass
                # two nodes with two nodes mapping
                elif len(edge_mapping) == 1:
                    nodes_i, nodes_j = edge_mapping.popitem()
                    if node_labels:
                        node_i_labels = [graph_att[nodes_i[0]] for graph_att in graph_attrs_i], \
                                        [graph_att[nodes_i[1]] for graph_att in graph_attrs_i]
                        node_j_labels = [graph_att[nodes_j[0]] for graph_att in graph_attrs_j], \
                                        [graph_att[nodes_j[1]] for graph_att in graph_attrs_j]

                        if _check_node_atts(node_i_labels[0], node_j_labels[0]):
                            node_mapping[nodes_i[0]] = nodes_j[0]
                            node_mapping[nodes_i[1]] = nodes_j[1]
                        else:
                            if not _check_node_atts(node_i_labels[1], node_j_labels[0]):
                                raise ValueError(
                                    'None of the nodes in the edge mapping {}: {} match in labels '
                                    'for graphs gid {} and gid {}'.format(nodes_i, nodes_j,
                                                                          chunk['i_gid'][index],
                                                                          chunk['j_gid'][index])
                                )
                            node_mapping[nodes_i[1]] = nodes_j[0]
                            node_mapping[nodes_i[0]] = nodes_j[1]
                    else:
                        node_mapping[nodes_i[0]] = nodes_j[0]
                        node_mapping[nodes_i[1]] = nodes_j[1]

                # O.W.
                else:
                    for nodes_i, nodes_j in edge_mapping.items():  # key, value
                        for node_i in nodes_i:  # nodes in first edge
                            if node_mapping.get(node_i) is None:  # no node_mapping yet
                                node_mapping[node_i] = list(nodes_j)
                            else:
                                # if node_i == '6':
                                #     kk = node_mapping[node_i]
                                #     a = set(node_mapping[node_i])
                                #     b = set(nodes_j)
                                #     c = b.intersection(a)
                                #     d = list(c)

                                # mapping is intersection of nodes in node_j and node_mappings for node_i
                                if type(node_mapping[node_i]) == list:
                                    node_mapping[node_i] = list(set(nodes_j).intersection(
                                        set(node_mapping[node_i])))[0]
                                else:
                                    node_mapping[node_i] = list(set(nodes_j).intersection(
                                        set([node_mapping[node_i]])))[0]

                    # <class 'dict'>: {'3': '6', '5': ['6', '8'], '0': ['7', '6'], '4': '2', '7': ['0', '2']}
                    for n_1, n_2 in node_mapping.items():
                        if type(n_2) == list:
                            n_1_p, n_2_p = None, None
                            for nodes_i, nodes_j in edge_mapping.items():
                                if n_1 in nodes_i:
                                    l = list(nodes_i)
                                    l.remove(n_1)
                                    n_1_p = l[0]
                                    assert (type(node_mapping[n_1_p]) != list)
                                    n_2_p = node_mapping[n_1_p]
                            assert (n_1_p is not None and n_2_p is not None)
                            n_2 = list(n_2)
                            n_2.remove(n_2_p)
                            assert (len(n_2) == 1)
                            node_mapping[n_1] = n_2[0]
                            assert (type(node_mapping[n_1]) != list)

                if node_ordering_helper_dict is not None:
                    node_mapping = _apply_node_ordering(
                        node_mapping, graph_i.graph['gid'],
                        graph_j.graph['gid'], node_ordering_helper_dict)

                node_mappings.append(node_mapping)
        except ValueError:
            graphs_skipped_mistmatched_nodes += 1
            chunk.to_csv(
                '{}.csv'.format(os.path.join(res_folder_path, res_file_name + "_bad_mappings")),
                header=False,
                mode='a', chunksize=1)
            continue
        except KeyError:
            graphs_skipped_mismatched_node_translation += 1
            chunk.to_csv(
                '{}.csv'.format(os.path.join(res_folder_path, res_file_name + "_bad_mappings")),
                header=False,
                mode='a', chunksize=1)
            continue

        chunk['node_mapping'] = str(node_mappings)

        # chunk.to_csv('{}.csv'.format(dataset_name[:-5]), header=False, mode='a', chunksize=1)
        if index == 0:
            chunk.to_csv('{}.csv'.format(res_path), header=chunk.columns, mode='w', chunksize=1)
        else:
            chunk.to_csv('{}.csv'.format(res_path), header=False, mode='a', chunksize=1)
    if node_labels:
        print("Skipped {} graphs due to mismatched labels".format(graphs_skipped_mistmatched_nodes))
    print("SKipped {} graphs due to mismatched node renumbering translation".format(
        graphs_skipped_mistmatched_nodes))


def _apply_node_ordering(node_mapping, graph_i_gid, graph_j_gid, node_ordering_helper_dict):
    assert node_ordering_helper_dict is not None
    rtn = {}
    for k, v in node_mapping.items():
        k = node_ordering_helper_dict[graph_i_gid][k]
        v = node_ordering_helper_dict[graph_j_gid][v]
        rtn[k] = v
    # exit(-1)
    return rtn

def _check_node_atts(n1_attrs, n2_attrs):
    for n1, n2 in zip(n1_attrs, n2_attrs):
        if n1 != n2:
            return False
    return True


def _remove_entries_from_dict(d, keeps):
    for k in set(d) - set(keeps):
        del d[k]


def _parse_graphs_info(name, zeroth_word):
    sp = name.split(':')
    assert len(sp) == 2
    assert sp[0] == zeroth_word
    graphs_info = {}
    for spec in sp[1].split(','):
        ssp = spec.split('=')
        graphs_info[ssp[0]] = '='.join(ssp[1:])
    return graphs_info
