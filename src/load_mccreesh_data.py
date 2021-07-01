from graph_pair import GraphPair
from graph import RegularGraph
from utils import get_model_path, get_data_path
from dataset import OurDataset
import os
import sys
from os.path import join
# from src.scripts.mccreesh_savers.helper_functions import *

def check_if_exists_preprocessed_data(save_path):
    return os.path.isdir(save_path) and len(os.listdir(join(save_path, 'mappings'))) > 0

def load_mccreesh_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert tvt is not None
    assert name in ['mcs33v', 'mcs33ve', 'mcs33ve-connected', 'mcs33ved', 'mcsplain', 'mcsplain-connected', 'sip']
    assert glabel is None

    # create data path
    path = join(get_model_path(), 'mccreesh2017')
    save_path = join(get_data_path(), name)
    limit_output = sys.maxsize

    # parameters for dataset object
    natts = [] if (name in ['mcsplain', 'mcsplain-connected', 'sip']) else ['feature']
    eatts = [] if (name in ['mcsplain', 'mcsplain-connected', 'sip']) else ['feature']
    graphs = []
    pairs = {}

    if check_if_exists_preprocessed_data(save_path):
        # process intermediary data
        print('loading saved data')
        for graph_file in os.listdir(join(save_path, 'graphs')):
            gid = graph_file.split('.')[0]
            g = nx.read_gexf(join(save_path, 'graphs', graph_file))
            g.graph['gid'] = int(gid)
            node_mapping = {}
            for node in g.nodes():
                node_mapping[node] = int(node)
                del g.nodes[node]['label']
            g = nx.relabel_nodes(g, node_mapping)
            for edge in g.edges:
                del g.edges[edge]['id']

            graphs.append(RegularGraph(g))
        for mapping_file in os.listdir(join(save_path, 'mappings')):
            f = open(join(save_path, 'mappings', mapping_file), 'r')
            data = f.read()
            f.close()
            mapping_list, has_true_matching = get_processed_mapping(data)
            gids = mapping_file.split('.')[0]
            gid1, gid2 = gids.split('_')
            pairs.update({(int(gid1), int(gid2)): GraphPair(g1=None, g2=None, y_true_dict_list=mapping_list, has_true_matching=has_true_matching)})
    else:
        # process raw data
        if os.path.exists(save_path) and os.path.isdir(save_path):
            pipe = subprocess.Popen(['rm', '-r', save_path])
            pipe.wait()

        pipe = subprocess.Popen(['mkdir', save_path])
        pipe.wait()
        pipe = subprocess.Popen(['mkdir', join(save_path, 'graphs')])
        pipe.wait()
        pipe = subprocess.Popen(['mkdir', join(save_path, 'mappings')])
        pipe.wait()

        # loading data straight from mccreesh repo
        print('loading raw data')
        ignore_list = get_ignore_list(name)
        graph_pairs = get_graph_pairs('runtimes.data', path=path, exp_name=name, ignore_list=ignore_list,
                                      timeout_time=1000000)
        mappings, algorithms = get_mappings(graph_pairs, exp_name=name, path=path)

        file2gid = {}
        gid = 0

        num_ground_truth_not = 0
        num_ground_truth = 0

        for i in range(min(len(mappings), limit_output)):
            file_1, file_2, graph_dataset = get_graph_filenames(graph_pairs[i][0], path + '/')

            raw_mapping = mappings[i]

            # process graphs
            if file_1 not in file2gid.keys():
                try:
                    g1 = get_processed_graph(file_1, graph_dataset, natts == [])
                    file2gid[file_1] = gid
                    g1.graph['gid'] = gid
                    if (g1.number_of_nodes()-1 < max(g1.nodes)):
                        mapping_dict = {}
                        offset = 0
                        for j in range(max(g1.nodes)+1):
                            if j not in g1.nodes():
                                offset += 1
                            elif offset != 0:
                                mapping_dict[j] = j-offset
                                if raw_mapping is not None:
                                    raw_mapping.replace('('+str(j), '('+str(j-offset))
                        g1 = nx.relabel_nodes(g1, mapping_dict)
                        print('relabeled nodes from:' + file_1)
                        print(mapping_dict)
                    assert(g1.number_of_nodes()-1 == max(g1.nodes))
                except:
                    print('Exception: could not load graph: ' + file_1)
                    continue
                try:
                    graphs.append(RegularGraph(g1))
                    nx.write_gexf(g1, join(save_path, 'graphs', '{}.gexf'.format(gid)))
                except:
                    print('Exception: could not process graph pair: ' + graph_pairs[i][0])
                    continue
                gid += 1

            if file_2 not in file2gid.keys():
                try:
                    g2 = get_processed_graph(file_2, graph_dataset, natts == [])
                    file2gid[file_2] = gid
                    g2.graph['gid'] = gid
                    if (g2.number_of_nodes() - 1 < max(g2.nodes)):
                        mapping_dict = {}
                        offset = 0
                        for j in range(max(g2.nodes) + 1):
                            if j not in g2.nodes():
                                offset += 1
                            elif offset != 0:
                                mapping_dict[j] = j - offset
                                if raw_mapping is not None:
                                    raw_mapping.replace(str(j) + ')', str(j - offset) + ')')
                        g2 = nx.relabel_nodes(g2, mapping_dict)
                        print('relabeled nodes from:' + file_2)
                        print(mapping_dict)
                    assert (g2.number_of_nodes() - 1 == max(g2.nodes))
                except:
                    print('Exception: could not load graph: ' + file_2)
                    continue
                try:
                    graphs.append(RegularGraph(g2))
                    nx.write_gexf(g2, join(save_path, 'graphs', '{}.gexf'.format(gid)))
                except:
                    print('Exception: could not process graph pair: ' + graph_pairs[i][0])
                    continue
                gid += 1

            gid1 = file2gid[file_1]
            gid2 = file2gid[file_2]

            # process mappings
            if raw_mapping is None:
                raw_mapping = 'NULL'
                has_true_matching = False
                num_ground_truth_not += 1
            else:
                has_true_matching = True
                num_ground_truth += 1

            fp = open(join(save_path, 'mappings', '{}_{}.txt'.format(gid1, gid2)), 'w')
            fp.write(raw_mapping)
            fp.close()

            mapping_list, has_true_matching = get_processed_mapping(raw_mapping)
            pairs.update({(gid1, gid2): GraphPair(g1=None, g2=None, y_true_dict_list=mapping_list,
                                                  has_true_matching=has_true_matching)})

        print('{}/{} pairs have no ground truth'.format(
            num_ground_truth_not, num_ground_truth_not+num_ground_truth))

        pipe = subprocess.Popen(['rm', '-r', 'temp', '__pycache__'])
        pipe.wait()

    print('data loading finished')
    return OurDataset(name, graphs, natts, eatts, pairs, tvt, align_metric,
                      node_ordering, None, None)