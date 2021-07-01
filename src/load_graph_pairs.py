from utils import get_src_path, load, save, sorted_nicely, get_data_path
from scripts.node_mapping_preprocess import preprocess_node_mapping
import pandas as pd
from graph_pair import *
from ast import literal_eval


# interchangeable dataset_path
dataset_path = '{}/{}/{}'.format(get_src_path(), 'scripts', 'mcs_data')
def load_graph_pairs(nx_graphs, dataset_path=dataset_path):
    graphs_gid_dict = {nx_graph.graph['gid']: nx_graph for nx_graph in nx_graphs}
    preprocess_node_mapping(dataset_path)
    graph_pairs = []
    for index, chunk in enumerate(pd.read_csv('{}.csv'.format(dataset_path[:-5]), sep=',', chunksize=1)):
        node_mappings = literal_eval(chunk['node_mapping'][index])

        gid1 = chunk['i_gid'][index]
        gid2 = chunk['j_gid'][index]
        g1 = graphs_gid_dict[gid1]
        g2 = graphs_gid_dict[gid2]
        #added maping in load_data which calls load_graph_pairs
        g1_node_label_mapping = g1.graph['node_label_mapping']
        g2_node_label_mapping = g2.graph['node_label_mapping']

        node_mappings_relabeled = []
        for node_mapping in node_mappings:
            node_mapping_relabeled ={g1_node_label_mapping[k]: g2_node_label_mapping[v] for k, v  in node_mapping.items()}
            node_mappings_relabeled.append(node_mapping_relabeled)



        graph_pair = GraphPair(y_true_dict_list=node_mappings_relabeled)
        graph_pairs.append(graph_pair)

    return graph_pairs

