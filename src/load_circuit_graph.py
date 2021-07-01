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
import csv


def load_circuit_graph(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
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
    elif 'circuit' in name:
        dir_name = ""
        node_labels = True
        assert glabel is None
    else:
        raise NotImplementedError()

    # get the location of the files we are loading
    file_path_list = ['lay.net', 'src.net']
    directory = '/home/username/Documents/GraphMatching/data/mentor_graphics/' \
                'mentor_graphics_ucla_test1/Test1/'
    file_path_list = [directory+file_path for file_path in file_path_list]

    # parse the files to get RegularGraph objexts
    port_representation = 'node_labelled_sparse'
    use_net_ids = True # domain specific heuristic (if net natts exist in both graphs -> they must match)
    graphs = get_graphs(file_path_list, use_net_ids, port_representation)

    # TODO: this is for Fedor's initial test case...
    #   I am only loading 1 graph pair!! -> remove hardcoding later!!!!!!!!!!!!!!!!!!!!!
    # construct the graph pairs from the dataset
    assert len(graphs) == 2
    mapping = [{0:0}] # MAPPING CANNOT BE {} -> messes up evaluation code!
    graph_pairs = {(0,1): GraphPair(mapping, ds_true=0, running_time=0)}

    if use_net_ids:
        for pair in graph_pairs:
            i,j = pair
            g1, g2 = graphs[i], graphs[j]

            # make sure we have the right graph pairs
            g1_nxgraph, g2_nxgraph = g1.get_nxgraph(), g2.get_nxgraph()
            assert g1_nxgraph.graph['gid'] == i and \
                   g2_nxgraph.graph['gid'] == j

            # find natts of net nodes that exist in both graphs
            g1_net_atts, g2_net_atts = set(), set()
            for nid in range(g1_nxgraph.number_of_nodes()):
                if not g1_nxgraph.nodes[nid]['is_device']:
                    g1_net_atts.add(g1_nxgraph.nodes[nid]['name'])
            for nid in range(g2_nxgraph.number_of_nodes()):
                if not g2_nxgraph.nodes[nid]['is_device']:
                    g2_net_atts.add(g2_nxgraph.nodes[nid]['name'])
            shared_net_atts = g1_net_atts.intersection(g2_net_atts)

            # only keep natts of net nodes that exist in both graphs
            for nid in range(g1_nxgraph.number_of_nodes()):
                if g1_nxgraph.nodes[nid]['name'] not in shared_net_atts and \
                        not g1_nxgraph.nodes[nid]['is_device']:
                    g1.nxgraph.nodes[nid]['name'] = ''
            for nid in range(g2_nxgraph.number_of_nodes()):
                if g2_nxgraph.nodes[nid]['name'] not in shared_net_atts and \
                        not g2_nxgraph.nodes[nid]['is_device']:
                    g2.nxgraph.nodes[nid]['name'] = ''

    # Load graph labels.
    # graphs = add_glabel_to_each_graph(graphs, dir_name, glabel)

    return OurDataset(name, graphs, natts, eatts, graph_pairs, tvt, align_metric,
                      node_ordering, glabel, None)

def get_graphs(file_path_list, use_net_ids, port_representation, check_connected=True):
    graphs = []
    header_lines = 1
    cur_gid = 0
    ##################################################################
    # Current Version treats port-id's as edge labels...
    # This assumes and the devices and nets are bipartite
    ##################################################################

    # create graph from single csv file
    for file_path in file_path_list:
        g = nx.Graph()

        # read SPICE filae
        fp = open(file_path)
        csv_reader = csv.reader(fp, delimiter=' ')

        # skip header lines
        for _ in range(header_lines):
            next(csv_reader)

        # construct the graph
        cur_nid = 0
        net2nid = {}
        for line in csv_reader:
            if len(line) == 1:
                assert line[0] == '.ENDS'
                break
            # both device and nets are nodes
            # email: "[instances can be any string] but the first character matters"
            device_instance = line[0][0]

            # find the port labels of supported device types:
            #   M (mosfet)
            if device_instance == 'M':
                port_labels = ['Drain_Source', 'Gate', 'Drain_Source', 'Bulk']
                assert '' not in port_labels # this would mess up how we represent devices...
                nets = line[1:5]
                # we can ignore properties for now (TODO: parse floating values)
                # properties = [float(prop) for prop in line[6:8]]
                device_type = line[5]
            else:
                assert False
            assert len(nets) == len(port_labels)

            # add new device node to graph
            device_nid = cur_nid
            g.add_node(device_nid, is_device=True, name=device_instance, type=device_type, port='')
            cur_nid += 1

            # add net node to graph if it does not already exist
            for net in nets:
                if net not in net2nid:
                    net2nid[net] = cur_nid
                    # OPTIONAL: use ground truth net mappings
                    net_label = net if use_net_ids else ''
                    g.add_node(cur_nid, is_device=False, name=net_label, type='', port='')
                    cur_nid += 1

            # add edge between connected devices and nets
            for i, net in enumerate(nets):
                port_label = port_labels[i]
                if port_representation == 'edge_labelled':
                    #TODO: figure out adding edge types
                    g.add_edge(device_nid, net2nid[net], type=port_label)
                elif port_representation == 'node_labelled_sparse':
                    port_nid = cur_nid
                    g.add_node(port_nid, is_device=True, name=device_instance, type=device_instance, port=port_label)
                    cur_nid += 1
                    g.add_edges_from(
                        [(device_nid, port_nid), (port_nid, net2nid[net])])

        # check connectivity
        g.graph['gid'] = cur_gid
        cur_gid += 1
        if not nx.is_connected(g):
            msg = '{} not connected'.format(cur_gid-1)
            if check_connected:
                raise ValueError(msg)
            else:
                print(msg)

        graphs.append(RegularGraph(g))

    if len(graphs) == 0:
        raise ValueError('Loaded 0 graphs from {}. Please check if directory is not empty.\n'.format(file_path_list))
    return graphs