import sys
import re
import networkx as nx
import matplotlib.pyplot as plt

def process_mapping(file_name):
    fp = open(file_name)
    mapping = fp.readline()
    raw_mapping = [[int(node) for node in nodes[1:-1].split(',')] for nodes in re.sub(' -> ', ',', mapping).split(' ')[:-1]]
    mapping = {}
    for node_pair in raw_mapping:
       mapping[node_pair[1]] = node_pair[0] 
    return mapping

def get_node(node_mapping, node):
    for i in range(len(node_mapping)):
        if list(node_mapping)[i] == node:
            return i

def show_graph(g):
    nx.draw(g, node_size=1)
    plt.show()

g1 = nx.read_gexf(sys.argv[2])
g2 = nx.read_gexf(sys.argv[3])
mapping = process_mapping(sys.argv[1])

feat_dict_1 = {}
node_mapping = {}
featureless = False
for node in range(g1.number_of_nodes()):
    node_mapping[str(node)] = node
    del g1.nodes[str(node)]['label']
    if not featureless:
        try:
            feat_dict_1[node] = g1.nodes[str(node)]['feature']
        except:
            featureless = True

g1 = nx.relabel_nodes(g1, node_mapping)
for edge in g1.edges:
    del g1.edges[edge]['id']

feat_dict_2 = {}
node_mapping = {}
for node in range(g2.number_of_nodes()):
    node_mapping[str(node)] = node
    del g2.nodes[str(node)]['label']
    if not featureless:
        feat_dict_2[node] = g2.nodes[str(node)]['feature']

g2 = nx.relabel_nodes(g2, node_mapping)
for edge in g2.edges:
    del g2.edges[edge]['id']

if not featureless:
    edge_g1 = {}
    edge_g2 = {}
    for edge in g1.edges:
        edge_g1[edge] = g1.edges[edge]['feature'] % 100
    for edge in g2.edges:
        edge_g2[edge] = g2.edges[edge]['feature'] % 100

pos_g1 = nx.spring_layout(g1)
pos_g2 = nx.spring_layout(g2)
for node in g2.nodes:
    pos_g2[node] = 0,0
color_g1 = []
color_g2 = []

for node in range(g1.number_of_nodes()):
    color_g1.append('blue')
for node in range(g2.number_of_nodes()):
    color_g2.append('blue')

for node in mapping.keys():
    pos_g2[node] = pos_g1[mapping[node]]
    color_g2[get_node(g2.nodes,node)] = 'red'
    color_g1[get_node(g1.nodes,mapping[node])] = 'red'

if featureless:
    plt.subplot(121)
    nx.draw(g1, node_color=color_g1, pos=pos_g1, with_labels=True)
    plt.subplot(122)
    nx.draw(g2, node_color=color_g2, pos=pos_g2, with_labels=True)
else:
    plt.subplot(121)
    nx.draw(g1, node_color=color_g1, pos=pos_g1, with_labels=True, labels = feat_dict_1)
    nx.draw_networkx_edge_labels(g1, pos=pos_g1, edge_labels=edge_g1)
    plt.subplot(122)
    nx.draw(g2, node_color=color_g2, pos=pos_g2, with_labels=True, labels = feat_dict_2)
    nx.draw_networkx_edge_labels(g2, pos=pos_g2, edge_labels=edge_g2)
plt.show()