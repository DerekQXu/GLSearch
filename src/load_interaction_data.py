from collections import defaultdict
#from bignn_dataset import BiLevelDataset
from graph import BioGraph
from graph_pair import GraphPair
import re
from utils import sorted_nicely, get_data_path, assert_valid_nid, load, get_sparse_mat, save_klepto, save_pickle
from os.path import join, basename, exists
from glob import glob
import networkx as nx
from utils_interaction import create_hyperlevel_nxgraphs, get_degree_dist
import collections
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

def load_interaction_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    if 'drugbank' in name:
        dir_name = 'DrugBank'
        drugbank_dir = join(get_data_path(), dir_name)
        interaction_dir = join(drugbank_dir, 'ddi_data')
        graph_data = load(join(drugbank_dir, "klepto", 'graph_data.klepto'))
        # add_glabel_drugbank(graph_data)
        fname_to_gid_func = lambda fname: int(fname[2:])
        if 'snap' in name:
            interaction_fname = 'ddi_snap.tsv'
            parse_edge_func = parse_edges_biosnap
            data_dir = join(interaction_dir, "drugs_snap")
        elif 'deepddi' in name:
            interaction_fname = 'ddi_deepddi.csv'
            parse_edge_func = parse_edges_deepddi
            data_dir = join(interaction_dir, "drugs_deepddi")
        elif 'small' in name:
            interaction_fname = 'ddi_snap.tsv'
            parse_edge_func = parse_edges_biosnap
            data_dir = join(interaction_dir, "drugs_small")
        else:
            raise NotImplementedError('DDI of {} does not exist in {}'.format(name, interaction_dir))
        interaction_file_path = join(interaction_dir, interaction_fname)
        edge_types_edge_list, nodes = get_interaction_edgelist(interaction_file_path, parse_edge_func, eatts, fname_to_gid_func)

    elif 'decagon' in name:
        dir_name = 'Decagon'
        fname_to_gid_func = lambda fname: int(fname)
        decagon_dir = join(get_data_path(), dir_name)
        data_dir = decagon_dir
        interaction_dir = join(decagon_dir, "ddi_data")
        graph_data = load(join(decagon_dir, "klepto", 'graph_data.klepto'))
        ddi_klepto_path = join(data_dir, "klepto", "combo.klepto")
        if not exists(ddi_klepto_path):
            combo2stitch, combo2se, se2name = load_combo_se(join(interaction_dir, "bio-decagon-combo.csv"))
            save = {"combo2stitch":combo2stitch, "combo2se":combo2se, "se2name": se2name}
            save_klepto(save, ddi_klepto_path, True)

        else:
            save = load(ddi_klepto_path)
            combo2stitch, combo2se, se2name = save["combo2stitch"], save["combo2se"], save["se2name"]
        edge_types_edge_list, nodes = get_interaction_edge_list_decagon(combo2se, combo2stitch)
        count_poly_se = collections.Counter({se: sum(map(len, el.values())) for se, el in edge_types_edge_list.items()})
        most_common_se = count_poly_se.most_common(963)
        edge_types_edge_list = {se: edge_types_edge_list[se] for se, cnt in most_common_se}
        node_feats_klepto_path = join(data_dir, "klepto", "node_feats.klepto")

        if not exists(node_feats_klepto_path):
            stitch2se, se2name = load_mono_se(join(interaction_dir, "bio-decagon-mono.csv" ))
            save = {"stitch2se": stitch2se, "se2name": se2name}
            save_klepto(save, node_feats_klepto_path, True)
        else:
            save = load(node_feats_klepto_path)
            stitch2se = save["stitch2se"]
            se2name = save["se2name"]
    elif 'drugcombo' in name:
        dir_name = 'DrugCombo'
        drugcombo_dir = join(get_data_path(), dir_name)
        graph_data = load(join(drugcombo_dir, "klepto", 'graph_data.klepto'))
        data_dir = drugcombo_dir
        interaction_dir = join(drugcombo_dir, 'ddi_data')
        interaction_file_path = join(interaction_dir, 'Syner&Antag_voting.csv')
        drugname_to_cid = load(join(drugcombo_dir, 'klepto', 'drug_name_to_cid'))
        edge_to_gid_func = lambda x: int(drugname_to_cid[x.lower()][4:])
        fname_to_gid_func = lambda x: int(x[4:])
        num_pairs_synergy_antagonism = count_pairs_synergy_antagonism(interaction_file_path)
        edge_types_edge_list, nodes = get_interaction_edgelist(interaction_file_path, parse_edges_drugcombo,
                                                               True, edge_to_gid_func, skip_first_line=True,
                                                               num_pairs_synergy_antagonism=num_pairs_synergy_antagonism)
    else:
        raise NotImplementedError
    # edge type to dict of node1 to node2 edge respresentation

    graphs = iterate_get_graphs(data_dir, graph_data, nodes, fname_to_gid_func, natts=natts)
    pairs, graph_ids, edge_types_edge_list_filtered = get_graph_pairs(edge_types_edge_list, graphs)
    sparse_node_feat = None
    hyper_edge_labels = {'interaction': 1, 'no_interaction': 0}
    if 'drugbank' in name:
        sparse_node_feat, gid_to_idx = get_drugbank_node_feats(graph_data, graph_ids, fname_to_gid_func)
        # assert(sparse_node_feat.shape[0] == len(graph_ids))
        # assert(set(gid_to_idx.keys()) == set(graph_ids))
    if 'decagon' in name:
        sparse_node_feat, gid_to_idx = get_decagon_node_feats(stitch2se, se2name, graph_ids)
        assert(sparse_node_feat.shape[0] == len(graph_ids))
        assert(set(gid_to_idx.keys()) == set(graph_ids))

    if 'drugcombo' in name:
        sparse_node_feat, gid_to_idx = get_drugbank_node_feats(graph_data, graph_ids, fname_to_gid_func)
        for pair in pairs.values():
            if next(iter(pair.edge_types)) == 'antagonism':
                pair._GraphPair__ds_true = 2
        hyper_edge_labels = {'antagonism': 2, 'synergy': 1, 'no_interaction': 0}

    if not all([exists(join(data_dir, "ddi_graphs", edge_type + "_ddi.gexf"))
            for edge_type in edge_types_edge_list_filtered.keys()]):
        hyperlevel_nxgraphs = create_hyperlevel_nxgraphs(name, graph_ids, edge_types_edge_list_filtered,
                                                         data_dir=data_dir, write_graphs=True)
    else:
        hyperlevel_nxgraphs = {}
        for edge_type in edge_types_edge_list_filtered.keys():
            hyperlevel_nxgraphs[edge_type] = nx.read_gexf(join(data_dir, "ddi_graphs", edge_type + "_ddi.gexf"))

    graphs = [graphs[gid] for gid in sorted(graph_ids)]
    node_degree_dists = {edge_type: get_degree_dist(nx_graph) for edge_type, nx_graph in hyperlevel_nxgraphs.items()}
    return BiLevelDataset(name, graphs, natts, hyper_edge_labels, eatts, pairs, tvt, align_metric, node_ordering, glabel,
                      hyperlevel_nxgraphs=hyperlevel_nxgraphs, node_degree_dists=node_degree_dists,
                      sparse_node_feat=sparse_node_feat)

def get_decagon_node_feats(stitch2se, se2name, gids):
    stitch_to_gid = lambda x: int(x[3:])
    se_to_idx = {se: i for i, se in enumerate(list(se2name.keys()))}
    gid_to_se = {stitch_to_gid(stitch): se for stitch, se in stitch2se.items()}
    gid_to_idx = {gid: i for i, gid in enumerate(sorted(list(gids)))}
    sparse_node_feat = get_sparse_mat(gid_to_se, gid_to_idx, se_to_idx)
    return sparse_node_feat, gid_to_idx

def get_drugbank_node_feats(graph_data, gids, fname_to_gid_func):
    gid_to_idx = {gid: i for i, gid in enumerate(sorted(list(gids)))}
    gid_graph_data = {fname_to_gid_func(id): g_data for id, g_data in graph_data.items()}
    mats = {}
    for feat_shape in list(gid_graph_data.values())[0]['drug_feat']:
        mat = np.zeros((len(gids), int(feat_shape)))
        for gid in gids:
            mat[gid_to_idx[gid]] = gid_graph_data[gid]["drug_feat"][feat_shape]
        mats[feat_shape] = csr_matrix(mat)

    return mats, gid_to_idx


def get_interaction_edge_list_decagon(combo2se, combo2stitch):
    edge_types_edge_list = defaultdict(lambda: defaultdict(set))
    nodes = set()
    cid_to_gid = lambda x: int(x[3:])
    for pair_str, side_effects in combo2se.items():
        pair = tuple(combo2stitch[pair_str])
        gid1 = cid_to_gid(pair[0])
        gid2 = cid_to_gid(pair[1])
        for se in side_effects:
            edge_types_edge_list[se][gid1].add(gid2)
        nodes.add(gid1)
        nodes.add(gid2)
    return edge_types_edge_list, nodes

def add_glabel_drugbank(graph_data):
    approved = 0
    not_approved = 0
    for db_id, data in graph_data.items():
        # print(data['db_grp'])
        if "approved" in  data['db_grp']:
            graph_data[db_id]['glabel'] = 1
            approved += 1
        else:
            graph_data[db_id]['glabel'] = 0
            not_approved += 1
    print("drugbank num approved: ", approved)
    print("drugbank num not approved: ", not_approved)


def get_graph_pairs(edge_types_edge_list, graphs):
    graph_pairs = {}
    no_graph_structures = set()
    final_graphs = set()
    edge_types_edge_list_filtered = defaultdict(lambda: defaultdict(set))
    for edge_type, edge_list in tqdm(edge_types_edge_list.items()):
        for gid1, gid2s in edge_list.items():
            if gid1 not in graphs.keys():
                no_graph_structures.add(gid1)
                continue
            graph1 = graphs[gid1]
            for gid2 in gid2s:
                gid_pair = tuple(sorted([gid1, gid2]))
                if gid_pair not in graph_pairs.keys():
                    if gid2 not in graphs.keys():
                        no_graph_structures.add(gid2)
                        continue
                    graph2 = graphs[gid2]
                    final_graphs.add(gid1)
                    final_graphs.add(gid2)
                    graph_pairs[gid_pair] = GraphPair(ds_true=1, g1=graph1, g2=graph2, edge_types=set([edge_type]))
                else:
                    graph_pairs[gid_pair].edge_types.add(edge_type)
                edge_types_edge_list_filtered[edge_type][gid1].add(gid2)
    return graph_pairs, final_graphs, edge_types_edge_list_filtered



def get_interaction_edgelist(file_path, parse_edges_func, eatts, edge_to_gid_func, skip_first_line=False, **kwargs):
    # assume each line in file is an edge, parse it using parse_edge_func
    edge_types_edge_list = defaultdict(lambda: defaultdict(list))
    nodes = set()
    skipped = set()
    with open(file_path, 'r') as f:
        readlines = f.readlines() if not skip_first_line else list(f.readlines())[1:]
        for i, line in enumerate(readlines):
            edge, edge_type = parse_edges_func(line, **kwargs)
            if edge:
                try:
                    e1 = edge_to_gid_func(edge[0])
                    e2 = edge_to_gid_func(edge[1])
                except KeyError as e:
                    skipped.add(str(e))
                    continue
                if eatts and edge_type:
                    edge_types_edge_list[edge_type][e1].append(e2)
                else:
                    edge_types_edge_list['default'][e1].append(e2)
                nodes.add(e1)
                nodes.add(e2)
    print("number skipped: ", len(skipped))
    return edge_types_edge_list, nodes


def count_pairs_synergy_antagonism(file_path):
    count_syn_ant = defaultdict(lambda: defaultdict(int))
    with open(file_path, 'r') as f:
        for i, line in enumerate(list(f.readlines())[1:]):
            line = line.split(',')
            count_syn_ant[tuple(sorted([line[1], line[2]]))][line[-1]] += 1
    return count_syn_ant


def parse_edges_biosnap(line):
    return line.rstrip('\n').split('\t'), None


def parse_edges_drugcombo(line, **kwargs):
    label_counts = kwargs['num_pairs_synergy_antagonism']
    line = line.split(',')
    drugs = [line[1], line[2]]
    if label_counts[tuple(sorted(drugs))]['synergy\n'] >= label_counts[tuple(drugs)]['antagonism\n']:
        label = 'synergy'
    else:
        label = 'antagonism'
    return drugs, label

def parse_edges_deepddi(line):
    line = line.split(',')[0]
    pattern = 'DB[0-9]{5}'
    drugs = list(set(re.findall(pattern, line)))
    edge = None
    if drugs:
        if line.find(drugs[0]) < line.find(drugs[1]):
            drug1, drug2 = drugs[0], drugs[1]
        else:
            drug1, drug2 = drugs[1], drugs[0]
        temp = re.sub(drug1, 'D1', line)
        edge = re.sub(drug2, 'D2', temp)
    return drugs, edge


def add_graph_data_to_nxgraph(g, graph_data):
    if graph_data:
        for k,v in graph_data.items():
            g.graph[k] = v

def iterate_get_graphs(dir, graph_data, nodes, fname_to_gid_func, check_connected=False, natts=(), eatts=()):
    graphs = {}
    not_connected = []
    no_edges = []
    graphs_not_in_edge_list = []
    for file in tqdm(sorted_nicely(glob(join(dir, '*.gexf')))):
        fname = basename(file).split('.')[0]
        gid = fname_to_gid_func(fname)
        if gid not in nodes:
            graphs_not_in_edge_list.append(fname)
            continue
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        is_connected = True
        if not nx.is_connected(g):
            msg = '{} not connected'.format(gid)
            if check_connected:
                raise ValueError(msg)
            else:
                not_connected.append(fname)
                is_connected = False
        # assumes default node mapping to convert_node_labels_to_integers

        nlist = sorted(g.nodes())
        g.graph['node_label_mapping'] = dict(zip(nlist,
                                                 range(0, g.number_of_nodes())))
        add_graph_data_to_nxgraph(g, graph_data[fname])
        g = nx.convert_node_labels_to_integers(g, ordering="sorted")
        if len(g.edges) == 0:
            no_edges.append(fname)
            continue
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
        graphs[gid] = BioGraph(g, is_connected)
    print("total graphs with edges: {}\nnon connected graphs: {}".format(len(graphs), len(not_connected)))
    print("not connected ids: ", not_connected)
    print("num no edges: ", len(no_edges), "\nno edges ids: ", no_edges)
    if not graphs:
        raise ValueError('Loaded 0 graphs from {}\n'
                         'Please download the gexf-formated dataset'
                         ' from Google Drive and extract under:\n{}'.
                         format(dir, get_data_path()))
    return graphs

def _remove_entries_from_dict(d, keeps):
    for k in set(d) - set(keeps):
        del d[k]


# Returns dictionary from combination ID to pair of stitch IDs,
# dictionary from combination ID to list of polypharmacy side effects,
# and dictionary from side effects to their names.
def load_combo_se(fname='bio-decagon-combo.csv'):
    combo2stitch = {}
    combo2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    print('Reading: {}'.format(fname))
    fin.readline()
    for line in fin:
        stitch_id1, stitch_id2, se, se_name = line.strip().split(',')
        combo = stitch_id1 + '_' + stitch_id2
        combo2stitch[combo] = [stitch_id1, stitch_id2]
        combo2se[combo].add(se)
        se2name[se] = se_name
    fin.close()
    n_interactions = sum([len(v) for v in combo2se.values()])
    print('Drug combinations: {} Side effects: {}'.format(len(combo2stitch), len(se2name)))
    print('Drug-drug interactions: {}'.format(n_interactions))
    return combo2stitch, combo2se, se2name

# Returns networkx graph of the PPI network
# and a dictionary that maps each gene ID to a number
def load_ppi(fname='bio-decagon-ppi.csv'):
    fin = open(fname)
    print('Reading: {}'.format(fname))
    fin.readline()
    edges = []
    for line in fin:
        gene_id1, gene_id2= line.strip().split(',')
        edges += [[gene_id1,gene_id2]]
    nodes = set([u for e in edges for u in e])
    print('Edges: {}'.format(len(edges)))
    print('Nodes: {}'.format(len(nodes)))
    net = nx.Graph()
    net.add_edges_from(edges)
    net.remove_nodes_from(nx.isolates(net))
    net.remove_edges_from(net.selfloop_edges())
    node2idx = {node: i for i, node in enumerate(net.nodes())}
    return net, node2idx

# Returns dictionary from Stitch ID to list of individual side effects,
# and dictionary from side effects to their names.
def load_mono_se(fname='bio-decagon-mono.csv'):
    stitch2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    print('Reading: {}'.format(fname))
    fin.readline()
    for line in fin:
        contents = line.strip().split(',')
        stitch_id, se, = contents[:2]
        se_name = ','.join(contents[2:])
        stitch2se[stitch_id].add(se)
        se2name[se] = se_name
    return stitch2se, se2name

# Returns dictionary from Stitch ID to list of drug targets
def load_targets(fname='bio-decagon-targets.csv'):
    stitch2proteins = defaultdict(set)
    fin = open(fname)
    print('Reading: {}'.format(fname))
    fin.readline()
    for line in fin:
        stitch_id, gene = line.strip().split(',')
        stitch2proteins[stitch_id].add(gene)
    return stitch2proteins

# Returns dictionary from side effect to disease class of that side effect,
# and dictionary from side effects to their names.
def load_categories(fname='bio-decagon-effectcategories.csv'):
    se2name = {}
    se2class = {}
    fin = open(fname)
    print('Reading: {}'.format(fname))
    fin.readline()
    for line in fin:
    	se, se_name, se_class = line.strip().split(',')
    	se2name[se] = se_name
    	se2class[se] = se_class
    return se2class, se2name

def stitch_flat_to_pubchem(cid):
    assert cid.startswith('CID')
    return int(cid[3:]) - 1e8

def stitch_stereo_to_pubchem(cid):
    assert cid.startswith('CID')
    return int(cid[3:])

def pubchem_to_stitch_stereo(pid):
    return 'CID' + '{0:0>9}'.format(pid)

def pubchem_to_stitch_flat(pid):
    return 'CID1' + '{0:0>8}'.format(pid)

