import csv
import pickle
import traceback
from itertools import zip_longest
from os.path import join
from os import path, remove
from time import time
from collections import defaultdict
import networkx as nx
import numpy as np
from bignn_dataset import OurStringPdbDataset
from utils_interaction import create_hyperlevel_nxgraphs, get_degree_dist
from graph import PDBGraph
from graph_pair import GraphPair
from utils import get_data_path, get_sparse_mat, get_ppi_data_path, save, load
import scipy.sparse as sp
from scipy.spatial.distance import euclidean
from itertools import combinations
import pandas as pd
from tqdm import tqdm
import gc


def load_pdb_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    species = name.split('_')[3]
    print("Loading string-pdb mapping...")
    total_string_pdb_mapping = _get_string_pdb_mapping(species)
    print("Loading string-seq mapping...")
    string_seq_map = _get_string_sequence(species)
    print("iter_graph_pairs...")
    graphs, graph_pairs, edge_list_excluded, edge_list_remained, \
    total_string_list, pdb_to_idx_map, string_pdb_mapping_filtered = \
        _iterate_get_graphs_pairs(total_string_pdb_mapping, glabel, species)

    graph_pairs = filter_duplicate_edges(graph_pairs)

    # convert str type string_ids to int
    sid_to_int_map = dict(
        zip(sorted(total_string_list), range(0, len(total_string_list))))

    int_to_sid_map = {v: k for k, v in sid_to_int_map.items()}

    # convert (sid1 (str), sid2 (str)) to (sid1 (int), str (int))
    graph_pairs_int = {(sid_to_int_map[gid1], sid_to_int_map[gid2]): pair
                       for (gid1, gid2), pair in graph_pairs.items()}

    # edge_list_aggregated: source: [target1, target2m ...]
    edge_list_aggregated = get_edge_from_pairs(graph_pairs_int.keys())
    sparse_node_feat = \
        get_ppi_sparse_node_feat(edge_list_aggregated,
                                 total_string_list)

    hyperlevel_nxgraphs = create_hyperlevel_nxgraphs(name, total_string_list,
                                                     edge_list_aggregated,
                                                     data_dir=None,
                                                     write_graphs=False)
    node_degree_dists = {edge_type: get_degree_dist(nx_graph) for
                         edge_type, nx_graph in hyperlevel_nxgraphs.items()}

    hyper_edge_labels = {"nb": 0, "fu": 1, "co": 2, "ce": 3, "ex": 4,
                         "da": 5, "tx": 6}

    print("creating pdb dataset")
    d = OurStringPdbDataset(name, graphs, natts, hyper_edge_labels, eatts,
                            graph_pairs_int, tvt, align_metric, node_ordering,
                            glabel, string_pdb_mapping_filtered,
                            string_seq_map, total_string_list,
                            sid_to_int_map, int_to_sid_map, pdb_to_idx_map,
                            hyperlevel_nxgraphs, node_degree_dists,
                            sparse_node_feat)

    # d._gen_stats()
    d.print_stats()

    return d


def filter_duplicate_edges(pairs):
    temp_pair = {}
    for (gid1, gid2), pair in pairs.items():
        if (gid2, gid1) not in temp_pair.keys():
            temp_pair[(gid1, gid2)] = pair
    return temp_pair


def _get_string_pdb_mapping(species=None):
    if species is None:
        protein_dir = join(get_data_path(), "PPI_Datasets",
                           "protein_identifier", "protein_identifier.pickle")
        with open(protein_dir, "rb") as f:
            protein_identifer = pickle.load(f)

    else:
        protein_dir = join(get_data_path(), "PPI_Datasets", "preprocess",
                           species, "{}_protein_map.pickle".format(species))
        with open(protein_dir, "rb") as f:
            protein_identifer = pickle.load(f)

    return protein_identifer


def _get_string_sequence(species):
    links_dir = join(get_data_path(), "PPI_Datasets", "preprocess", species,
                     "{}_sequence.tsv".format(species))
    df = pd.read_csv(links_dir, delimiter='\t', header=None)
    df = df.rename(columns={0: "id", 1: "seq"})
    string_seq_map = pd.Series(df.seq.values, index=df.id).to_dict()

    return string_seq_map


def _get_pdb_sequence(pdb_list):
    species_pdb_seq = {}

    pdb_seq_dir = join(get_data_path(), "PPI_Datasets",
                       "protein_identifier", "pdb_seq_processed.pickle")
    with open(pdb_seq_dir, "rb") as f:
        pdb_seq_map = pickle.load(f)

    for pid in pdb_list:
        species_pdb_seq[pid] = pdb_seq_map[pid]

    return species_pdb_seq


def myfunc(x):
    return euclidean([x[0], x[1], x[2]], [x[3], x[4], x[5]])


def _copy_gexf_for_uploading(total_pdb_list, species):
    from utils import exec_cmd, create_dir_if_not_exists
    dest_folder = join(get_data_path(), 'PPI_Datasets', 'preprocess', species, 'gexf')
    create_dir_if_not_exists(dest_folder)
    print(f'Copying {len(total_pdb_list)} gexf files to {dest_folder}')
    for pdb in tqdm(sorted(total_pdb_list)):
        # try:
        aa_file = pdb[0:4] + "_" + pdb[4] + "_aa.gexf"
        ss_file = pdb[0:4] + "_" + pdb[4] + "_ss.gexf"
        aa_file_path = join(get_pdb_data_path(), aa_file)
        ss_file_path = join(get_pdb_data_path(), ss_file)
        cmd = f'cp {aa_file_path} {dest_folder}'
        exec_cmd(cmd)
        cmd = f'cp {ss_file_path} {dest_folder}'
        exec_cmd(cmd)
        # except FileNotFoundError:
        #     error_count += 1
        #     print("pdb_id chain info: {} may not be correct".format(pdb))
        #     pass


def _construct_pdb_graphs(pdb, glabel):
    # TODO read gexf file and return Regular graph

    aa_file = pdb[0:4] + "_" + pdb[4] + "_aa.gexf"
    ss_file = pdb[0:4] + "_" + pdb[4] + "_ss.gexf"
    aa_file_path = join(get_pdb_data_path(), aa_file)
    ss_file_path = join(get_pdb_data_path(), ss_file)

    #     print(aa_file_path)
    #     print(ss_file_path)
    g_aa = nx.read_gexf(aa_file_path, node_type=int)
    ss_bond = list(g_aa.edges())
    peptide_list = [(i, i + 1) for i in range(len(g_aa.nodes) - 1)]
    # g_aa.add_edges_from(peptide_list)
    g_ss = nx.read_gexf(ss_file_path, node_type=int)

    # Add structural edges (pairwise distance < threshold)
    # nodes = list(g_aa.nodes())
    # nodes_data = g_aa.nodes(data=True)
    # #comb = list(combinations(nodes, 2))
    # comb = list(np.ndindex(len(g_aa.nodes), len(g_aa.nodes)))
    # dist_map = defaultdict(dict)
    # struct_edges = []
    # peptide_edges_weight = []
    # ss_bond_edges_weight = []
    # struct_edges_weight = []
    # for s, t in comb:
    #     sx = nodes_data[s]['x']
    #     sy = nodes_data[s]['y']
    #     sz = nodes_data[s]['z']
    #     s_coords = [sx, sy, sz]
    #     #     print(sx, sy, sz)
    #     tx = nodes_data[t]['x']
    #     ty = nodes_data[t]['y']
    #     tz = nodes_data[t]['z']
    #     t_coords = [tx, ty, tz]
    #     #     print(tx, ty, tz)
    #     dist = euclidean(s_coords, t_coords)
    #     # dist_map[s][t] = dist
    #     if dist < 5.0 and (s, t) not in peptide_list and (s, t) not in ss_bond:
    #         # struct_edges.append((s, t))
    #         struct_edges_weight.append((s, t, dist))
    #     if (s, t) in peptide_list:
    #         peptide_edges_weight.append((s, t, dist))
    #     if (s, t) in ss_bond:
    #         ss_bond_edges_weight.append((s, t, dist))
    #
    # g_aa.add_weighted_edges_from(struct_edges_weight, weight='dist')
    # g_aa.add_weighted_edges_from(peptide_edges_weight, weight='dist')
    # g_aa.add_weighted_edges_from(ss_bond_edges_weight, weight='dist')

    nodes = list(g_aa.nodes())
    nodes_data = g_aa.nodes(data=True)
    num_nodes = len(g_aa.nodes)
    # comb = list(combinations(nodes, 2))
    comb = list(np.ndindex(num_nodes, num_nodes))

    edges_info = np.array(
        [[nodes_data[s]['x'], nodes_data[s]['y'], nodes_data[s]['z'],
          nodes_data[t]['x'], nodes_data[t]['y'], nodes_data[t]['z']] for s, t
         in comb if t > s])
    edges_index = [(s, t) for s, t in comb if t > s]

    # print(comb)
    # print(len(edges_index))

    vfunc = np.apply_along_axis(myfunc, 1, edges_info)
    edge_dist_map = dict(zip(edges_index, vfunc))

    struct_edges_weight = [(s, t, dist) for (s, t), dist in edge_dist_map.items() if
                           dist < 5.0 and (s, t) not in peptide_list and (
                               s, t) not in ss_bond]
    peptide_edges_weight = [(s, t, dist) for (s, t), dist in edge_dist_map.items() if
                            (s, t) in peptide_list]
    ss_bond_edges_weight = [(s, t, dist) for (s, t), dist in edge_dist_map.items() if
                            (s, t) in ss_bond]
    g_aa.add_weighted_edges_from(struct_edges_weight, weight='dist')
    g_aa.add_weighted_edges_from(peptide_edges_weight, weight='dist')
    g_aa.add_weighted_edges_from(ss_bond_edges_weight, weight='dist')

    ss_mapping = {}
    # tt = [n for n in g_aa if g_aa.node[n]['label']=='a4']
    for k, v in g_ss.nodes(data=True):
        try:
            if v["type"] == "helix":
                range_string = v["range"].split(",")
                start = range_string[0]
                end = range_string[1]
                ss_type = v["type"][0] + range_string[2]
                # print(ss_type)
                # ss_type = range_string[2]
                # print(range_string)
                expand_list = [start[0] + str(i) for i in
                               range(int(start[1:]), int(end[1:]) + 1)]

                for seq in expand_list:
                    ss_mapping[seq] = ss_type
            elif v["type"] == "sheet":
                range_string = v["range"].split(",")
                i = iter(range_string)
                range_group = list(zip_longest(i, i, i))
                for sheet in range_group:
                    start = sheet[0]
                    end = sheet[1]
                    if sheet[2] == '':
                        sheet_check = '-1'
                    else:
                        sheet_check = sheet[2]
                    ss_type = v["type"][0] + str(int(sheet_check) + 1)
                    expand_list = [start[0] + str(i) for i in
                                   range(int(start[1:]), int(end[1:]) + 1)]
                    # print(ss_type)
                    for seq in expand_list:
                        # print(seq, ss_type)
                        ss_mapping[seq] = ss_type
        except Exception as e:
            print("File is {}".format(ss_file_path))
            print("range strings = {}".format(range_group))
    # print(ss_mapping)
    node_ss_mapping = {}
    # print(g_aa.nodes(data=True))
    # print(pdb)
    for a in g_aa.nodes(data=True):
        # print(a)
        if a[1]["label"] in ss_mapping.keys():
            node_ss_mapping[a[0]] = {"ss_type": ss_mapping[a[1]["label"]]}
        else:
            node_ss_mapping[a[0]] = {"ss_type": "none"}
    #             print(a[0])
    # print(aa_list)
    nx.set_node_attributes(g_aa, node_ss_mapping)

    for node in g_aa.nodes(data=True):
        del node[1]["label"]

    g_aa.graph["gid"] = pdb
    # g_aa.graph["glabel"] = glabel
    return PDBGraph(g_aa)


def graph_file_exists(pdb):
    filename = pdb[0:4] + "_" + pdb[4] + "_aa.gexf"
    filepath = join(get_pdb_data_path(), filename)
    return path.exists(filepath)


def get_pdb_data_path():
    return None


def _iterate_get_graphs_pairs(protein_mapping, glabel, species=None):
    # May not have to be different graph object, need to check
    # graphs: {"pdb_id_chain": RegularGraph(), "pdb_id_chain": RegularGraph(), ...}
    # graph_pairs: {('string_id1', 'string_id2'): GraphPair( ds_true=1 )}
    graphs = []
    protein_mapping_temp = defaultdict(list)
    protein_mapping_filtered = {}
    graph_pairs = {}
    # protein_dir = join(get_data_path(), "PPI_Datasets", "protein_identifier", "protein_identifier.pickle")
    # with open(protein_dir, "rb") as f:
    #     protein_identifer = pickle.load(f)

    i = 0
    links_file = join(get_data_path(), "PPI_Datasets", "preprocess",
                      species, "{}_links.tsv".format(species))

    df = pd.read_csv(links_file, delimiter=' ', header=0)
    # source = df['protein1'].values.tolist()
    # target = df['protein2'].values.tolist()
    # pairs = list(zip(source, target))
    df = df.rename(columns={"neighborhood": "nb", "fusion": "fu",
                            "cooccurence": "co", "coexpression": "ce",
                            "experimental": "ex", "database": "da",
                            "textmining": "tx"})
    df2 = df.drop(['protein1', 'protein2', 'combined_score'], axis=1)
    cols = df2.columns
    bt = df2.apply(lambda x: x > 0, raw=True)
    positive_list = bt.apply(lambda x: list(cols[x.values]), axis=1).tolist()
    pairs_raw = pd.Series(df2.to_dict('records'),
                          index=[df.protein1, df.protein2]).to_dict()
    del df, df2

    pairs = {}
    for ((s, t), info), positives in zip(pairs_raw.items(), positive_list):
        positive_dict = {}
        for positive in positives:
            positive_dict[positive] = info[positive]
        pairs[(s, t)] = positive_dict
    del pairs_raw

    pairs = filter_duplicate_edges(pairs)

    gc.collect()

    total_pdb_list = set()
    total_string_list = set()
    edge_list_excluded = set()
    edge_list_remained = set()

    t = time()

    # i = 0# TODO

    for (source, target), edge_info in tqdm(pairs.items()):
        # print(source, target)

        # i += 1# TODO
        #
        # if i == 10:# TODO
        #     break # TODO: remove later

        try:
            if source in protein_mapping:
                source_pdb_list = protein_mapping[source]["pdb"]
            else:
                source_pdb_list = []
            if target in protein_mapping:
                target_pdb_list = protein_mapping[target]["pdb"]
            else:
                target_pdb_list = []
            # source_pdb_list = protein_mapping[source]["pdb"]
            # target_pdb_list = protein_mapping[target]["pdb"]
            if len(source_pdb_list) == 0 or len(target_pdb_list) == 0:
                edge_list_excluded.add((source, target))
                continue
            filtered_source_pdb = [s_pdb for s_pdb in
                                   source_pdb_list if
                                   graph_file_exists(s_pdb["pdb_id"])]
            filtered_target_pdb = [t_pdb for t_pdb in
                                   target_pdb_list if
                                   graph_file_exists(t_pdb["pdb_id"])]
            if len(filtered_source_pdb) == 0 or len(
                    filtered_target_pdb) == 0:
                edge_list_excluded.add((source, target))
                continue
            else:
                edge_list_remained.add((source, target))
                total_string_list.add(source)
                total_string_list.add(target)
                # for pdb in filtered_source_pdb:
                protein_mapping_temp[source] = filtered_source_pdb
                # for pdb in filtered_target_pdb:
                protein_mapping_temp[target] = filtered_target_pdb
                graph_pairs[(source, target)] = \
                    GraphPair(ds_true=1, edge_types=edge_info)
            for pdb in filtered_source_pdb:  # + filtered_target_pdb
                total_pdb_list.add(pdb["pdb_id"])
            for pdb in filtered_target_pdb:
                total_pdb_list.add(pdb["pdb_id"])

        except KeyError:
            # print("source_id = {}".format(source))
            # print("target_id = {}".format(target))
            print("KeyError: {} {}".format(source, traceback.format_exc()))
            continue

    print('### len(total_pdb_list)', len(total_pdb_list))

    for sid, pdb_list in protein_mapping_temp.items():
        protein_mapping_filtered[sid] = {"pdb": pdb_list}

    graphs = []
    print("")
    print("Checking PDB complete graphs...")
    # filepath = join(get_data_path(), "PPI_Datasets", "preprocess", species, "pdb_graphs_complete")
    # graphs_dict = load(filepath) # TODO
    graphs_dict = None  # TODO
    if graphs_dict:
        # if False:
        graphs = graphs_dict['graphs']
    else:
        count = 0
        error_count = 0
        print("Constructing pdb graphs...")
        print("Checking PDB temp graphs...")
        filepath = join(get_data_path(), "PPI_Datasets", "preprocess", species,
                        "pdb_graphs_temp")
        # graphs_dict = load(filepath)
        graphs_dict = None  # TODO
        pdb_temp_num = 0
        # print('graphs_dict', graphs_dict)
        if graphs_dict:
            graphs = graphs_dict['graphs']
            pdb_temp_num = graphs_dict["count"]
            error_count = graphs_dict['error']

        _copy_gexf_for_uploading(total_pdb_list, species) # TODO
        exit() # TODO


        for pdb in tqdm(sorted(total_pdb_list)):
            try:
                if count < pdb_temp_num:
                    count += 1
                    continue
                print(pdb)
                g_temp = _construct_pdb_graphs(pdb, glabel)
                graphs.append(g_temp)
                if count % 200 == 1:
                    obj = {"graphs": graphs, "count": count,
                           "error": error_count}
                    filepath = join(get_data_path(), "PPI_Datasets",
                                    "preprocess", species, "pdb_graphs_temp")
                    save(obj, filepath, print_msg=True, use_klepto=True)
                count += 1
            except FileNotFoundError:
                error_count += 1
                print("pdb_id chain info: {} may not be correct".format(pdb))
                pass

        obj = {"graphs": graphs, "count": count, "error": error_count}
        filepath = join(get_data_path(), "PPI_Datasets", "preprocess",
                        species, "pdb_graphs_complete")

        save(obj, filepath, print_msg=True, use_klepto=True)
    print("Took {} mins processing".format((time() - t) / 60))
    filepath = join(get_data_path(), "PPI_Datasets", "preprocess",
                    species, "pdb_graphs_temp.klepto")
    try:
        remove(filepath)
    except FileNotFoundError:
        print("File already removed")
        pass

    pdb_to_idx_map = dict(
        zip(sorted(total_pdb_list), range(0, len(total_pdb_list))))

    # graphs = [_construct_pdb_graphs(pdb) for pdb in sorted(total_pdb_list)]
    # print(graphs)
    # TODO check connectivity
    print(len(graphs))
    print(len(graph_pairs))
    print(len(total_string_list))

    return graphs, graph_pairs, edge_list_excluded, edge_list_remained, \
           total_string_list, pdb_to_idx_map, protein_mapping_filtered
    # TODO: iterate the folder join(get_data_path(), 'PPI_Datasets', 'PDB') using glob


def get_ppi_sparse_node_feat(edge_list_aggregated, total_string_list):
    # for edge_type, edge_list in edge_list_aggregated.items():
    # n = len(a2idx)
    n = len(total_string_list)
    assoc = np.zeros((n, n))
    try:
        for source, target_list in edge_list_aggregated["positive"].items():
            for target in target_list:
                assoc[source, target] = assoc[target, source] = 1.
    except IndexError:
        print("s={}, t={}".format(source, target))
    assoc = sp.csr_matrix(assoc)
    return assoc


def get_edge_from_pairs(edge_list):
    edge_list_temp = defaultdict(lambda: defaultdict(set))
    for gid1, gid2 in edge_list:
        edge_list_temp["positive"][gid1].add(gid2)

    edge_list_aggregated = {
        "positive": {k: list(v) for k, v in edge_list_temp["positive"].items()}}
    return edge_list_aggregated


def _check_list_element_allowed(li1, li2):
    assert type(li1) is list and type(li2) is list
    for ele1 in li1:
        if 'ppi_snap_pdb' in ele1:
            assert 'ppi_snap_pdb' in li2
        # assert ele1 in li2
    return li2


if __name__ == '__main__':
    protein_pdb_map = _get_string_pdb_mapping()
    x, y, z, r = _iterate_get_graphs_pairs(protein_pdb_map, "394")
