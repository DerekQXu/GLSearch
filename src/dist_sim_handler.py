from utils import get_root_path, exec_cmd, get_ts, create_dir_if_not_exists, sorted_nicely
# from nx_to_gxl import nx_to_gxl
from os.path import isfile, join, basename
from os import getpid
from time import time
from nx_to_mivia import convert_to_mivia, get_current_label
import fileinput
import json
import networkx as nx
from glob import glob

import sys

# For mcsp python.
sys.path.insert(0, join(get_root_path(), 'model', 'OurMCS', 'mcsp'))
from mcsp_orig import mcsp_python_interface


# # For HED.
# sys.path.insert(0, '{}/model/aproximated_ged/aproximated_ged'.format(get_root_path()))


def ged(g1, g2, algo, debug=False, timeit=False, timeout=None):
    if algo in ['astar', 'hungarian', 'vj'] or 'beam' in algo:
        return handle_ged_gmt(g1, g2, algo, debug, timeit, timeout)
    elif algo in ['f2', 'f2lp', 'f24threads']:
        return handle_ged_fs(g1, g2, algo, debug, timeit)
    elif algo == 'hed':
        return handle_ged_hed(g1, g2, algo, debug, timeit)
    else:
        raise RuntimeError('Unknown ged algo {}'.format(algo))


def normalized_dist_sim(d, g1, g2, dec_gsize=False):
    g1_size = g1.number_of_nodes()
    g2_size = g2.number_of_nodes()
    if dec_gsize:
        g1_size -= 1
        g2_size -= 1
    import numpy as np
    # if type(d) is np.int64:
    #     print('@@@', d, g1_size + g2_size, 2 * d / (g1_size + g2_size))
    # else:
    #     print('####', d, g1_size + g2_size, 2 * d / (g1_size + g2_size))
    return 2 * d / (g1_size + g2_size)


def unnormalized_dist_sim(d, g1, g2, dec_gsize=False):
    g1_size = g1.number_of_nodes()
    g2_size = g2.number_of_nodes()
    if dec_gsize:
        g1_size -= 1
        g2_size -= 1
    return d * (g1_size + g2_size) / 2


def handle_ged_gmt(g1, g2, algo, debug, timeit, timeout):
    # https://github.com/dan-zam/graph-matching-toolkit
    gp = get_gmt_path()
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str)
    meta1 = write_to_temp(g1, t_datapath, algo, 'g1')
    meta2 = write_to_temp(g2, t_datapath, algo, 'g2')
    if meta1 != meta2:
        if not ((meta1 in meta2) or (meta2 in meta1)):
            raise RuntimeError(
                'Different meta data {} vs {}'.format(meta1, meta2))
        else:
            if meta1 in meta2:
                meta1 = meta2
    prop_file = setup_property_file(src, gp, meta1, append_str)
    rtn = []
    lcnt, result_file, t = None, None, None
    if not exec_cmd(
            'cd {} && java {}'
            ' -classpath {}/src/graph-matching-toolkit/bin algorithms.GraphMatching '
            './properties/properties_temp_{}.prop'.format(
                gp, '-XX:-UseGCOverheadLimit -XX:+UseConcMarkSweepGC -Xmx100g'
                if algo == 'astar' else '', get_root_path(), append_str), timeout):
        rtn.append(-1)
        lcnt = -1
        t = timeout
    else:
        d, t, lcnt, g1size, g2size, result_file = get_gmt_result(gp, algo, append_str)
        rtn.append(d)
        if g1size != g1.number_of_nodes():
            print('g1size {} g1.number_of_nodes() {}'.format(g1size, g1.number_of_nodes()))
        assert (g1size == g1.number_of_nodes())
        assert (g2size == g2.number_of_nodes())
    if debug:
        rtn += [lcnt, g1, g2]
    if timeit:
        rtn.append(t)
    clean_up([t_datapath, prop_file, result_file])
    if len(rtn) == 1:
        return rtn[0]
    return tuple(rtn)


def setup_temp_data_folder(gp, append_str, fp_prepend_info=''):
    dir = gp + '/data'
    create_dir_if_not_exists(dir)
    if fp_prepend_info != '':
        append_str = fp_prepend_info.replace(';', '_')
    tp = dir + '/temp_{}'.format(append_str)
    exec_cmd('rm -rf {} && mkdir {}'.format(tp, tp))
    src = get_root_path() + '/src/gmt_files'
    exec_cmd('cp {}/temp.xml {}/temp_{}.xml'.format(src, tp, append_str))
    return src, tp


def write_to_temp(g, tp, algo, g_name):
    node_attres, edge_attrs = nx_to_gxl(
        g, g_name, '{}/{}.gxl'.format(tp, g_name),
        ignore_node_attrs=['label'],
        ignore_edge_attrs=['id', 'weight'])  # cannot handle weighted edges
    return algo + '_' + '_'.join(sorted(list(node_attres.keys())) + \
                                 sorted(list(edge_attrs.keys())))


def setup_property_file(src, gp, meta, append_str):
    destfile = '{}/properties/properties_temp_{}.prop'.format(
        gp, append_str)
    srcfile = '{}/{}.prop'.format(src, meta)
    if not isfile(srcfile):
        if 'beam' in meta:  # for beam
            metasp = meta.split('_')
            s = int(metasp[0][4:])
            if s <= 0:
                raise RuntimeError('Invalid s for beam search: {}'.format(s))
            newmeta = '_'.join(['beam'] + metasp[1:])
            srcfile = '{}/{}.prop'.format(src, newmeta)
        else:
            raise RuntimeError('File {} does not exist'.format(srcfile))
    exec_cmd('cp {} {}'.format(srcfile, destfile))
    for line in fileinput.input(destfile, inplace=True):
        line = line.rstrip()
        if line == 's=':  # for beam
            print('s={}'.format(s))
        else:
            print(line.replace('temp', 'temp_{}'.format(append_str)))
    return destfile


def get_gmt_result(gp, algo, append_str):
    result_file = '{}/result/temp_{}'.format(gp, append_str)
    with open(result_file) as f:
        lines = f.readlines()
        ln = 16 if 'beam' in algo else 15
        t = int(lines[ln].split(': ')[1])  # msec
        ln = 23 if 'beam' in algo else 22
        d = float(lines[ln]) * 2
        if (d - int(d) != 0) and algo != 'hungarian' and algo != 'vj':
            raise RuntimeError('{} != {}'.format(d, int(d)))
        d = int(d)
        if d < 0:
            d = -1  # in case rtn == -2
        ln = 26 if 'beam' in algo else 25
        g1size = int(lines[ln])
        ln = 27 if 'beam' in algo else 26
        g2size = int(lines[ln])
        ln = 28 if 'beam' in algo else 27
        lcnt = int(float(lines[ln]))
        return d, t, lcnt, g1size, g2size, result_file


def get_gmt_path():
    return get_root_path() + '/src/graph-matching-toolkit'


def get_append_str(g1, g2):
    return '{}_{}_{}_{}'.format(
        get_ts(), getpid(), g1.graph['gid'], g2.graph['gid'])


def clean_up(path_list):
    # for path in path_list:
    #     exec_cmd('rm -rf {}'.format(path))
    pass

def handle_ged_fs(g1, g2, algo, debug, timeit):
    # https://drive.google.com/file/d/12MBjXcNko83mAUGKe9nVJqEKjLTjDJNd/view?usp=sharing
    gp = get_fs_path(algo)
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str)
    meta1 = write_to_temp(g1, t_datapath, algo, 'g1')
    meta2 = write_to_temp(g2, t_datapath, algo, 'g2')
    if meta1 != meta2:
        if not ((meta1 in meta2) or (meta2 in meta1)):
            raise RuntimeError(
                'Different meta data {} vs {}'.format(meta1, meta2))
    rtn = []
    result_file = t_datapath + '/reault.txt'
    t = None
    if not exec_cmd(
            'cd {}/symbolic && '
            'DISPLAY=:0 wine {}_symbolic_distance.exe '
            '1.0 1.0 0.0 1.0 {}/g1.gxl {}/g2.gxl | tee {}'.format(
                gp, algo.upper(), t_datapath, t_datapath, result_file)):
        rtn.append(-1)
    else:
        with open(result_file) as f:
            lines = f.readlines()
            assert (len(lines) == 1)
            line = lines[0]
            ls = line.rstrip().split(';')
            assert (len(ls) == 4)
            t = float(ls[1]) * 1000  # sec to msec
            d = int(float(ls[3]))
        rtn.append(d)
    if debug:
        rtn += [-1, g1, g2]
    if timeit:
        rtn.append(t)
    clean_up([t_datapath, result_file])
    if len(rtn) == 1:
        return rtn[0]
    return tuple(rtn)


def get_fs_path(algo):
    return get_root_path() + '/model/' + algo.upper()


def handle_ged_hed(g1, g2, algo, debug, timeit):
    # https://github.com/priba/aproximated_ged
    from VanillaHED import VanillaHED
    rtn = []
    assert (algo == 'hed')
    hed = VanillaHED(del_node=1.0, ins_node=1.0, del_edge=1.0, ins_edge=1.0,
                     node_metric='matching', edge_metric='matching')
    g1 = map_node_type_to_float(g1)
    g2 = map_node_type_to_float(g2)
    t = time()
    d, _ = hed.ged(g1, g2)
    rtn.append(d)
    if debug:
        rtn += [-1, g1, g2]
    if timeit:
        rtn.append((time() - t) * 1000)
    if len(rtn) == 1:
        return rtn[0]
    return tuple(rtn)


def map_node_type_to_float(g):
    for n, attr in g.nodes(data=True):
        if 'type' in attr:
            s = ''
            for c in attr['type']:
                num = ord(c)
                s = str(num)
            num = int(s)
            attr['hed_mapped'] = num
        else:
            attr['hed_mapped'] = 0
    for n1, n2, attr in g.edges(data=True):
        attr['hed_mapped'] = 0
    return g


""" MCS. """

"""
DEFAULT(0, "Default SMSD algorithm"),
MCSPlus(1, "MCS Plus algorithm"),
VFLibMCS(2, "VF Lib based MCS algorithm"),
CDKMCS(3, "CDK UIT MCS"),
SubStructure(4, "Substructure search"),
TurboSubStructure(5, "Turbo Mode- Substructure search"),
vfLibGAdMCS(6, "VF Lib based algorithm with genetic algorithm to return disconnected MCS"),
consR_dMCES(7, "Spectral Anchor-expansion-refinement algorithm, quick method to return a large dCES (but not necessarily the dMCES)"),
ChemAxon_cMCES(8, "ChemAxon's clique-detection method on modular product of 2 graphs.  Finds the cMCES"),
ChemAxon_dMCES(9, "ChemAxon's clique-detection method on modular product of 2 graphs.  Finds the dMCES"),
BK_dMCES(10, "Bron-Kerbosch maximal clique detection Algorithm.  Finds all possible dMCESs"),
CP_dMCES(11, "Carraghan-Pardalos maximum clique detection Algorithm.  Finds the dMCES"),
RASCAL_dMCES(12, "RASCAL maximum clique detection Algorithm.  Finds the dMCES"),
Depolli_dMCES(13, "Depolli et al. maximum clique detection Algorithm.  Finds the dMCES"),
fMCS(14, "Andrew Dalke's common subgraph enumeration algorithm.  Finds the cMCES"),
kCombu_dMCES(15, "kCombu build-up Algorithm.  Finds the dMCES"),
kCombu_cMCES(16, "kCombu build-up Algorithm.  Finds the cMCES"),
BK_cMCES(17, "Bron-Kerbosch maximal clique detection Algorithm.  Finds all possible cMCESs")

mccreesh2017
k_down
mcsp+rl
"""


def mcs(g1, g2, algo, labeled=False, label_key='',
        recursion_threshold=0, save_every_seconds=-1,
        save_every_iter=False, timeit=False,
        timeout=None,
        debug=False,
        computer_name='',
        fp_prepend_info=''):
    """
    :param g1:
    :param g2:
    :param algo:
    :param debug:
    :param timeit:
    :param timeout: Timeout in seconds that the MCS run is allowed to take.
    :return: The actual size of MCS (# of edges) (integer),
             the node mapping (e.g. [{('1', '2'): ('A', 'B'), ('7', '9'): ('C', 'D')}, ...]),
             the edge_id mapping (e.g. [{'1': '2', '4': '3'}, ...]
             wall time in msec (if timeit==True).
    """
    if algo == 'mccreesh2016' or algo == 'mccreesh2017' or algo == 'mcsp+rl':
        return mcs_cpp_helper(g1, g2, algo, labeled, label_key, recursion_threshold,
                              save_every_seconds, save_every_iter,
                              debug, timeit, timeout,
                              computer_name, fp_prepend_info)
    elif algo == 'mcsp_py':
        return mcs_mcsp_python_helper(g1, g2, algo, labeled, label_key, recursion_threshold,
                                      save_every_seconds,
                                      debug, timeit, timeout)
    ###############################################################
    elif algo == "k_down":
        return mcs_kdown_helper(g1, g2, algo, labeled, label_key, debug, timeit, timeout,
                                computer_name, fp_prepend_info)
    ###############################################################
    else:
        return mcs_java_helper(g1, g2, algo, debug, timeit, timeout)


def mcs_java_helper(g1, g2, algo, debug=False, timeit=False, timeout=None):
    """See mcs function. Must match return format."""
    # Input format is ./model/mcs/data/temp_<ts>_<pid>_<gid1>_<gid2>/<gid1>_<gid2>_<algo>.json
    # Prepare the json file for java to read.
    gp = get_mcs_path()
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str)
    filepath = '{base}/{g1}_{g2}_{algo}.json'.format(
        base=t_datapath,
        g1=g1.graph['gid'],
        g2=g2.graph['gid'],
        algo=algo)
    write_java_input_file(g1, g2, algo, filepath)

    # Run the java program.
    java_src = get_mcs_path() + '/Java_MCS_algorithms/bin/'
    classpath = '{}:{}'.format(get_mcs_path() + '/Java_MCS_algorithms/bin/',
                               get_mcs_path() + '/Java_MCS_algorithms/lib/*')
    main_class = 'org.cisrg.mcsrun.RunMcs'
    exec_result = exec_cmd('java -Xmx20g -classpath "{classpath}" {main_class} {data_file}'.format(
        root=java_src, classpath=classpath, main_class=main_class, data_file=filepath),
        timeout=timeout)

    # Get out immediately with a -1 so the csv file logs failed test.
    # mcs_size = -1 means failed in the time limit.
    # mcs_size = -2 means failed by memory limit or other error.
    if not exec_result:
        return -1, -1, -1, timeout * 1000

    # Check if the output file exists, otherwise java threw an exception and didn't output anything.
    output_filepath = filepath + '.out'
    if not isfile(output_filepath):
        return -2, -1, -1, 0

    # Process the output data from the java program, original filename + .out (*.json.out).
    with open(output_filepath, 'r') as jsonfile:
        output_data = json.load(jsonfile)

    # Get the relevant data.
    edge_mappings = output_data['mcsEdgeMapIds']
    elapsed_time = output_data['elapsedTime']
    if len(edge_mappings) == 0:
        mcs_size = 0
    else:
        mcs_size = len(edge_mappings[0])

    mcs_node_id_maps, mcs_node_label_maps = get_mcs_info(g1, g2, edge_mappings)

    clean_up([t_datapath])

    if timeit:
        return mcs_size, mcs_node_label_maps, edge_mappings, elapsed_time
    else:
        return mcs_size, mcs_node_label_maps, edge_mappings


def mcs_cpp_helper(g1, g2, algo, labeled, label_key, recursion_threshold,
                   save_every_seconds, save_every_iter,
                   debug=False, timeit=False, timeout=None,
                   computer_name='', fp_prepend_info=''):
    """See mcs function. Must match return format."""
    # Input format is ./model/mcs/data/temp_<ts>_<pid>_<gid1>_<gid2>/<gid1>.<extension>
    # Prepare both graphs to be read by the program.
    commands = []
    if algo == 'mccreesh2016':
        binary_name = 'solve_max_common_subgraph'
        extension = 'mivia'
        write_fn = write_mivia_input_file
        commands.append('' if labeled else '--unlabelled')
        commands.append('--connected')
        commands.append('--undirected')
    elif algo == 'mccreesh2017' or algo == 'mcsp+rl':
        # print('computer_name', computer_name)
        # binary_name = 'mcsp'  # 'mcsp_scai1'
        # if 'scai1' in computer_name:
        #     binary_name = 'mcsp_scai1'
        if algo == 'mccreesh2017':
            binary_name = 'code/james-cpp-periodic-save/mcsp'
        elif algo == 'mcsp+rl':
            binary_name = 'mcsp+rl'
        else:
            assert False
        extension = 'mivia'
        write_fn = write_mivia_input_file
        commands.append('--labelled' if labeled else '')
        commands.append('--connected')
        commands.append('--quiet')
        if timeout:
            commands.append('--timeout={}'.format(timeout))
        commands.append('min_product')
    else:
        raise RuntimeError('{} not yet implemented in mcs_cpp_helper'.format(algo))

    gp = get_mcs_path()
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str, fp_prepend_info)
    filepath_g1 = '{base}/{g1}.{extension}'.format(
        base=t_datapath,
        g1=g1.graph['gid'],
        extension=extension)
    filepath_g2 = '{base}/{g2}.{extension}'.format(
        base=t_datapath,
        g2=g2.graph['gid'],
        extension=extension)

    if labeled:
        label_map = _get_label_map(g1, g2, label_key)
    else:
        label_map = {}
    idx_to_node_1 = write_fn(g1, filepath_g1, labeled, label_key, label_map)
    idx_to_node_2 = write_fn(g2, filepath_g2, labeled, label_key, label_map)

    cpp_binary = '{mcs_path}/{algo}/{binary}'.format(
        mcs_path=get_mcs_path(), algo=algo,
        binary=binary_name)

    # Run the solver.
    t = time()
    if algo in ['mccreesh2017', 'mcsp+rl']:
        commands.append('--recursion_threshold=' + str(recursion_threshold))
        commands.append('--save_every_seconds=' + str(save_every_seconds))
        if algo == 'mcsp+rl' and save_every_iter:
            commands.append('--save_every_iter')

    # runthis = '{bin} {commands} {g1} {g2}'.format(
    #     bin=cpp_binary, commands=' '.join(commands),
    #     g1=filepath_g1, g2=filepath_g2)
    exec_result = exec_cmd('{bin} {commands} {g1} {g2}'.format(
        bin=cpp_binary, commands=' '.join(commands),
        g1=filepath_g1, g2=filepath_g2), timeout)
    elapsed_time = time() - t
    elapsed_time *= 1000  # sec to msec

    # # Get out immediately with a -1 so the csv file logs failed test.
    # # mcs_size = -1 means failed in the time limit.
    # # mcs_size = -2 means failed by memory limit or other error.
    # if not exec_result:
    #     return -1, -1, -1, timeout * 1000

    # # Check if the output file exists, otherwise something failed with no output.
    output_filepath = join(t_datapath, 'output.csv')
    # if not isfile(output_filepath):
    #     return -2, -1, -1, 0

    # Process the output data.

    all_csv_files = sorted_nicely(glob(join(t_datapath, 'output_*.csv')))
    if isfile(output_filepath):
        all_csv_files += [output_filepath]
    mcs_size_list = []
    idx_mapping_list = []
    refined_mcs_node_label_maps_list = []
    refined_edge_mapping_list = []
    time_list = []

    for f in all_csv_files:
        with open(f, 'r') as readfile:
            x = None
            try:
                x = readfile.readline().strip()
                num_nodes_mcis = int(x)
            except:
                print(f)
                print(x)
                exit(-1)
            idx_mapping = eval(readfile.readline().strip())
            mcs_node_id_mapping = {idx_to_node_1[idx1]: idx_to_node_2[idx2] for idx1, idx2 in
                                   idx_mapping.items()}
            # elapsed_time = int(readfile.readline().strip())

        idx_mapping_list.append(idx_mapping)

        # Sanity Check 1: connectedneses
        indices_left = idx_mapping.keys()
        indices_right = idx_mapping.values()
        subgraph_left = g1.subgraph(indices_left)
        subgraph_right = g2.subgraph(indices_right)
        is_connected_left = nx.is_empty(subgraph_left) or nx.is_connected(subgraph_left)
        is_connected_right = nx.is_empty(subgraph_right) or nx.is_connected(subgraph_right)
        assert is_connected_left and is_connected_right, \
            'Unconnected result for pair ={}\n{}'.format(f, idx_mapping)
        # # Sanity Check 2: isomorphism (NOTE: labels not considered!)
        # # import networkx.algorithms.isomorphism as iso
        # # natts = ['type']
        # # nm = iso.categorical_node_match(natts, [''] * len(natts))
        # assert nx.is_isomorphic(subgraph_left, subgraph_right)#, node_match=nm)

        refined_edge_mapping = mcis_edge_map_from_nodes(g1, g2, mcs_node_id_mapping)
        mcs_size_list.append(num_nodes_mcis)
        mcs_node_id_maps, mcs_node_label_maps = get_mcs_info(g1, g2, [refined_edge_mapping])
        refined_mcs_node_label_maps_list.append(mcs_node_label_maps)
        refined_edge_mapping_list.append(refined_edge_mapping)
        bfn = basename(f)
        if 'output_' in bfn:
            # print('Output', bfn)
            if algo == 'mcsp+rl':
                time_list.append(int(bfn.split('output_')[1].split('_')[1].split('.csv')[0])) # iter
            else:
                time_list.append(float(bfn.split('output_')[1].split('.csv')[0]))

    clean_up([t_datapath])

    if not debug:
        return mcs_size_list

    if timeit:
        return mcs_size_list, idx_mapping_list, refined_mcs_node_label_maps_list, \
               refined_edge_mapping_list, time_list + [elapsed_time]
    else:
        return mcs_size_list, idx_mapping_list, refined_mcs_node_label_maps_list, \
               refined_edge_mapping_list



# def _parse_(f, idx_to_node_1, idx_to_node_2):
#     with open(f, 'r') as readfile:
#         x = None
#         try:
#             x = readfile.readline().strip()
#             num_nodes_mcis = int(x)
#         except:
#             print(f)
#             print(x)
#             exit(-1)
#         idx_mapping = eval(readfile.readline().strip())
#         mcs_node_id_mapping = {idx_to_node_1[idx1]: idx_to_node_2[idx2] for idx1, idx2 in
#                                idx_mapping.items()}
#         # elapsed_time = int(readfile.readline().strip())
#
#     idx_mapping_list.append(idx_mapping)
#
#     # Sanity Check 1: connectedneses
#     indices_left = idx_mapping.keys()
#     indices_right = idx_mapping.values()
#     subgraph_left = g1.subgraph(indices_left)
#     subgraph_right = g2.subgraph(indices_right)
#     is_connected_left = nx.is_empty(subgraph_left) or nx.is_connected(subgraph_left)
#     is_connected_right = nx.is_empty(subgraph_right) or nx.is_connected(subgraph_right)
#     assert is_connected_left and is_connected_right, \
#         'Unconnected result for pair ={}\n{}'.format(f, idx_mapping)
#     # # Sanity Check 2: isomorphism (NOTE: labels not considered!)
#     # # import networkx.algorithms.isomorphism as iso
#     # # natts = ['type']
#     # # nm = iso.categorical_node_match(natts, [''] * len(natts))
#     # assert nx.is_isomorphic(subgraph_left, subgraph_right)#, node_match=nm)
#
#     refined_edge_mapping = mcis_edge_map_from_nodes(g1, g2, mcs_node_id_mapping)
#     mcs_size_list.append(num_nodes_mcis)
#     mcs_node_id_maps, mcs_node_label_maps = get_mcs_info(g1, g2, [refined_edge_mapping])
#     refined_mcs_node_label_maps_list.append(mcs_node_label_maps)
#     refined_edge_mapping_list.append(refined_edge_mapping)
#     bfn = basename(f)


def mcs_mcsp_python_helper(g1, g2, algo, labeled, label_key, recursion_threshold,
                           save_every_seconds,
                           debug=False, timeit=False, timeout=None):
    t = time()
    mcs_size, node_mapping = mcsp_python_interface(
        g1, g2, labeled, label_key, recursion_threshold,
        save_every_seconds, timeout)
    elapsed_time = time() - t
    elapsed_time *= 1000  # sec to msec
    if timeit:
        return mcs_size, node_mapping, '', '', elapsed_time
    else:
        return mcs_size, node_mapping, '', ''


def mcs_kdown_helper(g1, g2, algo, labeled, label_key, debug=False, timeit=False, timeout=None,
                     computer_name='', fp_prepend_info=''):
    """See mcs function. Must match return format."""
    # Input format is ./model/mcs/data/temp_<ts>_<pid>_<gid1>_<gid2>/<gid1>.<extension>
    # Prepare both graphs to be read by the program.
    commands = []
    if algo == "k_down":
        binary_name = 'solve_subgraph_isomorphism'
        extension = 'mivia'
        write_fn = write_mivia_input_file
        commands.append('sequentialix')
        if timeout is not None:
            timeout_int = int(timeout)  # the C++ program cannot handle "__.0"
            commands.append('--timeout {}'.format(timeout_int))
        commands.append('--high-wildcards')
        commands.append('--induced')
        commands.append('--format')
        commands.append('vf')
    else:
        raise RuntimeError('{} not yet implemented in mcs_cpp_helper'.format(algo))
    gp = get_mcs_path()
    append_str = get_append_str(g1, g2)
    src, t_datapath = setup_temp_data_folder(gp, append_str, fp_prepend_info)
    filepath_g1 = '{base}/{g1}.{extension}'.format(
        base=t_datapath,
        g1=g1.graph['gid'],
        extension=extension)
    filepath_g2 = '{base}/{g2}.{extension}'.format(
        base=t_datapath,
        g2=g2.graph['gid'],
        extension=extension)
    if labeled:
        label_map = _get_label_map(g1, g2, label_key)
    else:
        label_map = {}
    idx_to_node_1 = write_fn(g1, filepath_g1, labeled, label_key, label_map)
    idx_to_node_2 = write_fn(g2, filepath_g2, labeled, label_key, label_map)

    cpp_binary = '{mcs_path}/{algo}/{binary}'.format(mcs_path=get_mcs_path(), algo=algo,
                                                     binary=binary_name)

    # Run the solver.
    t = time()
    exec_result = exec_cmd('{bin} {commands} {g1} {g2} {out}'.format(
        bin=cpp_binary, commands=' '.join(commands),
        g1=filepath_g1, g2=filepath_g2, out=t_datapath),
        timeout=timeout)

    elapsed_time = time() - t
    elapsed_time *= 1000  # sec to msec

    if not exec_result:
        return -1, -1, -1, timeout * 1000

    # Check if the output file exists, otherwise something failed with no output.
    output_filepath = t_datapath + '/output.csv'
    if not isfile(output_filepath):
        return -2, -1, -1, timeout * 1000

    # Process the output data.
    with open(output_filepath, 'r') as readfile:
        num_nodes_mcis = int(readfile.readline().strip())
        idx_mapping = eval(readfile.readline().strip())
        # mcs_node_id_mapping = {idx_to_node_1[idx1]: idx_to_node_2[idx2] for idx1, idx2 in
        #                       idx_mapping.items()}
        # elapsed_time = float(readfile.readline().strip())

    edge_mapping = {}  # mcis_edge_map_from_nodes(g1, g2, mcs_node_id_mapping)
    mcs_size = num_nodes_mcis
    # mcs_node_id_maps = get_mcs_info_kdown(g1, g2, label_key, idx_mapping)
    # print(idx_mapping)
    # print(mcs_node_id_maps)
    # clean_up([t_datapath])

    if not debug:
        return mcs_size

    if timeit:
        return mcs_size, idx_mapping, [edge_mapping], elapsed_time
    else:
        return mcs_size, idx_mapping, [edge_mapping]


def _get_label_map(g1, g2, label_key):
    # Need this function because the two graphs needs consistent labelings in the mivia format. If they are called
    # separately, then they will likely have wrong labelings.
    label_dict = {}
    label_counter = 0
    # We make the labels into ints so that they can fit in the 16 bytes needed
    # for the labels in the mivia format. Each unique label encountered just gets a
    # unique label from 0 to edge_num - 1
    for g in [g1, g2]:
        for node, attr in g.nodes(data=True):
            current_label = get_current_label(attr,label_key)
            if current_label not in label_dict:
                label_dict[current_label] = label_counter
                label_counter += 1
    return label_dict


def mcis_edge_map_from_nodes(g1, g2, node_mapping):
    edge_map = {}
    induced_g1 = g1.subgraph([key for key in node_mapping.keys()])
    induced_g2 = g2.subgraph([key for key in node_mapping.values()])

    used_edge_ids_g2 = set()
    for u1, v1, edge1_attr in induced_g1.edges(data=True):
        u2 = node_mapping[int(u1)]
        v2 = node_mapping[int(v1)]
        edge1_id = edge1_attr['id']
        found = False
        for temp1, temp2, edge2_attr in induced_g2.edges(nbunch=[u2, v2], data=True):
            if (u2 == temp1 and v2 == temp2) or (u2 == temp2 and v2 == temp1):
                edge2_id = edge2_attr['id']
                if edge2_id in used_edge_ids_g2:
                    continue
                used_edge_ids_g2.add(edge2_id)
                edge_map[edge1_id] = edge2_id
                found = True
        # if not found:
        #     raise ValueError('X')

    return edge_map


def write_mivia_input_file(graph, filepath, labeled, label_key, label_map):
    bytes, idx_to_node = convert_to_mivia(graph, labeled, label_key, label_map)
    with open(filepath, 'wb') as writefile:
        for byte in bytes:
            writefile.write(byte)
    return idx_to_node


def get_mcs_info_cpp(g1, g2, mivia_edge_mapping):
    g1_edge_map = get_mivia_edge_map(g1)
    g2_edge_map = get_mivia_edge_map(g2)

    # Translate the mivia edge map to nx edge id map.
    edge_map = {}
    for mivia_edge_1, mivia_edge_2 in mivia_edge_mapping.items():
        edge_1 = g1_edge_map[mivia_edge_1]
        edge_2 = g2_edge_map[mivia_edge_2]
        edge_map[edge_1] = edge_2

    mcs_node_id_maps, mcs_node_label_maps = get_mcs_info(g1, g2, [edge_map])

    return mcs_node_id_maps, mcs_node_label_maps, [edge_map]


def get_mivia_edge_map(graph):
    edge_map = {}

    # Go through same order as how we create the mivia graph file.
    adj_iter = sorted(graph.adj.items(), key=lambda x: int(x[0]))
    edge_num = 0
    for source_id, adj_list in adj_iter:
        for target_id, attr in sorted(adj_list.items(), key=lambda x: int(x[0])):
            edge_id = attr['id']
            edge_map[edge_num] = edge_id
            edge_num += 1

    return edge_map


def get_mcs_info(g1, g2, edge_mappings):
    id_edge_map1 = get_id_edge_map(g1)
    id_edge_map2 = get_id_edge_map(g2)

    mcs_node_id_maps = []
    mcs_node_label_maps = []
    for edge_mapping in edge_mappings:
        node_id_map = get_node_id_map_from_edge_map(id_edge_map1, id_edge_map2, edge_mapping)
        node_label_map = node_id_map_to_label_map(g1, g2, node_id_map)
        mcs_node_id_maps.append(node_id_map)
        mcs_node_label_maps.append(node_label_map)
    return mcs_node_id_maps, mcs_node_label_maps


def node_id_map_to_label_map(g1, g2, node_id_map):
    node_label_map = {}
    for (source1, target1), (source2, target2) in node_id_map.items():
        g1_edge = (g1.node[source1]['label'], g1.node[target1]['label'])
        g2_edge = (g2.node[source2]['label'], g2.node[target2]['label'])
        node_label_map[g1_edge] = g2_edge
    return node_label_map


def get_node_id_map_from_edge_map(id_edge_map1, id_edge_map2, edge_mapping):
    node_map = {}
    for edge1, edge2 in edge_mapping.items():
        nodes_edge1 = id_edge_map1[edge1]
        nodes_edge2 = id_edge_map2[edge2]
        nodes1 = (nodes_edge1[0], nodes_edge1[1])
        nodes2 = (nodes_edge2[0], nodes_edge2[1])
        node_map[nodes1] = nodes2
    return node_map


def get_id_edge_map(graph):
    id_edge_map = {}
    for u, v, edge_data in graph.edges(data=True):
        edge_id = edge_data['id']
        assert edge_id not in id_edge_map
        id_edge_map[edge_id] = (u, v)
    return id_edge_map


def get_mcs_path():
    return get_root_path() + '/model/mcs'


def write_java_input_file(g1, g2, algo, filepath):
    """Prepares and writes a file in JSON format for MCS calculation."""
    write_data = {}
    write_data['graph1'] = graph_as_dict(g1)
    write_data['graph2'] = graph_as_dict(g2)
    write_data['algorithm'] = algo
    # Assume there's at least one node and get its attributes
    test_node_attr = g1.nodes_iter(data=True).__next__()[1]
    # This is the actual key we want the MCS algorithm to use to compare node labels. The
    # Java MCS code has a default "unlabeled" key, so for unlabeled graphs, can just use that.
    write_data['nodeLabelKey'] = 'type' if 'type' in test_node_attr else 'unlabeled'

    with open(filepath, 'w') as jsonfile:
        json.dump(write_data, jsonfile)


def graph_as_dict(graph):
    dict = {}
    dict['directed'] = nx.is_directed(graph)
    dict['gid'] = graph.graph['gid']
    dict['nodes'] = []
    dict['edges'] = []
    for node, attr in graph.nodes(data=True):
        node_data = {}
        node_data['id'] = node
        node_data['label'] = attr['label']
        if 'type' in attr:
            node_data['type'] = attr['type']
        dict['nodes'].append(node_data)
    for source, target, attr in graph.edges(data=True):
        dict['edges'].append({'id': attr['id'], 'source': source, 'target': target})
    return dict


def gen_BA(n_nodes, m_edges_density, query_size):
    assert n_nodes >= m_edges_density
    assert n_nodes >= query_size
    from networkx.generators.random_graphs import barabasi_albert_graph as BA
    g1 = BA(n_nodes, m_edges_density)
    # g2 = nx.from_numpy_matrix(nx.to_numpy_matrix(g1, nodelist=list(range(query_size))))
    g2 = nx.Graph(g1.subgraph(list(range(query_size))))
    g1.graph = {'gid': 0, 'glabel': 0, 'mode': 'static', 'node_default': {}, 'edge_default': {}}
    g2.graph = {'gid': 1, 'glabel': 1, 'mode': 'static', 'node_default': {}, 'edge_default': {}}
    for g in [g1, g2]:
        for nid in g.node:
            g.node[nid]['label'] = nid
        eid = 0
        history = set()
        for nid1 in g.edge:
            for nid2 in g.edge[nid1]:
                if (min(nid1, nid2), max(nid1, nid2)) not in history:
                    g.edge[nid1][nid2]['id'] = eid
                    eid += 1
                history.add((min(nid1, nid2), max(nid1, nid2)))
    return g1, g2


def assign_node_label_edge_id(g):
    for nid in g.nodes():
        g.node[nid]['label'] = str(nid)
    eid = 0
    history = set()
    for nid1, nid2, edata in g.edges(data=True):
        if (min(nid1, nid2), max(nid1, nid2)) not in history:
            edata['id'] = str(eid)
            eid += 1
        history.add((min(nid1, nid2), max(nid1, nid2)))
    return g


from utils import get_save_path, load
from load_data import load_dataset


def load_dataset_wrapper(name, tvt, align_metric, node_ordering, skip_pairs=False, node_feats=None):
    # TODO: following code may be buggy for non-road datasets
    dir = join(get_save_path(), 'OurModelData')

    sfn = '{}_train_test_{}_{}_{}'.format(
        name, align_metric, node_ordering,
        '_'.join(['one_hot', 'local_degree_profile']))
    tp = join(dir, sfn)
    # version option
    tp += '_{}'.format(node_feats)
    rtn = load(tp)
    if rtn is not None:
        if len(rtn) == 0:
            raise ValueError('Weird empty loaded dict')
        train_data, test_data = rtn['train_data'], rtn['test_data']
        dataset = train_data.dataset
    else:
        dataset = load_dataset(name, tvt, align_metric, node_ordering)

    return dataset

def mcs_wrapper(name, algo, recursion_threshold, save_every_iter, timeout, save_every_seconds, fp_prepend_info):
    if name == 'aids700nef' or name == 'linux':
        tvt = 'all'
        align_metric = 'mcs'
        node_ordering = 'bfs'
    elif 'syn' in name:
        tvt = 'train'
        align_metric = 'random'
        node_ordering = None
    elif 'circuit' in name:
        tvt = 'test'
        align_metric = 'mcs'
        node_ordering = None
    else:
        tvt = 'test'
        align_metric = 'mcs'
        node_ordering = 'bfs'
        # assert False

    dataset = load_dataset_wrapper(name, tvt, align_metric, node_ordering)
    # [{5: 1, 7: 2, 8: 0, 1: 3, 4: 6, 6: 8, 9: 4}]
    if name == 'aids700nef':
        gid1, gid2 = 6, 39
        gid1, gid2 = 6, 307
        # gid1, gid2 = 6, 7544
    elif name == 'linux':
        gid1, gid2 = 430, 3
        # gid1, gid2 = 382, 511
    elif 'syn' in name:
        gid1, gid2 = 0, 1
    elif 'circuit_graph' == name:
        gid1, gid2 = 0, 1
    else:
        gid1, gid2 = 0, 1
        # assert False

    i=0
    mcs_size_list = []
    for (gid1, gid2) in dataset.pairs:
        if i < 50:
            i+=1
        else:
            break
        g1 = dataset.look_up_graph_by_gid(gid1).get_nxgraph()
        g2 = dataset.look_up_graph_by_gid(gid2).get_nxgraph()

        g1 = assign_node_label_edge_id(g1)
        g2 = assign_node_label_edge_id(g2)

        # algo = 'mccreesh2017'
        # algo = 'mcsp+rl'
        # algo = 'mcsp_py'

        from utils import node_has_type_attrib

        if name == 'circuit_graph':
            labeled = True
            label_key = '|'.join(['is_device', 'name', 'type', 'port'])
        else:
            labeled = node_has_type_attrib(g1)
            label_key = 'type'
        out = mcs(
            g1, g2, algo, labeled=labeled, label_key=label_key,  # k_down
            recursion_threshold=recursion_threshold, save_every_seconds=save_every_seconds,
            save_every_iter=save_every_iter,
            timeit=True,
            timeout=timeout,
            debug=True,
            computer_name='username',
            fp_prepend_info=fp_prepend_info)

        if algo == 'mccreesh2017' or algo == 'mcsp+rl':
            mcs_size, mcs_node_mapping, refined_mcs_node_mapping, refined_mcs_edge_mapping, elapsed_time = out
        elif algo == 'k_down':
            mcs_size, idx_mapping, edge_mapping, elapsed_time = out
        else:
            assert False

        mcs_size_list.append(mcs_size)
        '''
        Tricky: If the underlying cpp does not timeout and abort "properly",
        the python exec_cmd with timeout along cannot work, i.e. the process cannot
        be terminated only by python side.
        For some magical reason, the python side will be notified that the cpp
        has timed out only when the cpp really times out 
        (check mccreesh2017 mcsp.c for details).
        '''
        print('elapsed_time:\t', elapsed_time)

    import numpy as np
    mcs_avg = np.mean(np.array(mcs_size_list))
    with open('/home/username/Documents/temp.txt', 'a') as the_file:
        the_file.write(f'{fp_prepend_info}\t {mcs_avg}\n')


def main():
    config_list = [
        {'name':'iters',
         'recursion_threshold':7500,
         'save_every_iter':True,#False,
         'timeout':0,
         'save_every_seconds':-1},
    ]

    name_list = [
        'duogexf::roadNet-CA_rw_1957_1;roadNet-CA_rw_1957_2',
    ]
    algo_list = [
        'mccreesh2017',
        'mcsp+rl',
        # 'k_down'
    ]

    for config in config_list:
        for algo in algo_list:
            for name in name_list:
                if algo == 'k_down' and config['name'] == 'iters':
                    continue
                fp_prepend_info = f'{algo}_{config["name"]}_{name.replace("|","_")}'#name.split("::")[-1]}'
                mcs_wrapper(name, algo,
                            config['recursion_threshold'],
                            config['save_every_iter'],
                            config['timeout'],
                            config['save_every_seconds'],
                            fp_prepend_info)

if __name__ == '__main__':
    main()
