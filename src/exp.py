#!/usr/bin/env python3
from load_data import load_dataset
from utils import get_result_path, create_dir_if_not_exists, \
    get_ts, exec_turnoff_print, prompt, prompt_get_computer_name, \
    prompt_get_cpu, \
    slack_notify, node_has_type_attrib
from dist_sim_handler import ged, mcs, assign_node_label_edge_id
import traceback

import multiprocessing as mp
import numpy as np
import matplotlib
from pandas import read_csv

# Fix font type for ACM paper submission.
# matplotlib.use('Agg')
matplotlib.rc('font', **{'family': 'serif', 'size': 22})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def exp(debug_print=False, dataset_name=None, tvt=None, ds_metric=None, node_ordering=None,
        algo=None, timeout=None,
        recursion_threshold=None, save_every_seconds=None,
        num_cpu=None, computer_name=None):
    """ Run baselines on real datasets. Take a while. """
    if dataset_name is None:
        dataset_name = prompt('Which dataset?')
    if tvt is None:
        tvt = prompt('tvt?')
    if ds_metric is None:
        ds_metric = prompt('Which metric (ged|mcs|random)?', options=['ged', 'mcs', 'random'])
    if node_ordering is None:
        node_ordering = prompt('Node ordering (None|bcs)?', options=['None', 'bfs'])
    if node_ordering == 'None':
        node_ordering = None
    dataset = load_dataset(dataset_name, tvt, ds_metric, node_ordering)
    graphs = [assign_node_label_edge_id(g.get_nxgraph()) for g in dataset.gs]
    if algo is None:
        algo = prompt('Which algorthm?')
    if timeout is None:
        timeout_temp = prompt('Time limit in sec? Empty for no limit')
        timeout = float(timeout_temp) if timeout_temp else None
    if recursion_threshold is None or save_every_seconds is None:
        recursion_threshold, save_every_seconds = prompt_for_recursion_threshold(
            ds_metric, algo)
    if not debug_print:
        exec_turnoff_print()  # turn off printing from exec commands
    if num_cpu is None:
        num_cpu = prompt_get_cpu()
    if computer_name is None:
        computer_name = prompt_get_computer_name()
    try:
        real_dataset_run_helper(computer_name, dataset, ds_metric, algo,
                                recursion_threshold, save_every_seconds,
                                graphs, num_cpu, timeout)
    except Exception as e:
        traceback.print_exc()
        slack_notify('machine {}, data {}, algo {}, exp1 error'.format(
            computer_name, dataset.name, algo))
    else:
        slack_notify('machine {}, data {}, algo {}, exp1 complete'.format(
            computer_name, dataset.name, algo))


def real_dataset_run_helper(computer_name, dataset, ds_metric, algo,
                            recursion_threshold, save_every_seconds,
                            graphs,
                            num_cpu, timeout):
    outdir, func, labeled, label_key, m, n, \
    exsiting_entries, is_symmetric, smart_needed, \
    chunk_needed, chunk_num, chunk_id = \
        prompt_for_params(dataset, ds_metric, graphs)
    assert m == n
    # Multiprocessing.
    pool = mp.Pool(processes=num_cpu)
    # Submit to pool workers.
    results = {}
    pairs_to_run = get_all_pairs_to_run(graphs, smart_needed,
                                        chunk_needed, chunk_num, chunk_id,
                                        sorted(list(dataset.pairs.keys())))

    addi_info_s = '{}_{}{}'. \
        format('sorted' if smart_needed else 'nosort',
               'chunked_{}_{}pairs'.format(chunk_num,
                                           chunk_id) if chunk_needed else 'whole',
               len(pairs_to_run))
    csv_fn = '{}/csv/{}_{}_{}{}_{}_{}_{}cpus_{}.csv'.format(
        outdir, ds_metric, dataset.name, algo,
        get_recursion_threshold_str(recursion_threshold),
        get_ts(), computer_name, num_cpu,
        addi_info_s)
    file = open(csv_fn, 'w')
    print('Saving to {}'.format(csv_fn))

    assert m == n
    submit_to_pool_workers(func, pairs_to_run, graphs, m, n,
                           exsiting_entries, is_symmetric, ds_metric, results,
                           pool, algo, dataset, labeled, label_key,
                           recursion_threshold, save_every_seconds, timeout,
                           computer_name, num_cpu)
    if ds_metric == 'ged':
        print_and_log('i,j,i_gid,j_gid,i_node,j_node,i_edge,j_edge,ged,lcnt,time(msec)',
                      file)
    else:
        print_and_log('i,j,i_gid,j_gid,i_node,j_node,i_edge,j_edge,mcs,node_mapping,'
                      'refined_node_mapping,refined_edge_mapping,time(msec)',
                      file)
    # Retrieve results from pool workers or a loaded csv file (previous run).
    retrieve_from_pool_workers(pairs_to_run, graphs, m, n,
                               exsiting_entries, is_symmetric, ds_metric, results,
                               algo, dataset, computer_name, num_cpu,
                               file)  # ds_mat, time_mat updated
    file.close()
    # save_as_np(outdir, ds_metric, ds_mat, time_mat, get_ts(),
    #            dataset, algo, computer_name, num_cpu, addi_info_s)


def prompt_for_params(dataset, ds_metric, graphs):
    if ds_metric == 'ged':
        func = ged
        labeled, label_key = None, None  # only needed by MCS
    elif ds_metric == 'mcs' or ds_metric == 'random':
        func = mcs
        # For MCS, since the solver can handle labeled and unlabeled graphs, but the compressed
        # encoding must be labeled (need to tell it to ignore labels or not).
        # TODO: this should go in some kind of config file specific for mcs
        if node_has_type_attrib(graphs[0]):
            labeled = True
            label_key = 'type'
            print('Has node type')
        else:
            labeled = False
            label_key = ''
            print('Does not have node type')
    else:
        raise RuntimeError('Unknown distance similarity metric {}'.format(ds_metric))
    m = len(graphs)
    n = len(graphs)
    # ds_mat = np.full((m, n), -1, dtype=int)
    # time_mat = np.full((m, n), -1., dtype=float)
    outdir = '{}/{}'.format(get_result_path(), dataset.name)
    create_dir_if_not_exists(outdir + '/csv')
    create_dir_if_not_exists(outdir + '/{}'.format(ds_metric))
    create_dir_if_not_exists(outdir + '/time')
    # exsiting_csv = prompt('File path to exsiting csv files?')
    exsiting_csv = '' # TODO
    exsiting_entries = load_from_exsiting_csv(exsiting_csv, ds_metric,
                                              skip_eval=False)
    # is_symmetric = prompt('Is the ds matrix symmetric? (1/0)',
    #                       options=['0', '1']) == '1'
    is_symmetric = False # TODO
    if is_symmetric:
        assert (m == n)
    # smart_needed = prompt('Is smart pair sorting needed? (1/0)',
    #                       options=['0', '1']) == '1'
    smart_needed = False # TODO
    chunk_needed, chunk_num, chunk_id = prompt_for_chunk_info()
    return outdir, func, labeled, label_key, m, n, \
           exsiting_entries, is_symmetric, smart_needed, \
           chunk_needed, chunk_num, chunk_id


def prompt_for_recursion_threshold(ds_metric, algo):
    recursion_threshold = None
    save_every_seconds = None
    if ds_metric in ['mcs', 'random'] and algo in ['mccreesh2017', 'mcsp+rl',
                                                   'mcsp_py']:
        while True:
            recursion_threshold = \
                prompt('mcsp recursion threshold? 0 no limit')
            try:
                recursion_threshold = int(recursion_threshold)
                if recursion_threshold >= 0:
                    break
            except:
                continue
        while True:
            save_every_seconds = \
                prompt('mcsp save_every_seconds? -1 no periodic saving')
            try:
                save_every_seconds = int(save_every_seconds)
                if save_every_seconds >= 0:
                    break
            except:
                continue
    return recursion_threshold, save_every_seconds


def get_recursion_threshold_str(recursion_threshold):
    if recursion_threshold is None:
        return ''
    else:
        return '_recthresh_{}'.format(recursion_threshold)


def submit_to_pool_workers(func, pairs_to_run, graphs, m, n,
                           exsiting_entries, is_symmetric, ds_metric, results,
                           pool, algo, dataset, labeled, label_key,
                           recursion_threshold, save_every_seconds, timeout, computer_name, num_cpu):
    cur_hit = 0
    for k, (i, j) in enumerate(pairs_to_run):
        g1, g2 = graphs[i], graphs[j]
        i_gid, j_gid = g1.graph['gid'], g2.graph['gid']
        if (i_gid, j_gid) in exsiting_entries:
            continue
        if is_symmetric and (j_gid, i_gid) in exsiting_entries:
            continue
        if ds_metric in ['mcs', 'random']:
            # print('#######', labeled, label_key)
            results[(i, j)] = pool.apply_async(
                func, args=(g1, g2, algo, labeled, label_key,
                            recursion_threshold, save_every_seconds,
                            True, True,
                            timeout, computer_name,))
        else:
            results[(i, j)] = pool.apply_async(
                func, args=(g1, g2, algo, True, True, timeout,))
        cur_hit = print_progress(
            k, len(pairs_to_run), 'submit: {} {} {} {} cpus;'.
                format(algo, dataset.name, computer_name, num_cpu), cur_hit)


def retrieve_from_pool_workers(pairs_to_run, graphs, m, n,
                               exsiting_entries, is_symmetric, ds_metric, results,
                               algo, dataset, computer_name, num_cpu,
                               file):
    cur_hit = 0
    for k, (i, j) in enumerate(pairs_to_run):
        cur_hit = print_progress(
            k, len(pairs_to_run), 'work: {} {} {} {} {} cpus;'.
                format(ds_metric, algo, dataset.name, computer_name, num_cpu),
            cur_hit, hits=None)
        g1, g2 = graphs[i], graphs[j]
        i_gid, j_gid = g1.graph['gid'], g2.graph['gid']
        if (i, j) not in results:
            lcnt, mcs_node_mapping, mcs_edge_mapping = None, None, None
            tmp = exsiting_entries.get((i_gid, j_gid))
            if tmp:
                if ds_metric == 'ged':
                    i_gid, j_gid, i_node, j_node, ds, lcnt, t = tmp
                else:
                    i_gid, j_gid, i_node, j_node, ds, mcs_node_mapping, \
                    refined_mcs_node_mapping, \
                    refined_mcs_edge_mapping, t = get_from
            else:
                assert (is_symmetric)
                get_from = exsiting_entries[(j_gid, i_gid)]
                if ds_metric == 'ged':
                    j_gid, i_gid, j_node, i_node, ds, lcnt, t = get_from
                else:
                    j_gid, i_gid, j_node, i_node, ds, mcs_node_mapping, \
                    refined_mcs_node_mapping, \
                    refined_mcs_edge_mapping, t = get_from
            if ds_metric == 'ged':
                assert (lcnt is not None)
                assert (g1.graph['gid'] == i_gid)
                assert (g2.graph['gid'] == j_gid)
                assert (g1.number_of_nodes() == i_node)
                assert (g2.number_of_nodes() == j_node)
                s = form_ged_print_string(i, j, g1, g2, ds, lcnt, t)
            else:
                assert (mcs_node_mapping is not None and
                        mcs_edge_mapping is not None)
                s = form_mcs_print_string(
                    i, j, g1, g2, ds, mcs_node_mapping,
                    refined_mcs_node_mapping, refined_mcs_edge_mapping, t)
        else:
            if ds_metric == 'ged':
                ds, lcnt, g1_a, g2_a, t = results[(i, j)].get()
                i_gid, j_gid, i_node, j_node = \
                    g1.graph['gid'], g2.graph['gid'], \
                    g1.number_of_nodes(), g2.number_of_nodes()
                assert (g1.number_of_nodes() == g1_a.number_of_nodes())
                assert (g2.number_of_nodes() == g2_a.number_of_nodes())
                exsiting_entries[(i_gid, j_gid)] = \
                    (i_gid, j_gid, i_node, j_node, ds, lcnt, t)
                s = form_ged_print_string(i, j, g1, g2, ds, lcnt, t)
            else:  # MCS
                ds, mcs_node_mapping, \
                    refined_mcs_node_mapping, \
                    refined_mcs_edge_mapping, t = \
                    results[(i, j)].get()
                exsiting_entries[(i_gid, j_gid)] = \
                    (ds, mcs_node_mapping,
                    refined_mcs_node_mapping,
                    refined_mcs_edge_mapping, t)
                s = form_mcs_print_string(
                    i, j, g1, g2, ds, mcs_node_mapping,
                    refined_mcs_node_mapping, refined_mcs_edge_mapping, t)
        print_and_log(s, file)
        # if ds_metric == 'mcs' and (i_gid, j_gid) in exsiting_entries:
        # # Save memory, clear the mappings since they're saved to file.
        #     exsiting_entries[(i_gid, j_gid)] = list(exsiting_entries[(i_gid, j_gid)])
        #     exsiting_entries[(i_gid, j_gid)][1] = {}
        #     exsiting_entries[(i_gid, j_gid)][2] = {}
        # ds_mat[i][j] = ds
        # time_mat[i][j] = t


def prompt_for_chunk_info():
    # chunk_needed = prompt('Is dividing the pairs to run into chunks needed? (1/0)',
    #                       options=['0', '1']) == '1'
    chunk_needed = False # TODO
    chunk_num, chunk_id = None, None
    if chunk_needed:
        while True:
            chunk_num = prompt('Number of chunks: (positive int)')
            try:
                chunk_num = int(chunk_num)
                assert (chunk_num >= 1)
                while True:
                    try:
                        chunk_id = prompt('Chunk id of this run: '
                                          '(0, 1, ..., <num_chunk>-1)')
                        chunk_id = int(chunk_id)
                        assert (chunk_id >= 0 and chunk_id < chunk_num)
                    except:
                        continue
                    break
            except:
                continue
            break
    return chunk_needed, chunk_num, chunk_id


def get_all_pairs_to_run(gs, smart_needed,
                         chunk_needed, chunk_num, chunk_id, pair_gids):
    rtn = get_pairs_from_pair_gids(gs, pair_gids)
    if smart_needed:
        print('Sorting {} pairs'.format(len(rtn)))
        rtn = sorted(rtn)
        print('Sorted {} pairs'.format(len(rtn)))
    for k in range(len(rtn)):
        _, i, j = rtn[k]
        # print(_, i, j)
        rtn[k] = (i, j)
    if chunk_needed:
        rtn = list(split_list_into_n_chunks(rtn, chunk_num))
        print('Divided into {} chunks with pairs: {}'.format(
            len(rtn), [len(x) for x in rtn] if len(rtn) <= 10 else '...'))
        rtn = rtn[chunk_id]
        print('Run chunk {} with {} pairs'.format(chunk_id, len(rtn)))
    return rtn


def get_pairs_from_pair_gids(row_gs, pair_gids):
    rtn = []
    gid_to_id = {}  # this ensures the returned pairs match with the pair_gids list
    for i, g in enumerate(row_gs):
        gid_to_id[g.graph['gid']] = i
    for (gid1, gid2) in pair_gids:
        i, j = gid_to_id[gid1], gid_to_id[gid2]
        g1, g2 = row_gs[i], row_gs[j]
        rtn.append((g1.number_of_nodes() + g2.number_of_nodes(), i, j))
    return rtn


def split_list_into_n_chunks(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def form_ged_print_string(i, j, g1, g2, ds, lcnt, t):
    return '{},{},{},{},{},{},{},{},{},{},{:.2f}'.format(
        i, j, g1.graph['gid'], g2.graph['gid'],
        g1.number_of_nodes(), g2.number_of_nodes(),
        g1.number_of_edges(), g2.number_of_edges(),
        ds, lcnt, t)


def form_mcs_print_string(i, j, g1, g2, ds, mcs_node_mapping,
                    refined_mcs_node_mapping, refined_mcs_edge_mapping, t):
    return '"{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}","{}"'.format(
        i, j, g1.graph['gid'], g2.graph['gid'],
        g1.number_of_nodes(), g2.number_of_nodes(),
        g1.number_of_edges(), g2.number_of_edges(),
        str(ds), str(mcs_node_mapping), str(refined_mcs_node_mapping),
        str(refined_mcs_node_mapping), str(t))


def post_real_dataset_run_convert_csv_to_np():
    """ Use in case only csv is generated,
        and numpy matrices need to be saved. """
    dataset = 'imdbmulti'
    model = 'CDKMCS'
    ds_metric = 'mcs'
    row_graphs = load_data(dataset, False).graphs
    col_graphs = load_data(dataset, True).graphs
    num_cpu = 40
    computer_name = 'scai1_all'
    ts = '2018-10-09T13:41:13.942414'
    outdir = '{}/{}'.format(get_result_path(), dataset)
    csv_fn = '{}/csv/{}_{}_{}_{}_{}_{}cpus.csv'.format(
        outdir, ds_metric, dataset, model, ts, computer_name, num_cpu)
    data = load_from_exsiting_csv(csv_fn, ds_metric)
    m = len(row_graphs)
    n = len(col_graphs)
    # -3 is identifier that the csv the data came from didn't include the data point.
    ds_mat = np.full((m, n), -3)
    time_mat = np.full((m, n), -3.)
    cnt = 0
    print('m: {}, n: {}, m*n: {}'.format(m, n, m * n))
    for (i, j), row_data in data.items():
        if cnt % 1000 == 0:
            print(cnt)
        ds_mat[i][j] = row_data[4]
        time_mat[i][j] = row_data[6] if ds_metric == 'ged' else row_data[7]
        cnt += 1
    print(cnt)
    assert (cnt == m * n)
    raise NotImplementedError()  # TODO: fix
    # save_as_np(outdir, ds_metric, ds_mat, time_mat, ts,
    #            dataset, model, computer_name, num_cpu)


def save_as_np(outdir, ds_metric, ds_mat, time_mat, ts,
               dataset, model, computer_name, num_cpu, addi_info_s):
    s = '{}_{}_{}_{}_{}cpus_{}'.format(
        dataset,
        model, ts, computer_name, num_cpu, addi_info_s)
    np.save('{}/{}/{}_{}_mat_{}'.format(
        outdir, ds_metric, ds_metric, ds_metric, s), ds_mat)
    np.save('{}/time/{}_time_mat_{}'.format(outdir, ds_metric, s), time_mat)


def print_progress(cur, tot, label, cur_hit, hits=(0, 0.2, 0.4, 0.6, 0.8)):
    perc = cur / tot
    if hits is None or (cur_hit < len(hits) and abs(perc - hits[cur_hit]) <= 0.05):
        print('----- {} progress: {}/{}={:.1%}'.format(label, cur, tot, perc))
        cur_hit += 1
    return cur_hit


def print_and_log(s, file):
    print(s)
    file.write(s + '\n')
    # file.flush() # less disk I/O (hopefully)


def load_from_exsiting_csv(csv_fn, ds_metric, skip_eval=True):
    rtn = {}
    if csv_fn:
        data = read_csv(csv_fn)
        for _, row in data.iterrows():
            i = int(row['i'])
            j = int(row['j'])
            # if j > 2:
            #     return rtn
            i_gid = int(row['i_gid'])
            j_gid = int(row['j_gid'])
            i_node = int(row['i_node'])
            j_node = int(row['j_node'])
            t = float(row['time(msec)'])
            if ds_metric == 'ged':
                lcnt = int(row['lcnt'])
                d = int(row['ged'])
                rtn[(i_gid, j_gid)] = (i_gid, j_gid, i_node, j_node, d, lcnt, t)
            elif ds_metric == 'mcs':
                d = int(row['mcs'])
                # Check the case where there was an error and we need to rerun it.
                if d < 0:
                    continue
                if skip_eval:
                    rtn[(i_gid, j_gid)] = (i_gid, j_gid, i_node, j_node, d, None, None, t)
                else:
                    # Check node mappings are right data format.
                    node_mappings = eval(row['node_mapping'])
                    assert isinstance(node_mappings, list), 'node_mapping must be a list'
                    if len(node_mappings) > 0:
                        assert isinstance(node_mappings[0],
                                          dict), 'node_mapping items must be dicts'
                        # assert isinstance(list(node_mappings[0].keys())[0], tuple), 'node_mapping keys must be tuples'
                        # assert isinstance(list(node_mappings[0].values())[0],
                        #                   tuple), 'node_mapping values must be tuples'
                    # Check edge mappings are right data format.
                    edge_mappings = eval(row['edge_mapping'])
                    assert isinstance(edge_mappings, list), 'edge_mapping must be a list'
                    if len(edge_mappings) > 0:
                        assert isinstance(edge_mappings[0],
                                          dict), 'edge_mapping items must be dicts'
                        # assert isinstance(list(edge_mappings[0].keys())[0], str), 'edge_mapping keys must be str'
                        # assert isinstance(list(edge_mappings[0].keys())[0], str), 'edge_mapping values must be str'
                    rtn[(i_gid, j_gid)] = (
                        i_gid, j_gid, i_node, j_node, d, node_mappings, edge_mappings, t)
            else:
                raise RuntimeError('Did not handle ds_metric parsing for {}'.format(ds_metric))
    print('Loaded {} entries from {}'.format(len(rtn), csv_fn))
    return rtn


if __name__ == '__main__':
    # debug_print = False
    # dataset_name = None
    # tvt = None
    # align_metric = None
    # algo = None
    # timeout = None
    # recursion_threshold = None
    # save_every_seconds = None
    # num_cpu = None
    # computer_name = None

    # dataset = 'syn:np_tr=1000,np_te=100,nn_core=40,nn_tot=64,ed=2,gen_type=BA'
    # dataset = 'syn:np_tr=1000,np_te=100,nn_core=32,nn_tot=64,ed=2,gen_type=BA'
    # 'syn:np_tr=1000,np_te=100,nn_core=40,nn_tot=64,ed=0.2|4,gen_type=WS'


    dataset_name_list = [ \
        'syn:np_tr=1000,np_te=100,nn_core=32,nn_tot=64,ed=2,gen_type=BA', \
        # 'syn:np_tr=1000,np_te=100,nn_core=40,nn_tot=64,ed=2,gen_type=BA',\
        'syn:np_tr=1000,np_te=100,nn_core=32,nn_tot=64,ed=0.2|4,gen_type=WS',\
        # 'syn:np_tr=1000,np_te=100,nn_core=40,nn_tot=64,ed=0.2|4,gen_type=WS',\
        'syn:np_tr=1000,np_te=100,nn_core=32,nn_tot=64,ed=0.07,gen_type=ER',\
        # 'syn:np_tr=1000,np_te=100,nn_core=40,nn_tot=64,ed=0.07,gen_type=ER',\
    ]
    algo_list = ['mcsp+rl']#, 'mccreesh2017'] #['mcsp_py']#,'mccreesh2017', 'mcsp+rl']
    threshold_list = [100]
    threshold_list = [90, 80, 70, 60, 50, 40, 30, 20, 10]

    debug_print = False
    tvt = 'test'
    align_metric = 'random'
    node_ordering = 'None'
    timeout = 0
    save_every_seconds = -1
    num_cpu = 5
    computer_name = 'username'

    for algo in algo_list:
        for recursion_threshold in threshold_list:
            for dataset_name in dataset_name_list:
                exp(debug_print, dataset_name, tvt, align_metric, node_ordering,
                    algo, timeout,
                    recursion_threshold, save_every_seconds, num_cpu, computer_name)

