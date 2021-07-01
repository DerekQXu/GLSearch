from dataset import OurOldDataset
from load_classic_data import iterate_get_graphs
from dist_sim_converter import get_ds_metric_config
from graph_pair import GraphPair
from dataset import OurDataset
from utils import get_data_path, load_pickle
from os.path import join
from collections import OrderedDict
import numpy as np


def load_old_data(name, natts, eatts, tvt, align_metric, node_ordering, glabel, skip_pairs):
    assert glabel is None
    dir_name = _get_dir_name(name, tvt, align_metric)
    train_gs, test_gs, all_gs, dist_or_sim, true_algo = \
        _get_gs_and_metric_info(name, dir_name, natts, eatts, align_metric)
    if tvt == 'train':
        gs1, gs2 = train_gs, train_gs
        pairs = _get_gs_pairs(
            gs1, gs2, name, dir_name, align_metric, true_algo, dist_or_sim)
    elif tvt == 'test':
        gs1, gs2 = test_gs, train_gs  # test by train
        pairs = _get_gs_pairs(
            gs1, gs2, name, dir_name, align_metric, true_algo, dist_or_sim)
    else:
        assert False
    rtn = OurOldDataset(name, gs1, gs2, all_gs, natts, eatts, pairs, tvt,
                        align_metric, node_ordering, None)
    return rtn
    # exit(-1)


def _get_dir_name(name, tvt, align_metric):
    assert tvt in ['train', 'test']
    assert align_metric in ['ged', 'mcs']
    if name in ['aids700nef_old', 'aids700nef_old_small']:
        dir_name = 'AIDS700nef'
    elif name == 'linux_old':
        dir_name = 'LINUX'
    elif name == 'imdbmulti_old':
        dir_name = 'IMDBMulti'
    elif name == 'ptc_old':
        dir_name = 'PTC'
    else:
        raise NotImplementedError()
    return dir_name


def _get_gs_and_metric_info(name, dir_name, natts, eatts, align_metric):
    train_gs = iterate_get_graphs(join(get_data_path(), dir_name, 'train'),
                                  natts=natts, eatts=eatts)
    test_gs = iterate_get_graphs(join(get_data_path(), dir_name, 'test'),
                                 natts=natts, eatts=eatts)
    if name == 'aids700nef_old_small':
        train_gs = train_gs[0:4]
        test_gs = test_gs[0:2]
    graphs = train_gs + test_gs
    dist_or_sim, true_algo = get_ds_metric_config(align_metric)
    return train_gs, test_gs, graphs, dist_or_sim, true_algo


def _get_gs_pairs(gs1, gs2, name, dir_name, align_metric, true_algo,
                  dist_or_sim):
    fn = '{}_{}_{}_gidpair_{}_map.pickle'.format(
        name.replace('_old', '').replace('_small', ''),
        align_metric, true_algo, dist_or_sim)
    fp = join(get_data_path(), dir_name, fn)
    ds_map = load_pickle(fp)
    if ds_map is None:
        raise ValueError('Please obtain the ds_map and put it as {}'.format(fp))
    assert type(ds_map) is OrderedDict
    pairs = {}
    # print(len(gs1), len(gs2))
    for g1 in gs1:
        for g2 in gs2:
            gid_pair = (g1.gid(), g2.gid())
            ds_true = ds_map.get(gid_pair)
            ds_true = _check_gid_pair_ds_true(gid_pair, ds_true, fn, ds_map)
            pairs[gid_pair] = GraphPair(ds_true=ds_true)
    return pairs


def _check_gid_pair_ds_true(gid_pair, ds_true, fn, ds_map):
    if ds_true is None:
        raise ValueError('gid_pair {} not in {} which has {} '
                         'entries'.
                         format(gid_pair, fn, len(ds_map)))
    if ds_true < 0:
        raise ValueError('gid_pair {} has ds_true={} < 0'.
                         format(gid_pair, ds_true))
    if type(ds_true) is np.float64 or type(ds_true) is int \
            or type(ds_true) is np.int64:
        ds_true_int = int(ds_true)
        if ds_true_int != ds_true:
            raise ValueError('gid_pair {} has ds_true={} '
                             'not actually an int'.
                             format(gid_pair, ds_true))
        ds_true = ds_true_int
    else:
        raise ValueError('gid_pair {} has ds_true={} '
                         'neither np.float64 nor int nor np.int64 {}'.
                         format(gid_pair, ds_true, type(ds_true)))
    return ds_true
