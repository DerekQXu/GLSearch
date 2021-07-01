from solve_parent_dir import cur_folder
from utils import sorted_nicely, get_ts, load
from config import FLAGS
from os.path import join, dirname
import torch
from collections import OrderedDict
from scipy.stats import mstats
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except:
    print('did not load seaborn')

def check_flags():
    # if FLAGS.node_feat_name:
    #     assert (FLAGS.node_feat_encoder == 'onehot')
    # else:
    #     assert ('constant_' in FLAGS.node_feat_encoder)
    # assert (0 < FLAGS.valid_percentage < 1)
    assert (FLAGS.layer_num >= 0)
    assert (FLAGS.batch_size >= 1)
    # assert (FLAGS.num_epochs >= 0)
    # assert (FLAGS.iters_val_start >= 1)
    # assert (FLAGS.iters_val_every >= 1)
    d = vars(FLAGS)
    ln = d['layer_num']
    ls = [False] * ln
    for k in d.keys():
        if 'layer_' in k and 'gc' not in k and 'branch' not in k and 'id' not in k:
            lt = k.split('_')[1]
            if lt != 'num':
                i = int(lt) - 1
                if not (0 <= i < len(ls)):
                    raise RuntimeError('Wrong spec {}'.format(k))
                ls[i] = True
    for i, x in enumerate(ls):
        if not x:
            raise RuntimeError('layer {} not specified'.format(i + 1))
    if 'cuda' in FLAGS.device:
        gpu_id = int(FLAGS.device.split(':')[1])
        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError('Wrong GPU ID {}; {} available GPUs'.
                             format(FLAGS.device, gpu_count))
    # TODO: finish.


def get_flag(k, check=False):
    if hasattr(FLAGS, k):
        return getattr(FLAGS, k)
    else:
        if check:
            raise RuntimeError('Need flag {} which does not exist'.format(k))
        return None


def get_flags_with_prefix_as_list(prefix):
    rtn = []
    d = vars(FLAGS)
    i_check = 1  # one-based
    for k in sorted_nicely(d.keys()):
        v = d[k]
        sp = k.split(prefix)
        if len(sp) == 2 and sp[0] == '' and sp[1].startswith('_'):
            id = int(sp[1][1:])
            if i_check != id:
                raise ValueError('Wrong flag format {}={} '
                                 '(should start from _1'.format(k, v))
            rtn.append(v)
            i_check += 1
    return rtn


def load_replace_flags():
    assert FLAGS.load_model is not None
    loaded_flags = load(join(dirname(dirname(FLAGS.load_model)), 'FLAGS.klepto'))['FLAGS']
    excluded_flags = {'device', 'dataset', 'split_by', 'node_feats_for_mcs',
                      'node_feats_for_soft_mcs', 'tvt_options', 'num_iters',
                      'only_iters_for_debug', 'user', 'hostname', 'ts',
                      'train_test_ratio', 'Q_BD', 'mcsplit_heuristic_on_iter_one',
                      'load_model', 'use_cached_gnn', 'long_running_val_mcsp',
                      'animation_size', 'recursion_threshold', 'promise_mode', 'regret_iters',
                      'explore_n_pairs', 'prune_n_bd_by_Q', 'save_every_recursion_count',
                      'eps_argmin', 'buffer_type', 'compute_loss_during_testing',
                      'debug_first_train_iters', 'restore_bidomains', 'no_pruning',
                      'mcsplit_heuristic_perc', 'populate_reply_buffer_every_iter',
                      'total_runtime', 'sample_all_edges', 'priority_correction',
                      'sample_all_edges_thresh', 'plot_final_tree', 'shuffle_input',
                      'time_analysis', 'no_search_tree', 'dataset_list', 'num_bds_max',
                      'num_nodes_max', 'val_every_iter', 'val_debug', 'plot_final_tree',
                      'drought_iters', 'val_method_list', 'beam_search',
                      'num_bds_max', 'num_nodes_degree_max', 'randQ'}
    for k in vars(loaded_flags):
        if k in excluded_flags:
            continue
        cur_v = getattr(FLAGS, k)
        loaded_v = getattr(loaded_flags, k)
        if cur_v != loaded_v:
            if 'layer' not in k:
                setattr(FLAGS, k, loaded_v)
                print('\t{}={}\n\t\tto {}={}'.format(k, cur_v, k, loaded_v))
                continue
            cur_l_flags = _get_layer_flags(cur_v)
            loaded_l_flags = _get_layer_flags(loaded_v)
            replaced_l_flags = _get_replaced_layer_flags(
                cur_l_flags, loaded_l_flags, excluded_flags)
            replaced_layer_s = ','.join(['{}={}'.format(k, v) for k, v in replaced_l_flags.items()])
            setattr(FLAGS, k, replaced_layer_s)
            print('\t{}={}\n\t\tto {}={}'.format(k, cur_v, k, replaced_layer_s))
            a = cur_v
            b = replaced_layer_s
            splitA = set(a.split(','))
            splitB = set(b.split(','))
            diff = splitB.difference(splitA)
            diff = ','.join(diff)
            print('\t\t\t diff:', diff)
    print('Done loading FLAGS')
    # exit(-1)


def _get_layer_flags(layer_s):
    layer_split = layer_s.split(',')
    assert len(layer_split) >= 1
    rtn = OrderedDict()
    for s in layer_split:
        ss = s.split('=')
        assert len(ss) == 2
        rtn[ss[0]] = ss[1]
    return rtn


def _get_replaced_layer_flags(cur_l_flags, loaded_l_flags, excluded_flags):
    rtn = OrderedDict()
    for k in cur_l_flags:
        if k in excluded_flags:
            rtn[k] = cur_l_flags[k]
        elif k not in loaded_l_flags:
            rtn[k] = cur_l_flags[k]
        else:  # flags such as 'Q_mode'
            rtn[k] = loaded_l_flags[k]
    return rtn


def get_branch_names():
    bnames = get_flag('branch_names')
    if bnames:
        rtn = bnames.split(',')
        if len(rtn) == 0:
            raise ValueError('Wrong number of branches: {}'.format(bnames))
        return rtn
    else:
        assert bnames is None
        return None


def extract_config_code():
    with open(join(get_our_dir(), 'config.py')) as f:
        return f.read()


def convert_long_time_to_str(sec):
    def _give_s(num):
        return '' if num == 1 else 's'

    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    return '{} day{} {} hour{} {} min{} {:.3f} sec{}'.format(
        int(day), _give_s(int(day)), int(hour), _give_s(int(hour)),
        int(minutes), _give_s(int(minutes)), seconds, _give_s(seconds))


def get_our_dir():
    return cur_folder


def get_model_info_as_str():
    rtn = []
    d = vars(FLAGS)
    for k in d.keys():
        v = str(d[k])
        if k == 'dataset_list':
            s = '{0:26} : {1}'.format(k, v)
            rtn.append(s)
        else:
            vsplit = v.split(',')
            assert len(vsplit) >= 1
            for i, vs in enumerate(vsplit):
                if i == 0:
                    ks = k
                else:
                    ks = ''
                if i != len(vsplit) - 1:
                    vs = vs + ','
                s = '{0:26} : {1}'.format(ks, vs)
                rtn.append(s)
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)


def get_model_info_as_command():
    rtn = []
    d = vars(FLAGS)
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '--{}={}'.format(k, v)
        rtn.append(s)
    return 'python {} {}'.format(join(get_our_dir(), 'main.py'), '  '.join(rtn))


def debug_tensor(tensor, g1=None, g2=None):
    xxx = tensor.detach().cpu().numpy()
    # if g1 != None and g2 != None:
    #     import networkx as nx
    #     import matplotlib.pyplot as plt
    #     plt.subplot(121)
    #     nx.draw(g1, with_labels=True)  -
    #     plt.subplot(122)
    #     nx.draw(g2, with_labels=True)
    #     plt.show()
    return


TDMNN = None


# Let me know you question I turn on your voice now. ..............................................

def get_train_data_max_num_nodes(train_data):
    global TDMNN
    if TDMNN is None:
        TDMNN = train_data.dataset.stats['#Nodes']['Max']
    return TDMNN


def pad_extra_rows(g1x, g2x, padding_value=0):  # g1x and g2x are 2D tensors
    max_dim = max(g1x.shape[0], g2x.shape[0])

    x1_pad = torch.nn.functional.pad(g1x, (0, 0, 0, (max_dim - g1x.shape[0])),
                                     mode='constant',
                                     value=padding_value)
    x2_pad = torch.nn.functional.pad(g2x, (0, 0, 0, (max_dim - g2x.shape[0])),
                                     mode='constant',
                                     value=padding_value)

    return x1_pad, x2_pad


def plot_dist(data, label, save_dir, saver=None, analyze_dist=True, bins=None):
    if analyze_dist:
        _analyze_dist(saver, label, data)
    fn = f'distribution_{label}.png'
    plt.figure()
    sns.set()
    ax = sns.distplot(data, bins=bins, axlabel=label)
    plt.xlabel(label)
    ax.figure.savefig(join(save_dir, fn))
    plt.close()


def _analyze_dist(saver, label, data):
    if saver is None:
        func = print
    else:
        func = saver.log_info
    func(f'--- Analyzing distribution of {label} (len={len(data)})')
    if np.isnan(np.sum(data)):
        func(f'{label} has nan')
    probs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999, 0.99999]
    quantiles = mstats.mquantiles(data, prob=probs)
    func(f'{label} {len(data)}')
    s = '\t'.join([str(x) for x in probs])
    func(f'\tprob     \t {s}')
    s = '\t'.join(['{:.2f}'.format(x) for x in quantiles])
    func(f'\tquantiles\t {s}')
    func(f'\tnp.min(data)\t {np.min(data)}')
    func(f'\tnp.max(data)\t {np.max(data)}')
    func(f'\tnp.mean(data)\t {np.mean(data)}')
    func(f'\tnp.std(data)\t {np.std(data)}')


def plot_heatmap(data, label, save_dir, saver=None, analyze_dist=False):
    if analyze_dist:
        _analyze_dist(saver, label, data)
    if saver is None:
        func = print
    else:
        func = saver.log_info
    func(f'--- Plotting heatmap of {label} (shape={len(data.shape)})')
    if np.isnan(np.sum(data)):
        func(f'{label} has nan')
    fn = f'heatmap_{label}.png'
    plt.figure()
    sns.heatmap(data, cmap='YlGnBu').figure.savefig(join(save_dir, fn))
    ax = sns.distplot(data, axlabel=label)
    plt.title(label)
    plt.close()


def plot_scatter_line(data_dict, label, save_dir):
    fn = f'scatter_{label}_iterations.png'
    ss = ['rs-','b^-','g^-','c^-','m^-','ko-','yo-']
    cs = [s[0] for s in ss]
    plt.figure()
    i = 0

    # min_size = min([len(x['incumbent_data']) for x in data_dict.values()])
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li, y_li = [], []

        # min_len = float('inf')
        # for x in data_dict_elt['incumbent_data']:
        #     if x[1] < min_len:
        #         min_len = x[1]

        for x in data_dict_elt['incumbent_data']:
            # if x[1] > FLAGS.recursion_threshold:
            #     break
            x_li.append(x[1])
            y_li.append(x[0])
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    plt.close()

    plt.figure()
    fn = f'scatter_{label}_time.png'
    i = 0
    for line_name, data_dict_elt in sorted(data_dict.items()):
        x_li = [x[2] for x in data_dict_elt['incumbent_data']]
        y_li = [x[0] for x in data_dict_elt['incumbent_data']]
        plt.scatter(np.array(x_li), np.array(y_li), label=line_name, color=cs[i % len(cs)])
        plt.plot(np.array(x_li), np.array(y_li), ss[i % len(ss)])
        i += 1

    plt.title(label)
    plt.grid(True)
    plt.legend()
    plt.axis('on')
    plt.savefig(join(save_dir, fn), bbox_inches='tight')
    # plt.close()
