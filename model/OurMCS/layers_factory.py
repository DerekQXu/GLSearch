from config import FLAGS
if FLAGS.scalable:
    from layers_MCSRL_backtrack_scalable import MCSRLBacktrack
else:
    from layers_MCSRL_backtrack import MCSRLBacktrack
from node_feat import obtain_nfe_feat_idx_div
import torch.nn as nn


def create_layers(model, pattern, num_layers):
    layers = nn.ModuleList()
    for i in range(1, num_layers + 1):  # 1-indexed
        sp = vars(FLAGS)['{}_{}'.format(pattern, i)].split(':')
        name = sp[0]
        layer_info = {}
        if len(sp) > 1:
            assert (len(sp) == 2)
            for spec in sp[1].split(','):
                ssp = spec.split('=')
                layer_info[ssp[0]] = '='.join(ssp[1:])  # could have '=' in layer_info
        if name in layer_ctors:
            layers.append(layer_ctors[name](layer_info, model, i, layers))
        else:
            raise ValueError('Unknown layer {}'.format(name))
    return layers

def create_MCSRL_backtrack(lf, model, *unused):
    return MCSRLBacktrack(in_dim=model.num_node_feat,
                          tot_num_train_pairs=sum([x.gid1gid2_list.shape[0] for x in model.train_data]),
                          feat_map=obtain_nfe_feat_idx_div(model.train_data[0].dataset, FLAGS.node_feats),
                          Q_sampling=lf['Q_sampling'],
                          DQN_mode=lf['DQN_mode'],
                          Q_BD=_parse_as_bool(lf['Q_BD']),
                          loss_fun=lf['loss_fun'],
                          q_signal=lf['q_signal'],
                          sync_target_frames=int(lf['sync_target_frames']),
                          beta_reward=float(lf['beta_reward']),
                          perc_IL=float(lf['perc_IL']),
                          buffer_start_iter=int(lf['buffer_start_iter']),
                          buffer_size=int(lf['buffer_size']),
                          sample_size=int(lf['sample_size']),
                          sample_all_edges=_parse_as_bool(lf['sample_all_edges']),
                          sample_all_edges_thresh=int(lf['sample_all_edges_thresh']),
                          eps_testing=_parse_as_bool(lf['eps_testing']),
                          recursion_threshold=int(lf['recursion_threshold']),
                          total_runtime=int(lf['total_runtime']),
                          save_every_recursion_count=int(lf['save_every_recursion_count']),
                          save_every_runtime=float(lf['save_every_runtime']),
                          mcsplit_heuristic_on_iter_one=_parse_as_bool(lf['mcsplit_heuristic_on_iter_one']),
                          restore_bidomains=_parse_as_bool(lf['restore_bidomains']),
                          no_pruning=_parse_as_bool(lf['no_pruning']),
                          regret_iters=int(lf['regret_iters']),
                          populate_reply_buffer_every_iter=int(lf['populate_reply_buffer_every_iter']),
                          encoder_type=lf['encoder_type'],
                          embedder_type=lf['embedder_type'],
                          interact_type=lf['interact_type'],
                          n_dim=int(lf['n_dim']),
                          n_layers=int(lf['n_layers']),
                          GNN_mode=lf['GNN_mode'],
                          learn_embs=lf['learn_embs'],
                          layer_AGG_w_MLP=lf['layer_AGG_w_MLP'],
                          Q_mode=lf['Q_mode'],
                          Q_act=lf['Q_act'],
                          disentangle_search_tree=_parse_as_bool(lf['disentangle_search_tree']),
                          mcsp_before_perc=float(lf['mcsp_before_perc'])
                          )


"""
Register the constructor caller function here.
"""
layer_ctors = {
    'MCSRL_backtrack': create_MCSRL_backtrack
}


def _check_spec(allowed_nums, lf, ln):
    if len(lf) not in allowed_nums:
        raise ValueError('{} layer must have {} specs NOT {} {}'.
                         format(ln, allowed_nums, len(lf), lf))


def _parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string {}'.format(b))


def _parse_as_int_list(il):
    rtn = []
    for x in il.split('_'):
        x = int(x)
        rtn.append(x)
    return rtn