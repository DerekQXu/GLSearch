from solve_parent_dir import solve_parent_dir
from dataset_config import get_dataset_conf
from dist_sim_converter import get_ds_metric_config
from utils import format_str_list, C, get_user, get_host
import argparse
import torch

solve_parent_dir()
parser = argparse.ArgumentParser()

"""
Data.
"""

""" 
dataset: 
    (for MCS)
    debug, mini_debug, debug_no-1, mini_debug_no-1 debug_single_iso
    mcsplain mcsplain-connected sip (tune sip with smaller D/batch_size)
    ptc redditmulti10k
    mcs33ve (dropped) mcs33ve-connected (dropped)
    aids700nef linux imdbmulti ptc nci109 webeasy redditmulti10k mutag
    (for similarity)
    aids700nef_old linux_old imdbmulti_old ptc_old aids700nef_old_small
"""
# prev best
load_model = None
parser.add_argument('--load_model', default=load_model)

skip_all_iters_but = -1
parser.add_argument('--skip_all_iters_but', type=int, default=skip_all_iters_but)

model = 'MCSRL_backtrack'
threshold_var = 50  # used for csv_baseline
stk_ver = True  # used for csv_baseline
parser.add_argument('--model', default=model)

dataset = 'cocktail'

dataset_list = [
    ([('aids700nef', 30),
      ('linux', 30),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=16,ed=5,gen_type=BA', -1),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=14,ed=0.14,gen_type=ER', -1),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=18,ed=0.2|2,gen_type=WS', -1),
      ], 2500),
    ([('ptc', 30),
      ('imdbmulti', 30),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=32,ed=4,gen_type=BA', -1),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=30,ed=0.12,gen_type=ER', -1),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=34,ed=0.2|2,gen_type=WS', -1),
      ], 2500),
    ([('mutag', 30),
      ('redditmulti10k', 30),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=48,ed=4,gen_type=BA', -1),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=46,ed=0.1,gen_type=ER', -1),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=50,ed=0.2|4,gen_type=WS', -1),
      ], 2500),
    ([('webeasy', 30),
      ('mcsplain-connected', 30),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=64,ed=3,gen_type=BA', -1),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=62,ed=0.08,gen_type=ER', -1),
      ('syn:np_tr=20,np_te=20,nn_core=-1,nn_tot=66,ed=0.2|4,gen_type=WS', -1),
      ], 2500)
]

drought_iters = 1e9
parser.add_argument('--drought_iters', type=list, default=drought_iters)

parser.add_argument('--dataset_list', type=list, default=dataset_list)
parser.add_argument('--dataset', default=dataset)

split_by = 'graph'
if 'mcsplain' in dataset or 'circuit' in dataset:
    split_by = 'pair'
parser.add_argument('--split_by', default=split_by)

dataset_version = None  # 'v2' # NOTE: NEVER CHANGE THIS WILL CAUSE TROUBLE!!!!
parser.add_argument('--dataset_version', default=dataset_version)

filter_large_size = None  # if dataset == 'redditmulti10k' else None
parser.add_argument('--filter_large_size', type=int, default=filter_large_size)  # None or >= 1

# deprecated
shrink_graph_for_RLMCS = -1  # 100 # IMPORTANT: ONLY AFFECTS RLMCS!!!
parser.add_argument('--shrink_graph_for_RLMCS', type=int, default=shrink_graph_for_RLMCS)

select_node_pair = None
parser.add_argument('--select_node_pair', type=str, default=select_node_pair)  # None or gid1_gid2

"""
Model.
"""

n_outputs = 1
parser.add_argument('--n_outputs', type=int, default=n_outputs)

hard_mask = True
parser.add_argument('--hard_mask', type=bool, default=hard_mask)

model_name = 'fancy'
parser.add_argument('--model_name', default=model_name)

c = C()

D = 70

theta = 0
parser.add_argument('--theta', type=float, default=theta)

training_mode = True

n = '--layer_{}'.format(c.c())
populate_reply_buffer_every_iter = -1
buffer_size = 1024
sample_size = 32
perc_IL = -1

sync_target_frames = 100
buffer_start_iter = 1 if 'circuit' in dataset else 11

DQN_mode = 'tgt_q_network'  # 'tgt_q_network'#'tgt_q_network'#'mcsp_degree'#'rand'
sample_all_edges = False
sample_all_edges_thresh = -1

# TODO: try with Q_BD
Q_BD = True
beta_reward = 0 # deprecated!
regret_iters = 3

mcsp_before_perc = 0.1  # tune to 1.0
recursion_threshold = -1  # -1 if training_mode else 6000  # 30#200#12000#8000#7000#1000
total_runtime = -1  # 100000 # msec
save_every_recursion_count = -1  # 2000#500 if huge_graph else 100
save_every_runtime = -1  # 10000 # msec

loss_fun = 'mse'  # deprecated!
q_signal = 'fitted-tgt-leaf'  # multisample
no_pruning = False
restore_bidomains = True if load_model is None else False
mcsplit_heuristic_on_iter_one = False

# Q_sampling = 'canonical_0.0001_1.0_0.05'
# Q_sampling = 'canonical_0.1_1.0_0.2'
Q_sampling = 'canonical_0.000016_0.16_0.01'
eps_testing = False

encoder_type = 'abcd'
embedder_type = 'abcd'
interact_type = 'dvn'  # 'abcd4'
n_dim = 16#32#16
n_layers = 3
GNN_mode = 'GAT'  # 'JSE_ATT' #TODO: if v4 is good -> turn into GAT?; TODO:
learn_embs = True
layer_AGG_w_MLP = True
Q_mode = '8'  # ''ourQ;g1+g2_sg1+sg2_bd1+bd2;x1+x2'
Q_act = 'elu'  # elu+1'  # 'elu+1'#'identity'
disentangle_search_tree = False

animation_size = -1  # print out this many graphs per pair!
debug_first_train_iters = 100 if load_model is None else 1
s = '{}:Q_sampling={},DQN_mode={},Q_BD={},loss_fun={},q_signal={},' \
    'sync_target_frames={},beta_reward={},perc_IL={},buffer_start_iter={},' \
    'buffer_size={},sample_size={},sample_all_edges={},sample_all_edges_thresh={},' \
    'eps_testing={},recursion_threshold={},total_runtime={},save_every_recursion_count={},' \
    'save_every_runtime={},mcsplit_heuristic_on_iter_one={},restore_bidomains={},' \
    'no_pruning={},regret_iters={},populate_reply_buffer_every_iter={},encoder_type={},' \
    'embedder_type={},interact_type={},n_dim={},n_layers={},GNN_mode={},learn_embs={},' \
    'layer_AGG_w_MLP={},Q_mode={},Q_act={},disentangle_search_tree={},mcsp_before_perc={}' \
    .format(model, Q_sampling, DQN_mode, Q_BD, loss_fun, q_signal, sync_target_frames,
            beta_reward, perc_IL, buffer_start_iter, buffer_size, sample_size,
            sample_all_edges, sample_all_edges_thresh, eps_testing, recursion_threshold,
            total_runtime,
            save_every_recursion_count, save_every_runtime, mcsplit_heuristic_on_iter_one,
            restore_bidomains, no_pruning, regret_iters, populate_reply_buffer_every_iter,
            encoder_type, embedder_type, interact_type, n_dim, n_layers, GNN_mode, learn_embs,
            layer_AGG_w_MLP, Q_mode, Q_act, disentangle_search_tree, mcsp_before_perc)
parser.add_argument(n, default=s)

logging = 'end'
parser.add_argument('--logging', type=str, default=logging)
smarter_bin_sampling = False
parser.add_argument('--smarter_bin_sampling', type=bool, default=smarter_bin_sampling)
smart_bin_sampling = True
parser.add_argument('--smart_bin_sampling', type=bool, default=smart_bin_sampling)
subtract_sg_size = False
parser.add_argument('--subtract_sg_size', type=bool, default=subtract_sg_size)
priority_correction = False
parser.add_argument('--priority_correction', type=bool, default=priority_correction)

debug_loss_threshold = None # 500  # 0.64
# debug_loss_threshold = 150#50
parser.add_argument('--debug_loss_threshold', type=float, default=debug_loss_threshold)
plot_final_tree = True
parser.add_argument('--plot_final_tree', type=bool, default=plot_final_tree)
sample_entire_replay_buffer = False
parser.add_argument('--sample_entire_replay_buffer', type=bool,
                    default=sample_entire_replay_buffer) # deprecated
no_trivial_pairs = True
parser.add_argument('--no_trivial_pairs', type=bool, default=no_trivial_pairs)
search_path = training_mode  # True if training_mode else False
parser.add_argument('--search_path', type=bool, default=search_path)
down_proj_by = 4
parser.add_argument('--down_proj_by', type=int, default=down_proj_by)
with_bdgnn = False
parser.add_argument('--with_bdgnn', type=bool, default=with_bdgnn)
with_gnn_per_action = False
parser.add_argument('--with_gnn_per_action', type=bool, default=with_gnn_per_action)
max_chunk_size = 64
parser.add_argument('--max_chunk_size', type=int, default=max_chunk_size)
a2c_networks = False  # False # True
parser.add_argument('--a2c_networks', type=bool, default=a2c_networks)

interact_ops = ['32', '1dconv+max_1', 'add']
parser.add_argument('--interact_ops', default=interact_ops)
run_bds_MLP_before_interact = False
parser.add_argument('--run_bds_MLP_before_interact', type=bool,
                    default=run_bds_MLP_before_interact)

inverse_bd_size_order = False
parser.add_argument('--inverse_bd_size_order', type=bool, default=inverse_bd_size_order)
num_bds_max = 1
num_nodes_degree_max = 20*num_bds_max
num_nodes_dqn_max = -1
parser.add_argument('--num_bds_max', type=int, default=num_bds_max)
parser.add_argument('--num_nodes_degree_max', type=int, default=num_nodes_degree_max)
parser.add_argument('--num_nodes_dqn_max', type=int, default=num_nodes_dqn_max)

val_every_iter = 100 if load_model is None else 1
parser.add_argument('--val_every_iter', type=int, default=val_every_iter)

parser.add_argument('--val_debug', type=bool, default=False)

clipping_val = -1
parser.add_argument('--clipping_val', type=float, default=clipping_val)
supervised_before = 1250 if load_model is None else -1
parser.add_argument('--supervised_before', type=int, default=supervised_before)
imitation_before = 3750 if load_model is None else -1
parser.add_argument('--imitation_before', type=int, default=imitation_before)
recursion_threshold = 80 if load_model is None else -1# if dvn_mode else -1
parser.add_argument('--recursion_threshold', type=int, default=recursion_threshold)
total_runtime = -1 if load_model is None else 6000
parser.add_argument('--total_runtime', type=int, default=total_runtime)
long_running_val_mcsp = False
parser.add_argument('--long_running_val_mcsp', type=bool, default=long_running_val_mcsp)
promise_mode = 'diverse'
parser.add_argument('--promise_mode', default=promise_mode)
# Below are for NeuralMCS.
binarize_q_true = False
parser.add_argument('--binarize_q_true', type=bool, default=binarize_q_true)
# loss_func = 'MSE'
loss_func = 'MSE'  # 'BCEWithLogits'
parser.add_argument('--loss_func', default=loss_func)

attention_bds = False
parser.add_argument('--attention_bds', type=bool, default=attention_bds)
simplified_sg_emb = True
parser.add_argument('--simplified_sg_emb', type=bool, default=simplified_sg_emb)
emb_mode_list = ['gs', 'sgs', 'abds', 'ubds']
parser.add_argument('--emb_mode_list', type=list, default=emb_mode_list)
# default_emb = 'learnable'
default_emb = 'learnable'
parser.add_argument('--default_emb', default=default_emb)
parser.add_argument('--normalize_emb', type=bool, default=True)
beam_search = None
parser.add_argument('--beam_search', default=beam_search)

use_cached_gnn = False
parser.add_argument('--use_cached_gnn', type=bool, default=use_cached_gnn)

parser.add_argument('--batched_logging', default=True)
parser.add_argument('--randQ', default=False)
parser.add_argument('--val_method_list', default=['dqn'])#,'mcspv2','mcsprl'])

use_mcsp_policy = False
parser.add_argument('--use_mcsp_policy', type=bool, default=use_mcsp_policy)

exclude_root = False
parser.add_argument('--exclude_root', type=bool, default=exclude_root)

parser.add_argument('--layer_num', type=int, default=c.t())

########################

natts, eatts, tvt_options, align_metric_options, *_ = \
    get_dataset_conf(dataset)

""" Must use exactly one alignment metric across the entire run. """
align_metric = align_metric_options[0]
if len(align_metric_options) == 2:
    """ Choose which metric to use. """
    align_metric = 'ged'
    # align_metric = 'mcs'
parser.add_argument('--align_metric', default=align_metric)

dos_true, _ = get_ds_metric_config(align_metric)
parser.add_argument('--dos_true', default=dos_true)

# Assume the model predicts None. May be updated below.
dos_pred = None

c = C()

if 'pdb' in dataset:
    natts_mcs = ['type']
    natts_soft_mcs = ['type']
elif 'cocktail' in dataset and load_model is None:
    natts_mcs = []
    natts_soft_mcs = []
else:
    natts_mcs = natts
    natts_soft_mcs = natts

shuffle_input = False
parser.add_argument('--shuffle_input', type=bool, default=shuffle_input)

reward_calculator_mode = 'vanilla' # 'edge_count', 'normalized_edge_count', 'normalized_edge_count_hybrid;0.5'
parser.add_argument('--reward_calculator_mode', default=reward_calculator_mode)
encode_labels = False  # manually do this!
if encode_labels:
    assert False
    natts_encoding = natts
    parser.add_argument('--node_fe_{}'.format(c.c()), default='one_hot')
else:
    natts_encoding = []
    parser.add_argument('--node_fe_{}'.format(c.c()), default='one_hot')
    parser.add_argument('--node_fe_{}'.format(c.c()), default='local_degree_profile')

# TODO: we do not use strings here (for CLI)
parser.add_argument('--node_feats_for_mcs', default=natts_mcs)  # for iso-checking
parser.add_argument('--node_feats_for_soft_mcs',
                    default=natts_soft_mcs)  # for "tolerance (mentor graphics)"
parser.add_argument('--node_feats', default=format_str_list(natts_encoding))  # for one-hot encoding

parser.add_argument('--edge_feats', default=format_str_list(eatts))

# Finally we set dos_pred.
parser.add_argument('--dos_pred', default=dos_pred)

parser.add_argument('--tvt_options', default=format_str_list(tvt_options))

""" holdout, (TODO) <k>-fold. """
tvt_strategy = 'holdout'
parser.add_argument('--tvt_strategy', default=tvt_strategy)

if tvt_strategy == 'holdout':
    if tvt_options == ['all']:
        parser.add_argument('--train_test_ratio', type=float, default=0.8)
    elif tvt_options == ['train', 'test']:
        pass
    elif tvt_options == ['test']:
        pass
    else:
        raise NotImplementedError()
else:
    raise NotImplementedError()

parser.add_argument('--train_test_ratio', type=float, default=0.8)

parser.add_argument('--debug', type=bool, default='debug' in dataset)

"""
Optimization.
"""

lr = 1e-4
parser.add_argument('--lr', type=float, default=lr)

gpu = 0
device = str('cuda:{}'.format(gpu) if torch.cuda.is_available() and gpu != -1
             else 'cpu')
parser.add_argument('--device', default=device)

num_epochs = None  # how many times we can cycle through whole dataset
parser.add_argument('--num_epochs', type=int, default=num_epochs)

retain_graph = True if load_model is None else False  # True if 'RL' in model else False
parser.add_argument('--retain_graph', type=bool, default=retain_graph)

periodic_save = 1e9 if skip_all_iters_but > 0 else 100  # 1000 if 'RL' in model else 2000
parser.add_argument('--periodic_save', type=int, default=periodic_save)

num_iters = -1  # if 'RL' in model else 2000
parser.add_argument('--num_iters', type=int, default=num_iters)

validation = False  # using validation set?
parser.add_argument('--validation', type=bool, default=validation)

throw_away = 0  # throwing away data?
parser.add_argument('--throw_away', type=float, default=throw_away)

print_every_iters = 5
parser.add_argument('--print_every_iters', type=int, default=print_every_iters)

only_iters_for_debug = None  # 200  # only train and test this number of pairs
parser.add_argument('--only_iters_for_debug', type=int, default=only_iters_for_debug)

time_analysis = False  # currently only works for RL search unsupervised
parser.add_argument('--time_analysis', type=bool, default=time_analysis)

save_model = True  # TODO: tune this
parser.add_argument('--save_model', type=bool, default=save_model)

batch_size = 1 if 'RL' in model else 64
parser.add_argument('--batch_size', type=int, default=batch_size)

if 'syn' in dataset or 'pdb' in dataset:
    node_ordering = None
else:
    node_ordering = 'bfs'

parser.add_argument('--node_ordering', default=node_ordering)
parser.add_argument('--no_probability', default=False)
parser.add_argument('--positional_encoding', default=False)  # TODO: dataset.py cannot see this

parser.add_argument('--user', default=get_user())

parser.add_argument('--hostname', default=get_host())
parser.add_argument('--scalable', default=True)
parser.add_argument('--no_bd_MLPs', default=False)

FLAGS = parser.parse_args()