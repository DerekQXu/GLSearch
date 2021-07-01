# from mcsp_our import mcsp_python_interface
from batch import create_edge_index, create_adj_set
from config import FLAGS
from utils import OurTimer

from data_structures_search_tree import Bidomain, StateNode, ActionEdge, SearchTree, ActionSpaceData
from data_structures_buffer import BinBuffer
from data_structures_common import StackHeap, DoubleDict, DQNInput
from layers_dqn_v1 import Q_network_v1
from layers_pca import PCAModel
from layers_sgw import sgw
from reward_calculator import RewardCalculator
from saver import saver
from embedding_saver import EMBEDDING_SAVER

try:
    from vis import plot_node_mapping, plot_single_graph
except:
    print("Should not use vis when doing long running experiments with many pairs!")

import random
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from copy import deepcopy

'''
MCSRL:
    Solves for the exact MCS mapping using reinforcement learning.
'''


if FLAGS.time_analysis:
    TIMER = OurTimer()
else:
    TIMER = None


class McspVec():
    def __init__(self, ldeg, rdeg):
        self.ldeg = ldeg
        self.rdeg = rdeg

class BufferEntry():
    def __init__(self, edge, g1, g2, search_tree):
        self.edge = edge
        self.g1 = g1
        self.g2 = g2
        self.search_tree = search_tree

#########################################################################
# MCSRL Procedure
#########################################################################

is_positive = lambda x: x > 0
is_between_0_and_1 = lambda x: x > 0 and x <= 1

PRETRAIN_MODE = 'pr'
IMITATION_MODE = 'il'
TRAIN_MODE = 'tr'
TEST_MODE = 'te'


class ForwardConfig():
    def __init__(self, total_runtime, recursion_threshold,
                 q_signal, restore_bidomains, no_pruning,
                 search_path, training):
        self.total_runtime = total_runtime
        self.recursion_threshold = recursion_threshold
        self.q_signal = q_signal
        self.restore_bidomains = restore_bidomains
        self.no_pruning = no_pruning
        self.search_path = search_path
        self.training = training


class MethodConfig():
    def __init__(self, DQN_mode, regret_iters):
        self.DQN_mode = DQN_mode
        self.regret_iters = regret_iters


class MCSRLBacktrack(nn.Module):
    def __init__(self, in_dim, tot_num_train_pairs, feat_map, Q_sampling,
                 DQN_mode, Q_BD, loss_fun, q_signal, sync_target_frames,
                 beta_reward, perc_IL, buffer_start_iter, buffer_size,
                 sample_size, sample_all_edges, sample_all_edges_thresh,
                 eps_testing, recursion_threshold, total_runtime,
                 save_every_recursion_count, save_every_runtime,
                 mcsplit_heuristic_on_iter_one, restore_bidomains, no_pruning,
                 regret_iters, populate_reply_buffer_every_iter,
                 encoder_type, embedder_type, interact_type, n_dim, n_layers,
                 GNN_mode, learn_embs, layer_AGG_w_MLP, Q_mode, Q_act,
                 disentangle_search_tree, mcsp_before_perc
                 ):
        super(MCSRLBacktrack, self).__init__()

        self.init_var(buffer_size, int, is_positive)
        self.init_var(tot_num_train_pairs, int, is_positive)

        self.perc_IL = self.init_var(perc_IL, float, is_between_0_and_1, -1.0)
        self.sync_target_frames = self.init_var(sync_target_frames, int, is_positive)

        self.buffer_start_iter = self.init_var(buffer_start_iter, int, is_positive)
        self.sample_size = self.init_var(sample_size, int, is_positive)
        self.sample_all_edges = self.init_var(sample_all_edges, bool)
        self.sample_all_edges_thresh = self.init_var(sample_all_edges_thresh, int, is_positive,
                                                     float('inf'))
        self.Q_BD = self.init_var(Q_BD, bool)
        self.loss_fun = self.init_var(loss_fun, str)
        self.save_every_runtime = self.init_var(save_every_runtime, float, is_positive, None)
        self.save_every_recursion_count = self.init_var(save_every_recursion_count, int,
                                                        is_positive, None)
        self.mcsplit_heuristic_on_iter_one = self.init_var(mcsplit_heuristic_on_iter_one, bool)
        self.eps_testing = self.init_var(eps_testing, bool)
        self.animation_size = None  # 20
        self.populate_reply_buffer_every_iter = self.init_var(populate_reply_buffer_every_iter, int,
                                                              is_positive, None)
        self.disentangle_search_tree = self.init_var(disentangle_search_tree, bool)
        # if self.populate_reply_buffer_every_iter is not None and self.buffer_start_iter is not None:
        #     assert self.populate_reply_buffer_every_iter < self.buffer_start_iter
        # assert reward_calculator_mode in ['vanilla', 'weighted_reward', 'fuzzy_matching']
        # TODO: currently IL with feat_map is not supported
        self.init_var(feat_map, dict)
        # self.beta_reward = self.init_var(beta_reward, float, is_between_0_and_1, 0)
        # self.mcsp_before_perc = self.init_var(mcsp_before_perc, float, is_between_0_and_1, 1.1)

        self.seed = random.Random(123)
        self.global_iter_debugging = 20
        if FLAGS.smarter_bin_sampling:
            self.buffer = BinBuffer(buffer_size, sample_strat='sg', biased='full')
        elif FLAGS.smart_bin_sampling:
            self.buffer = BinBuffer(buffer_size, sample_strat='q_max', biased=None)
        else:
            self.buffer = BinBuffer(buffer_size, sample_strat=None, biased='biased')
        self.red_tickets = self.init_var(tot_num_train_pairs * self.perc_IL, float, is_positive, -1)
        self.is_dvn = 'dvn' in interact_type  # IMPORTANT TRICKY IMPLICATIONS TO LB LOSS_FUNCTION!
        self.reward_calculator = RewardCalculator(
            FLAGS.reward_calculator_mode, feat_map, self.calc_bound)

        self.dqn = Q_network_v1(encoder_type, embedder_type, interact_type, in_dim, n_dim,
                                n_layers, GNN_mode, learn_embs, layer_AGG_w_MLP, Q_mode, Q_act,
                                self.reward_calculator, self._environment)
        self.dqn_tgt = Q_network_v1(encoder_type, embedder_type, interact_type, in_dim, n_dim,
                                    n_layers, GNN_mode, learn_embs, layer_AGG_w_MLP, Q_mode,
                                    Q_act, self.reward_calculator, self._environment)

        if DQN_mode == 'pca':
            self.pca = PCAModel([in_dim] + [n_dim] * 3)
        else:
            self.pca = None

        self.debug_first_train_iters = 50
        self.debug_train_iter_counter = 0
        if FLAGS.loss_func == 'MSE':
            self.loss = nn.MSELoss()
        elif FLAGS.loss_func == 'BCEWithLogits':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError()

        self.init_var(Q_sampling, str)
        Q_eps_dec_each_iter, Q_eps_start, Q_eps_end = [float(x) for x in Q_sampling.split('_')[1:]]
        self.Q_eps_dec_each_iter, self.Q_eps_start, self.Q_eps_end = \
            self.init_var(Q_eps_dec_each_iter, float, is_between_0_and_1), \
            self.init_var(Q_eps_start, float, is_between_0_and_1), \
            self.init_var(Q_eps_end, float, is_between_0_and_1)

        self.reward_mode = 'None'

        ############################################
        self.forward_config_dict = \
            self.get_forward_config_dict(
                restore_bidomains, total_runtime, recursion_threshold, q_signal)

        self.method_config_dict = \
            self.get_method_config_dict(DQN_mode, regret_iters)
        self.train_counter = -1
        self.curriculum_info = defaultdict(dict)

    #########################################################
    # Forward Procedure
    #########################################################
    def forward(self, ins, batch_data, model, iter=None, cur_id=None):
        if FLAGS.time_analysis:
            TIMER.time_and_clear(f'forward iter {iter} start')

        forward_mode = self.get_forward_mode(iter)
        self.apply_forward_config(self.forward_config_dict[forward_mode])

        if forward_mode == PRETRAIN_MODE:
            loss = self._forward_pretrain(forward_mode, ins, batch_data, cur_id)
        elif forward_mode == IMITATION_MODE:
            loss = self._forward_imitation(forward_mode, ins, batch_data, cur_id)
        elif forward_mode == TRAIN_MODE:
            loss = self._forward_train(forward_mode, ins, batch_data, cur_id)
        elif forward_mode == TEST_MODE:
            loss = self._forward_test(forward_mode, ins, batch_data, cur_id, iter)
        else:
            assert False

        if self._tgt_net_sync_itr(forward_mode):
            self._sync_tgt_networks()

        # saver.curriculum_info = self.curriculum_info
        saver.curriculum_info = None
        return loss

    def _forward_pretrain(self, forward_mode, ins, batch_data, cur_id):
        # fill replay buffer
        self.apply_method_config(self.method_config_dict['mcspv2'])
        pair_list, state_init_list = \
            self._preprocess_forward(ins, batch_data, cur_id)
        self._forward_batch(forward_mode, cur_id, pair_list, state_init_list)

        # run loss function
        self.apply_method_config(self.method_config_dict['dqn'])
        loss = self._loss_wrapper(forward_mode)
        return loss

    def _forward_imitation(self, forward_mode, ins, batch_data, cur_id):
        # TODO: epsilon-greedy during imitation learning?

        # fill replay buffer
        self.apply_method_config(self.method_config_dict['mcspv2'])
        pair_list, state_init_list = \
            self._preprocess_forward(ins, batch_data, cur_id)
        self._forward_batch(forward_mode, cur_id, pair_list, state_init_list)

        # run loss function
        self.apply_method_config(self.method_config_dict['dqn'])
        loss = self._loss_wrapper(forward_mode)
        return loss

    def _forward_train(self, forward_mode, ins, batch_data, cur_id):
        # fill replay buffer
        self.apply_method_config(self.method_config_dict['dqn'])
        pair_list, state_init_list = \
            self._preprocess_forward(ins, batch_data, cur_id)
        self._forward_batch(forward_mode, cur_id, pair_list, state_init_list)

        # run loss function
        loss = self._loss_wrapper(forward_mode)
        return loss

    def _forward_test(self, forward_mode, ins, batch_data, cur_id, iter):
        for method in FLAGS.val_method_list:
            # run forward model
            self.apply_method_config(self.method_config_dict[method])
            pair_list, state_init_list = \
                self._preprocess_forward(ins, batch_data, cur_id)
            self._forward_batch(f'{forward_mode}-{method}', cur_id, pair_list, state_init_list)

        return None

    def _forward_batch(self, forward_mode, cur_id, pair_list, state_init_list):
        for pair, state_init in zip(pair_list, state_init_list):
            # run the search procedure
            search_tree = self._forward_single_tree(forward_mode, cur_id, pair, state_init)

            # extend buffer with search tree
            if self.training:
                buffer_entry_list = \
                    self.search_tree2buffer_entry_list(search_tree, pair)
                self.buffer.extend(buffer_entry_list)

    def _forward_single_tree(self, forward_mode, cur_id, pair, state_init):
        # initializations
        search_stack = StackHeap()
        search_stack.add(state_init, 0)
        search_tree = SearchTree(root=state_init)
        since_last_update_count = 0

        timer = OurTimer()
        recursion_count = 0
        incumbent_list = [['incumbent','recursion_count','time']]

        # run the search code
        incumbent = {}
        print('search starting!')
        while len(search_stack) != 0:
            # print(f'on iteration {recursion_count}')
            recursion_count += 1
            # TODO: put in logging!
            if recursion_count % 10000 == 0:
                print(f'{recursion_count} : {len(incumbent)}')

            # pop from stack
            cur_state, promise, since_last_update_count = \
                self.sample_search_stack(
                    search_stack, incumbent, since_last_update_count)

            # update incumbent
            if self.is_better_solution(cur_state, incumbent):
                incumbent = deepcopy(cur_state.nn_map)
                if FLAGS.logging == 'all':
                    incumbent_list.append(
                        [incumbent, recursion_count, timer.get_duration()])

            # check for exit conditions
            if self.exit_condition(recursion_count, timer, since_last_update_count):
                break

            # get action space
            action_space_data = \
                self.get_action_space_data_wrapper(cur_state, is_mcsp=self.get_is_mcsp())

            prune_flag = \
                self.prune_condition(cur_state, action_space_data, incumbent)

            if prune_flag:
                if self.training and self.search_path:
                    break
                else:
                    continue

            # run the policy
            action_edge, next_state, promise_tuple, q_pred = \
                self._forward_single_edge(
                    search_tree, cur_state, action_space_data, recursion_count)

            # update search stack
            promise, promise_new_state = promise_tuple
            (v, w) = action_edge.action
            cur_state.prune_action(
                v=v, w=w, remove_nodes=(not self.restore_bidomains))
            search_stack.add(cur_state, promise)
            search_stack.add(next_state, promise_new_state)
            search_tree.link_states(
                cur_state,
                action_edge,
                next_state,
                q_pred,
                self.reward_calculator.discount
            )

        incumbent_list.append(
            [incumbent, recursion_count, timer.get_duration()])

        self.post_process(search_tree, incumbent_list)

        return search_tree

    def _forward_single_edge(self, search_tree, state, action_space_data, recursion_count):

        # estimate the q values
        q_vec = self.compute_q_vec(state, action_space_data).detach()

        # greedily select the action using q values
        action, q_vec_idx, is_q_vec_idx_argmax = \
            self._rank_actions_from_q_vec(q_vec, state, action_space_data)
        q_pred = q_vec[q_vec_idx]

        if self.DQN_mode == 'fixedv_mcsprl':
            v, w = action
            for bd in state.bidomains:
                self.lgrade[v] += min(len(bd.left), len(bd.right))
                self.rgrade[w] += min(len(bd.left), len(bd.right))

        # compute the next_state and any accompanying metadata
        action_edge, next_state = \
            self._environment(
                state, action, q_vec_idx, recursion_count=recursion_count, real_update=True)

        # compute promise
        promise_tuple = self.find_promise(state, next_state)

        return action_edge, next_state, promise_tuple, q_pred

    #########################################################
    # Utility Functions
    #########################################################
    def get_forward_mode(self, iter):
        if iter % FLAGS.val_every_iter == 0:
            forward_mode = TEST_MODE
        else:
            self.train_counter += 1
            if self.train_counter < FLAGS.supervised_before:
                forward_mode = PRETRAIN_MODE
            elif self.train_counter < FLAGS.imitation_before:
                forward_mode = IMITATION_MODE
            else:
                forward_mode = TRAIN_MODE
        return forward_mode

    def get_forward_config_dict(self, restore_bidomains, total_runtime, recursion_threshold,
                                q_signal):
        self.q_signal = None
        self.recursion_threshold = None
        self.restore_bidomains = None
        self.total_runtime = None
        self.search_path = None
        self.no_pruning = None

        q_signal = self.init_var(q_signal, str)
        restore_bidomains = self.init_var(restore_bidomains, bool)
        total_runtime_train = self.init_var(total_runtime, int, is_positive, None)
        recursion_threshold_train = self.init_var(recursion_threshold, int, is_positive, None)
        total_runtime_test = self.init_var(FLAGS.total_runtime, int, is_positive, None)
        recursion_threshold_test = self.init_var(FLAGS.recursion_threshold, int, is_positive, None)

        forward_config_dict = {
            PRETRAIN_MODE:
                ForwardConfig(
                    10,
                    None,
                    'LB',
                    True,
                    True,
                    False,
                    True,
                ),
            IMITATION_MODE:
                ForwardConfig(
                    total_runtime_train,
                    recursion_threshold_train,
                    q_signal,
                    restore_bidomains,
                    False,
                    True,
                    True
                ),
            TRAIN_MODE:
                ForwardConfig(
                    total_runtime_train,
                    recursion_threshold_train,
                    q_signal,
                    restore_bidomains,
                    False,
                    True,
                    True
                ),
            TEST_MODE:
                ForwardConfig(
                    total_runtime_test,
                    recursion_threshold_test,
                    q_signal,
                    False,
                    False,
                    False,
                    False
                )
        }

        return forward_config_dict

    def get_method_config_dict(self, DQN_mode, regret_iters):
        self.DQN_mode = None
        self.regret_iters = None
        DQN_mode = self.init_var(DQN_mode, str)
        regret_iters = self.init_var(regret_iters, int, is_positive, None)

        method_config_dict = {
            'dqn':
                MethodConfig(
                    DQN_mode,
                    regret_iters
                ),
            'mcspv2':
                MethodConfig(
                    'fixedv_mcsp',
                    None
                ),
            'mcsprl':
                MethodConfig(
                    'fixedv_mcsprl',
                    None
                )
        }
        return method_config_dict

    def apply_forward_config(self, forward_config):
        self.total_runtime = forward_config.total_runtime
        self.recursion_threshold = forward_config.recursion_threshold
        self.q_signal = forward_config.q_signal
        self.restore_bidomains = forward_config.restore_bidomains
        self.no_pruning = forward_config.no_pruning
        self.search_path = forward_config.search_path
        self.training = forward_config.training

    def apply_method_config(self, method_config):
        self.DQN_mode = method_config.DQN_mode
        self.regret_iters = method_config.regret_iters

    def get_action_space_data_wrapper(self, state, is_mcsp=False):
        # get action space
        bds_unexhausted, bds_adjacent, bdids_adjacent = \
            state.get_action_space_bds()
        if len(bds_adjacent) == 0:
            action_space = self._get_empty_action_space()
            bds_pruned, bdids_pruned = [], []
        else:
            num_bds, num_nodes_degree, num_nodes_dqn = \
                self._get_prune_parameters(is_mcsp)

            # prune topK adjacent bidomains
            bds_pruned, bdids_pruned = \
                self._prune_topk_bidomains(
                    bds_adjacent, bdids_adjacent, num_bds)

            # prune top(L1/#bidomains) nodes
            bds_pruned, bdids_pruned = \
                self._prune_topk_nodes(
                    bds_pruned, bdids_pruned, num_nodes_degree, state, 'deg')

            # get possible node pairs from list of bidomains
            # all combinations of nodes from bd.left and bd.right for all bds
            action_space = self._format_action_space(bds_pruned, state)

        # put action space into a wrapper
        action_space_data = \
            ActionSpaceData(
                action_space, bds_unexhausted, bds_pruned, bdids_pruned)
        return action_space_data

    def _get_prune_parameters(self, is_mcsp):
        if is_mcsp:
            num_bds = 1
            num_nodes_degree = float('inf')
            num_nodes_dqn = float('inf')
        else:
            num_bds = \
                float('inf') \
                    if FLAGS.num_bds_max < 0 \
                    else FLAGS.num_bds_max
            num_nodes_degree = \
                float('inf') \
                    if FLAGS.num_nodes_degree_max < 0 \
                    else FLAGS.num_nodes_degree_max
            num_nodes_dqn = \
                float('inf') \
                    if FLAGS.num_nodes_dqn_max < 0 \
                    else FLAGS.num_nodes_dqn_max

        return num_bds, num_nodes_degree, num_nodes_dqn

    def _prune_topk_bidomains(self, bds, bdids, num_bds):
        # select for topk bidomains
        prune_flag = len(bds) > num_bds
        if prune_flag:
            bds_pruned, bdids_pruned = \
                self._filter_topk_bds_by_size(bds, bdids, num_bds)
        else:
            bds_pruned, bdids_pruned = bds, bdids

        return bds_pruned, bdids_pruned

    def _prune_topk_nodes(self, bds, bdids, num_nodes, state, method):
        # get L value (max number of nodes in each bidomain)
        num_nodes_per_bd = num_nodes // len(bds)

        # prune for topl nodes
        bds_pruned, bdids_pruned = [], []
        for k, (bd, bdid) in enumerate(zip(bds, bdids)):
            prune_flag_l = len(bd.left) > num_nodes_per_bd
            prune_flag_r = len(bd.right) > num_nodes_per_bd
            if prune_flag_l:
                left_domain = self._filter_topk_nodes_by_degree(
                    bd.left, num_nodes_per_bd, state.g1)
            else:
                left_domain = bd.left
            if prune_flag_r:
                right_domain = self._filter_topk_nodes_by_degree(
                    bd.right, num_nodes_per_bd, state.g2)
            else:
                right_domain = bd.right
            bds_pruned.append(Bidomain(left_domain, right_domain, bd.is_adj))
            bdids_pruned.append(bdid)
        return bds_pruned, bdids_pruned

    def _get_empty_action_space_data(self, state):
        bds_unexhausted, bds_adjacent, bdids_adjacent = \
            state.get_action_space_bds()
        action_space_data = \
            ActionSpaceData(self._get_empty_action_space(),
                            bds_unexhausted,
                            bds_adjacent,
                            bdids_adjacent)
        return action_space_data

    def _get_empty_action_space(self):
        return [[], [], []]

    def _format_action_space(self, bds, state):
        left_indices = []
        right_indices = []
        bd_indices = []
        # soft matching: possibly give diff scores to pairs
        for k, bd in enumerate(bds):
            for v in bd.left:
                for w in bd.right:
                    # bds only contain unexhausted nodes NOT unexhausted edges
                    #   -> MUST check here to ensure nodes aren't revisited!
                    if v in state.pruned_actions.l2r and \
                            w in state.pruned_actions.l2r[v]:
                        continue
                    left_indices.append(v)
                    right_indices.append(w)
                    bd_indices.append(k)

        action_space = [left_indices, right_indices, bd_indices]
        return action_space

    def _filter_topk_bds_by_size(self, bds, bd_indices, num_bds_max):
        degree_list = np.array([max(len(bd.left), len(bd.right)) for bd in bds])
        if FLAGS.inverse_bd_size_order:
            degree_list_sorted = degree_list.argsort(kind='mergesort')[::-1]
        else:
            degree_list_sorted = degree_list.argsort(kind='mergesort')
        indices = degree_list_sorted[:num_bds_max]
        return [bds[idx] for idx in indices], [bd_indices[idx] for idx in indices]

    def _filter_topk_nodes_by_degree(self, all_nodes, num_nodes_max, g):
        nodes = list(all_nodes)
        degree_list = np.array([g.degree[node] for node in nodes])
        indices = degree_list.argsort(kind='mergesort')[-num_nodes_max:][::-1]
        return [nodes[idx] for idx in indices]

    def init_var(self, val, val_type, fun_valid=lambda x: True, val_default='temp'):
        assert type(val) is val_type

        if fun_valid(val):
            return val
        else:
            if val_default is 'temp':
                assert False
            return val_default

    def _get_mcs_increase_key(self, cur_id, iter, mcs_increase_mcsp):
        pair_id = mcs_increase_mcsp.pair_id
        g1 = mcs_increase_mcsp.g1
        g2 = mcs_increase_mcsp.g2
        m = g1.number_of_nodes()
        n = g2.number_of_nodes()
        if 'src' in g1.graph.keys():
            assert g1.graph["src"] == g2.graph["src"]
            pair_id_orig = (g1.graph['orig_gid'], g2.graph['orig_gid'])
            graph_src = g1.graph["src"]
        else:
            pair_id_orig = None
            graph_src = 'None'
        return f'{cur_id} {iter} {pair_id} (m {m} n {n}) ({graph_src})'

    ##########################################################
    # Utility functions (forward procedure)
    ##########################################################

    def _preprocess_forward(self, ins, batch_data, cur_id):
        offset = 0
        state_init_list = []
        pair_list = batch_data.pair_list
        for pair in pair_list:
            # set up general input data
            g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
            ins_g1, ins_g2, offset = self.compute_ins(g1, g2, ins, offset)
            edge_index1, edge_index2 = \
                create_edge_index(g1), create_edge_index(g2)
            adj_list1, adj_list2 = \
                create_adj_set(g1), create_adj_set(g2)
            nn_map = {}
            bidomains, abidomains, ubidomains = \
                self._update_bidomains(g1, g2, nn_map, None, None)
            MCS_size_UB = self.calc_bound_helper(ubidomains)

            # set up special input data
            degree_mat, mcsp_vec, sgw_mat, pca_mat = None, None, None, None
            if self.DQN_mode in ['fixedv_mcsp', 'fixedv_mcsprl'] or FLAGS.use_mcsp_policy:
                mcsp_vec = self.get_mcsp_vec(g1, g2)
            if self.DQN_mode == 'sgw':
                sgw_mat = sgw(g1, g2)
            if self.DQN_mode == 'pca':
                pca_mat = self.pca(pair, ins_g1, ins_g2)

            # create input data object
            torch.set_printoptions(profile="full")
            state_init = StateNode(ins_g1,
                                   ins_g2,
                                   nn_map,
                                   bidomains,
                                   abidomains,
                                   ubidomains,
                                   edge_index1,
                                   edge_index2,
                                   adj_list1,
                                   adj_list2,
                                   g1, g2,
                                   degree_mat,
                                   sgw_mat,
                                   pca_mat,
                                   cur_id,
                                   mcsp_vec,
                                   MCS_size_UB)
            state_init_list.append(state_init)
        assert len(pair_list) == len(state_init_list)
        return pair_list, state_init_list

    def compute_ins(self, g1, g2, ins, offset):
        M, N = g1.number_of_nodes(), g2.number_of_nodes()
        ins_g1, ins_g2 = \
            ins[offset:offset + M], ins[offset + M:offset + N + M]
        offset += (N + M)  # used for grabbing the right input embeddings
        return ins_g1, ins_g2, offset

    def _tgt_net_sync_itr(self, forward_mode):
        valid_mode = forward_mode in \
                     [PRETRAIN_MODE,
                      IMITATION_MODE,
                      TRAIN_MODE]
        valid_iteration = self.train_counter % self.sync_target_frames == 0
        return valid_mode and valid_iteration

    def _sync_tgt_networks(self):
        if not FLAGS.a2c_networks:
            self.dqn_tgt.load_state_dict(self.dqn.state_dict())

    def get_mcsp_vec(self, g1, g2):
        deg_vec_g1 = np.array(list(g1.degree[j] for j in range(g1.number_of_nodes())))
        deg_vec_g2 = np.array(list(g2.degree[j] for j in range(g2.number_of_nodes())))
        self.lgrade = deg_vec_g1 / (np.max(deg_vec_g1) + 2)
        self.rgrade = deg_vec_g2 / (np.max(deg_vec_g2) + 2)
        mcsp_vec = McspVec(deg_vec_g1, deg_vec_g2)
        return mcsp_vec


    ##########################################################
    # Utility functions (forward single tree procedure)
    ##########################################################
    def sample_search_stack(self, search_stack, incumbent, since_last_update_count):
        if self.regret_iters is None:
            method = 'stack'
        else:
            if since_last_update_count > self.regret_iters:
                method = 'heap'
            else:
                method = 'stack'

        cur_state, promise = search_stack.pop_task(method)

        if self.is_better_solution(cur_state, incumbent):
            since_last_update_count = 0
        else:
            if method == 'heap':
                since_last_update_count = -(len(incumbent) - len(cur_state.nn_map))
            elif method == 'stack':
                since_last_update_count += 1
            else:
                assert False

        return cur_state, promise, since_last_update_count

    def is_better_solution(self, cur_state, incumbent):
        return len(cur_state.nn_map) > len(incumbent)

    def exit_condition(self, recursion_count, timer, since_last_update_count):
        # exit search if recursion threshold
        recursion_thresh = \
            self.recursion_threshold is not None \
            and (recursion_count > self.recursion_threshold)
        timout_thresh = \
            self.total_runtime is not None \
            and (timer.get_duration() > self.total_runtime)
        return recursion_thresh or timout_thresh

    def prune_condition(self, cur_state, action_space_data, incumbent):
        # compute bound
        bound = self.calc_bound(cur_state)

        # check prune conditions
        empty_action_space = len(action_space_data.action_space[0]) == 0
        bnb_condition = len(cur_state.nn_map) + bound <= len(incumbent)

        return empty_action_space or ((not self.no_pruning) and bnb_condition)

    def post_process(self, search_tree, incumbent_list):
        EMBEDDING_SAVER.clear()
        if self.training:
            search_tree.assign_v_search_tree(self.reward_calculator.discount)
            if FLAGS.val_debug:
                search_tree.associate_q_pred_true_with_node()
        else:
            if FLAGS.logging == 'end':
                incumbent_end, recursion_iter_end, time_end = incumbent_list[-1]
                saver.log_info('=========================')
                saver.log_info(f'length of largest incumbent: {len(incumbent_end)}')
                saver.log_info(f'iteration at end: {recursion_iter_end}')
                saver.log_info(f'time at end: {time_end}')
                saver.log_info('=========================')
            elif FLAGS.logging == 'all':
                saver.log_list_of_lists_to_csv(incumbent_list, 'incumbents.csv')
            else:
                assert False

    def search_tree2buffer_entry_list(self, search_tree, pair):
        if self.training:
            g1, g2 = pair.g1.get_nxgraph(), pair.g2.get_nxgraph()
            if FLAGS.exclude_root:
                buffer_entry_list = [
                    BufferEntry(edge, g1, g2, search_tree)
                    for edge in search_tree.edges if edge.state_prev.action_prev is not None
                ]
            else:
                buffer_entry_list = [
                    BufferEntry(edge, g1, g2, search_tree)
                    for edge in search_tree.edges
                ]
        else:
            buffer_entry_list = None

        return buffer_entry_list

    def calc_bound(self, state, exhaust_revisited_nodes=True):
        if exhaust_revisited_nodes:
            bds_unexhausted = state.get_unexhausted_bidomains()
        else:
            bds_unexhausted = state.bidomains
        return self.calc_bound_helper(bds_unexhausted)

    def calc_bound_helper(self, bds):
        bound = 0
        for bd in bds:
            bound += min(len(bd.left), len(bd.right))
        return bound

    ##########################################################
    # Utility functions (forward single edge procedure)
    ##########################################################
    def compute_q_vec(self, state, action_space_data):
        # estimate the q values
        if len(action_space_data.action_space[0]) > 1 or FLAGS.plot_final_tree:
            # we want all q_pred if we are plotting tree!
            q_vec = \
                self._Q_network(
                    state, action_space_data,
                    tgt_network=False, detach_in_chunking_stage=True)
        else:
            q_vec = torch.ones(1)
        return q_vec


    def _N_network(self, state, action_space_data,
                   tgt_network=False, is_first_edge=False,
                   detach_in_chunking_stage=False,
                   critic_network=False, recursion_count=None
                   ):
        # create fake dqn input
        if action_space_data is None:
            fake_action_space_data = self._get_empty_action_space_data(state)
        else:
            fake_action_space_data = action_space_data
        dqn_input = \
            DQNInput(state, fake_action_space_data, self.restore_bidomains)

        # run dqn to get g scores
        # NOTE:
        #   because g_scores are per node NOT per edge,
        #   get_g_scores does NOT look at action_space...
        #   it ALWAYS computes g_scores for ALL nodes!
        if tgt_network:
            g_scores = self.dqn_tgt.get_g_scores(dqn_input)
        else:
            g_scores = self.dqn.get_g_scores(dqn_input)

        if detach_in_chunking_stage:
            g1_scores, g2_scores = g_scores
            g_scores = g1_scores.detach(), g2_scores.detach()

        # if not tup:
        #     g_scores = torch.cat(tuple(g_scores),dim=0)

        return g_scores

    def _Q_network(self, state, action_space_data, tgt_network=False, detach_in_chunking_stage=False):
        # unpack inputs
        if self.DQN_mode == 'pca':
            q_vec = state.pca_mat[action_space_data.action_space[:2]].view(-1, 1)
        elif self.DQN_mode == 'sgw':
            q_vec = torch.tensor(state.sgw_mat[action_space_data.action_space[:2]]).view(-1, 1)
        elif self.DQN_mode == 'rand':
            q_vec = state.degree_mat[action_space_data.action_space[:2]].view(-1, 1)
            q_vec = torch.rand_like(q_vec)
        elif 'fixedv' in self.DQN_mode:
            mode = self.DQN_mode.split('_')[-1]
            q_vec = self.fixed_v(action_space_data, state, mode)
        else:
            dqn_input = DQNInput(state, action_space_data, self.restore_bidomains)
            if tgt_network:
                q_vec = self.dqn_tgt(dqn_input, detach_in_chunking_stage)
            else:
                q_vec = self.dqn(dqn_input, detach_in_chunking_stage)
        return q_vec

    def fixed_v(self, action_space_data, state, mode):
        mcsp_vec = state.mcsp_vec
        last_v = state.last_v

        if mode == 'mcsp':
            lvec = mcsp_vec.ldeg
            rvec = mcsp_vec.rdeg
        elif mode == 'mcsprl':
            lvec = self.lgrade
            rvec = self.rgrade
        else:
            assert False

        lactions = np.array(action_space_data.action_space[0])
        ractions = np.array(action_space_data.action_space[1])

        q_vec = np.zeros(len(lactions))
        large_num = len(q_vec) + 1
        if (
                last_v is None or
                last_v not in lactions
        ):
            # TODO: what does argmax return?
            argmax, = np.where(lvec[lactions] == lvec[lactions].max())
            # print('if', len(argmax))
        else:
            argmax, = np.where(lactions == last_v)
            # print('el', len(argmax))
        indices, = np.where(lactions == np.min(lactions[argmax]))
        q_vec[indices] += 2 * large_num
        argmax, = np.where(rvec[ractions] == rvec[ractions].max())
        indices, = np.where(ractions == np.min(ractions[argmax]))
        q_vec[indices] += large_num

        return torch.tensor(q_vec)

    def _compute_eps(self):
        eps = max(self.Q_eps_end, self.Q_eps_start -
                  self.train_counter * self.Q_eps_dec_each_iter)
        return eps

    def _rank_actions_from_q_vec(self, q_vec, state, action_space_data):
        if FLAGS.use_mcsp_policy:
            q_vec = self.fixed_v(action_space_data, state, 'mcsp')

        # compute epsilon (eps-greedy)
        if self.training:
            eps = self._compute_eps()
        else:
            if self.eps_testing:
                eps = self.Q_eps_end
            else:
                eps = -1

        # epsilon greedy policy
        q_vec_idx_argmax = torch.argmax(q_vec, dim=0)
        if random.random() < eps or FLAGS.randQ:
            q_vec_idx = int(random.random() * q_vec.size(0))
        else:
            q_vec_idx = q_vec_idx_argmax
        is_q_vec_idx_argmax = q_vec_idx == q_vec_idx_argmax

        action = (action_space_data.action_space[0][q_vec_idx],
                  action_space_data.action_space[1][q_vec_idx])

        return action, q_vec_idx, is_q_vec_idx_argmax

    def _environment(self, state, action, q_vec_idx, recursion_count=None, real_update=False):

        if FLAGS.time_analysis:
            timer_env = OurTimer()
            timer_env.time_and_clear(f'environment starts')

        nn_map = deepcopy(state.nn_map)
        exhausted_v, exhausted_w = state.exhausted_v, state.exhausted_w
        if 'mcsp' in self.DQN_mode:
            pruned_actions = state.pruned_actions
        else:
            pruned_actions = None

        if FLAGS.time_analysis:
            timer_env.time_and_clear(f'environment deepcopy')

        bidomains_in = state.get_unexhausted_bidomains()

        if FLAGS.time_analysis:
            timer_env.time_and_clear(f'environment get_pruned_bidomains')

        v, w = action

        # apply action
        nn_map[v] = w

        # get next state
        g1, g2 = state.g1, state.g2
        bidomains_out, abidomains, ubidomains = \
            self._update_bidomains(g1, g2, nn_map, action, bidomains_in)

        # ##########################
        # # TODO: logging purposes
        # if real_update:
        #     bd_in_size = sum([min(len(bd.left), len(bd.right)) for bd in bidomains_in])
        #     bd_out_size = sum([min(len(bd.left), len(bd.right)) for bd in bidomains_out])
        #     self.logging_bd_size.extend([bd_in_size, bd_out_size])
        # ##########################

        if FLAGS.time_analysis:
            timer_env.time_and_clear(f'environment _update_bidomains')

        MCS_size_UB = len(nn_map) + self.calc_bound_helper(bidomains_out)

        # make new state node
        state.last_v = v
        next_state = StateNode(state.ins_g1,
                               state.ins_g2,
                               nn_map,
                               bidomains_out,
                               abidomains,
                               ubidomains,
                               state.edge_index1,
                               state.edge_index2,
                               state.adj_list1,
                               state.adj_list2,
                               state.g1, state.g2,
                               state.degree_mat,
                               state.sgw_mat,
                               state.pca_mat,
                               state.cur_id,
                               state.mcsp_vec,
                               MCS_size_UB,
                               tree_depth=state.tree_depth + 1,
                               num_steps=state.num_steps + 1)
        if FLAGS.time_analysis:
            timer_env.time_and_clear(f'environment _init StateNode')

        reward = self.reward_calculator.compute_reward(v,w,g1,g2,state,next_state)

        action_edge = ActionEdge(
            action, q_vec_idx, reward,
            deepcopy(pruned_actions),
            deepcopy(exhausted_v),
            deepcopy(exhausted_w))

        if FLAGS.time_analysis:
            timer_env.time_and_clear(f'environment end')
            timer_env.print_durations_log()

        return action_edge, next_state

    def _update_bidomains(self, g1, g2, nn_map, action, bidomains):

        timer = None
        if FLAGS.time_analysis:
            timer = OurTimer()
            timer.time_and_clear(f'_update_bidomains starts')

        if len(nn_map) == 0:
            # TODO: form new bidomains for natts graphs
            # add code here
            assert action is None
            assert bidomains is None

            if 'fuzzy_matching' in FLAGS.reward_calculator_mode:
                natts = []
            else:
                natts = FLAGS.node_feats_for_mcs
            # create a new bidomain for each unique natts
            natts2bd = {}

            if FLAGS.time_analysis:
                timer = OurTimer()
                timer.time_and_clear(f'_update_bidomains nn_map == 0 setup')

            for v in range(g1.number_of_nodes()):
                key = tuple([g1.nodes[v][natt] for natt in natts])
                if key not in natts2bd:
                    natts2bd[key] = Bidomain({v}, set(), False)
                else:
                    natts2bd[key].left.add(v)

            if FLAGS.time_analysis:
                timer = OurTimer()
                timer.time_and_clear(f'_update_bidomains nn_map == 0 for loop one')

            for w in range(g2.number_of_nodes()):
                key = tuple([g2.nodes[w][natt] for natt in natts])
                if key not in natts2bd:
                    natts2bd[key] = Bidomain(set(), {w}, False)
                else:
                    natts2bd[key].right.add(w)

            if FLAGS.time_analysis:
                timer = OurTimer()
                timer.time_and_clear(f'_update_bidomains nn_map == 0 for loop two')

            new_bidomains, abidomains, ubidomains = [], [], []
            for bidomain in natts2bd.values():
                if len(bidomain.left) > 0 and len(bidomain.right) > 0:
                    new_bidomains.append(bidomain)
                    ubidomains.append(bidomain)

            if FLAGS.time_analysis:
                timer = OurTimer()
                timer.time_and_clear(f'_update_bidomains nn_map == 0 for loop three')
            # bd_left = set(range(g1.number_of_nodes()))
            # bd_right = set(range(g2.number_of_nodes()))
            # new_bidomains = [Bidomain(bd_left, bd_right, False)]
        else:
            i, j = action
            neighborhood_i = set(g1.neighbors(i))
            neighborhood_j = set(g2.neighbors(j))

            new_bidomains, abidomains, ubidomains = [], [], []
            nn_map_keys = set(nn_map.keys())
            nn_map_vals = set(nn_map.values())

            if FLAGS.time_analysis:
                timer.time_and_clear(f'_update_bidomains setup')

            for k, bidomain in enumerate(bidomains):
                # TODO: set operation implicitly does deepcopy()
                bd_left_disconnected = bidomain.left - neighborhood_i - nn_map_keys
                bd_right_disconnected = bidomain.right - neighborhood_j - nn_map_vals
                bd_left_connected = bidomain.left - bd_left_disconnected - nn_map_keys
                bd_right_connected = bidomain.right - bd_right_disconnected - nn_map_vals



                if FLAGS.time_analysis:
                    timer.time_and_clear(f'_{k} update_bidomains set minus'
                                         f' {len(bidomain.left)} {len(neighborhood_i)} '
                                         f' {len(nn_map_keys)} {len(bd_left_disconnected)} '
                                         f' {bidomain.is_adj}')

                bidomain_connected = Bidomain(bd_left_connected, bd_right_connected, True)
                bidomain_disconnected = Bidomain(bd_left_disconnected, bd_right_disconnected,
                                                 bidomain.is_adj)

                if FLAGS.time_analysis:
                    timer.time_and_clear(f'_{k} update_bidomains Bidomain()')

                if len(bidomain_connected) > 0:
                    new_bidomains.append(bidomain_connected)
                    abidomains.append(bidomain_connected)
                if len(bidomain_disconnected) > 0:
                    new_bidomains.append(bidomain_disconnected)
                    ubidomains.append(bidomain_disconnected)

                if FLAGS.time_analysis:
                    timer.time_and_clear(f'_{k} update_bidomains []')


        if FLAGS.time_analysis:
            timer.time_and_clear(f'update_bidomains end')
            timer.print_durations_log()

        return new_bidomains, abidomains, ubidomains

    def find_promise(self, state, next_state):
        promise = state.get_adjacent_action_space_size()
        promise_new_state = next_state.get_adjacent_action_space_size()
        promise, promise_new_state = -promise, -promise_new_state
        promise_tuple = (promise, promise_new_state)
        return promise_tuple

    def get_is_mcsp(self):
        return self.DQN_mode in ['mcsp_degree', 'fixedv_mcspv2', 'fixedv_mcsprl']

    ##########################################################
    # Loss Function
    ##########################################################
    def _loss_wrapper(self, forward_mode):
        if forward_mode == PRETRAIN_MODE:
            loss = self._loss(self.buffer.sample(self.sample_size))
            self.buffer.empty()
        elif forward_mode in [IMITATION_MODE, TRAIN_MODE]:
            if self.train_counter < self.buffer_start_iter:
                loss = None
            else:
                loss = self._loss(self.buffer.sample(self.sample_size))
        else:
            assert False

        return loss

    def _loss(self, buffer_entry_list):
        assert not self.get_is_mcsp()
        if len(buffer_entry_list) == 0:
            # edge case 1: we are still buffering
            loss = None
        else:
            loss = torch.tensor(0.0, device=FLAGS.device)
            for buffer_entry in buffer_entry_list:
                self._process_buffer_entry(buffer_entry)
                loss += self._batched_loss([buffer_entry])
            loss /= len(buffer_entry_list)
        return loss

    def _batched_loss(self, buffer_entry_list):
        assert len(buffer_entry_list) != 0
        q_vec_pred, q_vec_true = [], []
        for buffer_entry in buffer_entry_list:
            q_pred, q_true, state = \
                self._get_pred_and_true_q(buffer_entry)
            q_vec_pred.append(q_pred)
            q_vec_true.append(q_true)

        q_vec_pred = torch.stack(tuple(q_vec_pred)).view(-1)
        q_vec_true = torch.stack(tuple(q_vec_true)).view(-1)
        loss = torch.sum(self.loss(q_vec_pred, q_vec_true))

        self._log_loss(
            q_vec_pred, q_vec_true, buffer_entry_list[0].edge.state_prev, loss)

        return loss

    def _process_buffer_entry(self, buffer_entry):
        edge = buffer_entry.edge
        state = edge.state_prev
        next_state = edge.state_next
        self._push_edge_search_info_to_state(state, edge, next_state)

    def _push_edge_search_info_to_state(self, state, edge, next_state):
        state.pruned_actions, state.exhausted_v, state.exhausted_w = \
            edge.pruned_actions, edge.exhausted_v, edge.exhausted_w

        next_state.pruned_actions, next_state.exhausted_v, next_state.exhausted_w = \
            DoubleDict(), set(), set()

    def _get_pred_and_true_q(self, buffer_entry):
        EMBEDDING_SAVER.clear()
        g1, g2 = buffer_entry.g1, buffer_entry.g2
        edge = buffer_entry.edge
        state = edge.state_prev
        next_state = edge.state_next

        # compute q pred
        next_action_space_data = \
            self.get_action_space_data_wrapper(next_state, is_mcsp=self.get_is_mcsp())
        v, w = edge.action
        action_space_data = self._get_empty_action_space_data(state)
        action_space_data.filter_action_space_data(v,w)

        q_pred = \
            torch.squeeze(
                self._Q_network(state, action_space_data, tgt_network=False)
            )
        EMBEDDING_SAVER.clear()

        # compute q true
        reward = \
            torch.tensor(
                self.reward_calculator.compute_reward_batch(
                    action_space_data.action_space, g1, g2, state, next_state),
                device=FLAGS.device
            ).type(torch.cuda.FloatTensor)
        q_next = \
            self._get_q_next(next_state, next_action_space_data)
        q_true = (reward + self.reward_calculator.discount * q_next).detach()
        EMBEDDING_SAVER.clear()

        return q_pred, q_true, state

    def _compute_tgt_q_max(self, next_state, next_action_space_data):
        is_empty_action_space = len(next_action_space_data.action_space[0]) == 0
        if is_empty_action_space:
            q_max = self.create_FloatTensor(0.0)
        else:
            q_vec = \
                self._Q_network(
                    next_state, next_action_space_data,
                    tgt_network=True, detach_in_chunking_stage=True)
            q_max = torch.max(q_vec.to(FLAGS.device).detach())

        return q_max

    def get_cum_reward(self, start_state, end_state, num_steps):
        discount = self.reward_calculator.discount ** num_steps
        cum_reward = \
            start_state.v_search_tree - discount*end_state.v_search_tree
        return cum_reward

    def _get_q_next(self, next_state, next_action_space_data):
        assert 'clamp' not in self.q_signal
        if 'fitted-tgt' in self.q_signal:
            # from state si, get the list (si, si+1, si+2, ..., sN)
            cur_end_state = next_state
            end_state_li = [next_state]
            while len(cur_end_state.action_next_list) > 0:
                if self.search_path:
                    assert len(cur_end_state.action_next_list) == 1
                    idx = 0
                else:
                    idx = int(self.seed.random() * len(cur_end_state.action_next_list))
                next_next_state = cur_end_state.action_next_list[idx].state_next
                end_state_li.append(next_next_state)
                cur_end_state = next_next_state

            # store as a list of (cum_reward, end_state, num_steps)
            if 'random-path' in self.q_signal:
                num_steps = int(self.seed.random() * len(end_state_li))
                end_state = end_state_li[num_steps]
                cum_reward = \
                    self.get_cum_reward(next_state, end_state, num_steps)
                cum_reward_end_state_li = [(cum_reward, end_state, num_steps)]
            elif 'leaf' in self.q_signal:
                num_steps = len(end_state_li) - 1
                end_state = end_state_li[num_steps]
                cum_reward = \
                    self.get_cum_reward(next_state, end_state, num_steps)
                cum_reward_end_state_li = [(cum_reward, end_state, num_steps)]
            else:
                assert False

            q_max = 0.0
            for cum_reward, end_state, num_steps in cum_reward_end_state_li:
                cur_next_state_action_space = \
                    self.get_action_space_data_wrapper(end_state, is_mcsp=self.get_is_mcsp())
                q_max_tgt = \
                    self._compute_tgt_q_max(
                        end_state, cur_next_state_action_space)
                discount_factor = self.reward_calculator.discount ** num_steps
                q_max += cum_reward + discount_factor * q_max_tgt
            q_max /= len(cum_reward_end_state_li)
        elif 'vanilla-tgt' in self.q_signal:
            # compute q_next via DQN
            q_max = self._compute_tgt_q_max(next_state, next_action_space_data)
        elif self.q_signal == 'LB':
            q_max = self.create_FloatTensor(next_state.v_search_tree)
        else:
            print(self.q_signal)
            assert False

        q_next = q_max.detach()
        return q_next

    def create_FloatTensor(self, li, requires_grad=False):
        if FLAGS.device == 'cpu':
            tsr = torch.tensor(li, requires_grad=requires_grad, device=FLAGS.device).type(
                torch.FloatTensor)
            # tsr = torch.FloatTensor(li, requires_grad=requires_grad, device=FLAGS.device)
        else:
            tsr = torch.tensor(li, requires_grad=requires_grad, device=FLAGS.device).type(
                torch.cuda.FloatTensor)
            # tsr = torch.cuda.FloatTensor(li, requires_grad=requires_grad, device=FLAGS.device)
        return tsr

    def _log_loss(self, q_pred, q_true, state, loss):
        # logging
        if self.debug_train_iter_counter < self.debug_first_train_iters:
            saver.log_info(
                '\npred: {}\n' \
                'true: {}'.format(
                    ','.join(["{0:0.2f}".format(i) for i in q_pred.tolist()]),
                    ','.join(["{0:0.2f}".format(i) for i in q_true.tolist()]))
            )
            saver.log_info(
                'g1 size {} ' \
                'g2 size {} ' \
                'nn_map size {} ' \
                'loss_iter {:.3f} ' \
                'buffer size {} ' \
                'pair_id {}'.format(
                    state.g1.number_of_nodes(), state.g2.number_of_nodes(),
                    len(state.nn_map), loss.item(), len(self.buffer),
                    (state.g1.graph['gid'], state.g2.graph['gid']))
            )

        self.debug_train_iter_counter += 1
